"""
Matrix-form assignment model using HiGHS.

The mathematical model is

    Stage 1:
        maximize    ones^T z
        subject to  A z <= b
                    z in {0,1}^n

    Stage 2:
        minimize    z^T Q z
        subject to  A z <= b
                    ones^T z == T*
                    z in {0,1}^n

where

    Q = P^T P

and P maps the assignment vector z to the number of tasks assigned
to each capacitor.

Thus

    P z = [load_0, load_1, ..., load_{K-1}]

and

    z^T P^T P z = sum_j load_j^2.

Stage 1 guarantees the maximum number of executable tasks.
Stage 2 distributes those tasks as evenly as possible across capacitors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from pyscipopt import Model, quicksum
from pyscipopt.recipes.nonlinear import set_nonlinear_objective

from .models import (
    ConstantSource,
    SineSource,
    BonitoSource,
    Capacitor,
    CapacitorStorageSim,
    CapacitorStorageSimConfig,
    LeacsCapacitor,
    SMSink,
    Event,
)
from .state_machines import Task, init_MTRSinkSM

_TASK_IDX = {"measure": 0, "analyze": 1, "tx": 2, "rx": 3}


@dataclass
class AssignmentModel:
    """
    Solver-independent sparse representation of the assignment problem.
    """

    A: sp.csr_matrix
    b: np.ndarray

    # Primary objective (Stage 1):
    #
    # maximize objective @ z = 1^T M1 z
    #
    # This is the number of assigned tasks.
    objective: np.ndarray

    # M5 maps z -> tasks-per-capacitor (Stage 2 loads).
    #
    #     loads = M5 @ z
    #
    load_operator: sp.csr_matrix

    # Feasibility mask f = [0_K; f_1; ...; f_N] and assignment costs
    # d = [0_K; d_1; ...; d_N], both laid out like z.
    f: np.ndarray
    d: np.ndarray

    N: int
    K: int
    M: int

    E_allowed: np.ndarray | None = None
    energy_constraint: bool = True

    @property
    def c(self) -> np.ndarray:
        """Backward-compatible alias for d."""
        return self.d


def assignment_costs(
    task_power: np.ndarray,
    task_duration: np.ndarray,
    caps: list[Capacitor],
    visits: np.ndarray | None = None,
    switching: np.ndarray | None = None,
    kappa: float = 0.01,
    I0: float = 3e-6,
) -> np.ndarray:
    """
    Task-to-capacitor assignment costs, Eq. (costs), shape (N, K):

        d_ij = n_i (E_i + l_ij + sigma_j)

    with
        E_i    = task_power_i * task_duration_i                  (J)
        l_ij   = tau_i V_j max(kappa C_j V_j, I0)                (J), Eq. (leak-bound)
        n_i    = visits_i (default 1)
        sigma_j = switching_j (default 0)

    V_j is cap.voltage, the voltage at planning time (V_j^current in the
    paper). Leakage current falls as the capacitor discharges, so using
    the planning-time voltage overestimates leakage over the task.
    """

    task_power = np.asarray(task_power, dtype=float)
    task_duration = np.asarray(task_duration, dtype=float)

    N = task_power.shape[0]
    K = len(caps)

    if task_duration.shape != (N,):
        raise ValueError(f"task_duration must have shape ({N},)... has {task_duration.shape}")

    if visits is None:
        visits = np.ones(N)
    visits = np.asarray(visits, dtype=float)
    if visits.shape != (N,):
        raise ValueError(f"visits must have shape ({N},)... has {visits.shape}")

    if switching is None:
        switching = np.zeros(K)
    switching = np.asarray(switching, dtype=float)
    if switching.shape != (K,):
        raise ValueError(f"switching must have shape ({K},)... has {switching.shape}")

    # E_i: energy of one execution of task i.
    E = task_power * task_duration                                   # (N,)

    # Leakage power bound of capacitor j at its planning-time voltage.
    leak_power = np.array(
        [cap.voltage * max(kappa * cap.farads * cap.voltage, I0) for cap in caps]
    )                                                                # (K,)

    # l_ij = tau_i * leak_power_j
    l = task_duration[:, np.newaxis] * leak_power[np.newaxis, :]     # (N, K)

    return visits[:, np.newaxis] * (E[:, np.newaxis] + l + switching[np.newaxis, :])


def expected_visits(
    transition: np.ndarray,
    start: int,
    depth: int,
) -> np.ndarray:
    """
    One choice of n_i: the expected number of executions of each task in
    the D + 1 tasks starting at `start` (the task running at the replan
    plus the next `depth`), for a Markov chain with row-stochastic
    `transition`. These are expectations; for a worst-case certificate,
    use the maximum visit count over trajectories instead.
    """

    transition = np.asarray(transition, dtype=float)

    dist = np.zeros(transition.shape[0])
    dist[start] = 1.0

    visits = np.zeros_like(dist)

    for _ in range(depth + 1):
        visits += dist
        dist = dist @ transition

    return visits


def build_operators(N: int, K: int):
    """
    Construct M1, ..., M5 as in the paper, all acting on
    z = [x; y_1; ...; y_N] in R^{K(N+1)} (task-major y block):

        M1 = [[0_KxK, 0], [0, I_KN]]               extracts y
        M2 = [[I_K, 0], [0, 0_KNxKN]]              extracts x
        M3 = [[0_KxK, 0], [1_N (x) I_K, 0_KNxKN]]  replicates x into each y_i
        M4 = [0_NxK, I_N (x) 1_K^T]                sums y_i per task
        M5 = [0_KxK, 1_N^T (x) I_K]                counts tasks per capacitor
    """

    KN = K * N

    M1 = sp.block_diag(
        (sp.csr_matrix((K, K)), sp.eye(KN, format="csr")),
        format="csr",
    )

    M2 = sp.block_diag(
        (sp.eye(K, format="csr"), sp.csr_matrix((KN, KN))),
        format="csr",
    )

    M3 = sp.bmat(
        [
            [sp.csr_matrix((K, K)), sp.csr_matrix((K, KN))],
            [sp.kron(np.ones((N, 1)), sp.eye(K), format="csr"), sp.csr_matrix((KN, KN))],
        ],
        format="csr",
    )

    M4 = sp.hstack(
        [sp.csr_matrix((N, K)), sp.kron(sp.eye(N), np.ones((1, K)), format="csr")],
        format="csr",
    )

    M5 = sp.hstack(
        [sp.csr_matrix((K, K)), sp.kron(np.ones((1, N)), sp.eye(K), format="csr")],
        format="csr",
    )

    return M1, M2, M3, M4, M5


def build_model(
    N: int,
    K: int,
    M: int,
    caps: list[Capacitor],
    energy_costs: np.ndarray,
    energy_constraint: bool = True,
    feasibility_costs: np.ndarray | None = None,
) -> AssignmentModel:
    """
    Build the sparse matrix representation of Eq. (stage1-cons).

    energy_costs:
        d_ij of Eq. (costs), shape (N, K). Build it with
        assignment_costs(...). With visits = 1 and no switching cost this
        is E_i + l_ij, i.e. what you passed before.

    energy_constraint:
        True   include the blue per-capacitor energy row (paper default).
        False  red feasibility row only, i.e. the LEACS-style formulation.

    feasibility_costs:
        Optional (N, K) costs for Eq. (f) only. None (default) uses d_ij,
        as the paper is written. With fractional visit counts (n_i < 1),
        pass the one-execution costs, assignment_costs(..., visits=None),
        so that f_ij = 1 still means one run of u_i fits in c_j.
    """

    d_mat = np.asarray(energy_costs, dtype=float)

    if d_mat.shape != (N, K):
        raise ValueError(f"energy_costs must have shape ({N},{K})... has {d_mat.shape}")

    # ------------------------------------------------------------
    # Eq. (e-allowed): usable energy in each capacitor.
    # ------------------------------------------------------------

    E_allowed = np.array(
        [cap.farads * (cap.voltage**2 - cap.v_min**2) / 2 for cap in caps],
        dtype=float,
    )

    # ------------------------------------------------------------
    # Eq. (f): f_ij = 1 iff d_ij <= E_allowed_j. Shape (N, K).
    # ------------------------------------------------------------

    f_costs = d_mat if feasibility_costs is None else np.asarray(feasibility_costs, dtype=float)

    if f_costs.shape != (N, K):
        raise ValueError(f"feasibility_costs must have shape ({N},{K})... has {f_costs.shape}")

    feasibility = f_costs <= E_allowed[np.newaxis, :]

    stranded = np.flatnonzero(~feasibility.any(axis=1))
    if stranded.size:
       raise ValueError(f"tasks {stranded.tolist()} fit no capacitor")

    M1, M2, M3, M4, M5 = build_operators(N, K)

    # ------------------------------------------------------------
    # Block vectors laid out like z = [x; y_1; ...; y_N]. An (N, K)
    # array flattens task-major with reshape(-1), matching y.
    # ------------------------------------------------------------

    d = np.concatenate([np.zeros(K), d_mat.reshape(-1)])
    f = np.concatenate([np.zeros(K), feasibility.reshape(-1)])

    # ------------------------------------------------------------
    # Eq. (stage1-obj): maximize 1^T M1 z.
    # ------------------------------------------------------------

    ones = np.ones(K * (N + 1))
    objective = np.asarray(ones @ M1).ravel()

    # ------------------------------------------------------------
    # Eq. (stage1-cons), row by row.
    # ------------------------------------------------------------

    # M4 M1 z - 1_N <= 0: each task on at most one capacitor.
    A_task1 = M4 @ M1
    b_task1 = np.ones(N)

    # M4 M1 z - 1_N >= 0: each task on at least one capacitor.
    A_task2 = -M4 @ M1
    b_task2 = -np.ones(N)

    # ^^ above two constraints make 1 equality constraint

    # (M1 - M3) z <= 0: y_ij <= x_j.
    A_supply_assignment = M1 - M3
    b_supply_assignment = np.zeros(K * (N + 1))

    # Red: M1 z - f <= 0: y_ij <= f_ij.
    A_feasibility = M1
    b_feasibility = f

    blocks_A = [A_task1, A_task2, A_supply_assignment, A_feasibility]
    blocks_b = [b_task1, b_task2, b_supply_assignment, b_feasibility]

    # blocks_A = [A_task1, A_supply_assignment, A_feasibility]
    # blocks_b = [b_task1, b_supply_assignment, b_feasibility]

    # Blue: M5 diag(d) z - diag(E_allowed) [I_K 0_{K x KN}] z <= 0_K.
    if energy_constraint:
        X = sp.hstack(
            [sp.eye(K, format="csr"), sp.csr_matrix((K, K * N))],
            format="csr",
        )
        A_energy = (M5 @ sp.diags(d)) - (sp.diags(E_allowed) @ X)

        blocks_A.append(A_energy)
        blocks_b.append(np.zeros(K))

    # 1^T M2 z - M <= 0: at most M capacitors.
    A_supply_count = sp.csr_matrix(np.asarray(ones @ M2).ravel())
    b_supply_count = np.array([M])

    blocks_A.append(A_supply_count)
    blocks_b.append(b_supply_count)

    A = sp.vstack(blocks_A, format="csr")
    b = np.concatenate(blocks_b)

    return AssignmentModel(
        A=A,
        b=b,
        objective=objective,
        load_operator=M5,
        f=f,
        d=d,
        N=N,
        K=K,
        M=M,
        E_allowed=E_allowed,
        energy_constraint=energy_constraint,
    )


def _scip_status(model: Model) -> str:
    """Convert SCIP status to a simple string."""

    status = model.getStatus()

    status_map = {
        "optimal": "OPTIMAL",
        "feasible": "FEASIBLE",
        "infeasible": "INFEASIBLE",
        "unbounded": "UNBOUNDED",
        "timelimit": "TIME_LIMIT",
        "gaplimit": "GAP_LIMIT",
        "nodelimit": "NODE_LIMIT",
        "memlimit": "MEMORY_LIMIT",
        "stallnodelimit": "STALL_NODE_LIMIT",
        "userinterrupt": "USER_INTERRUPT",
        "unknown": "UNKNOWN",
    }

    return status_map.get(
        str(status).lower(),
        str(status).upper().replace(" ", "_"),
    )


def _build_scip_model(
    A: sp.csr_matrix,
    b: np.ndarray,
    objective: np.ndarray,
    sense: str,
) -> tuple[Model, list]:
    """
    Build a binary MILP using SCIP.

    Objective:

        objective @ z

    Constraints:

        A z <= b

    Variables:

        z_j in {0, 1}

    Returns:

        (SCIP model, SCIP variable list)
    """

    A = A.tocsr()

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    objective = np.asarray(
        objective,
        dtype=np.float64,
    )

    n = A.shape[1]
    m = A.shape[0]

    if b.shape != (m,):
        raise ValueError(f"b has shape {b.shape}; expected {(m,)}")

    if objective.shape != (n,):
        raise ValueError(f"objective has shape {objective.shape}; expected {(n,)}")

    # ------------------------------------------------------------
    # SCIP model
    # ------------------------------------------------------------

    model = Model()

    # Keep SCIP quiet.
    model.hideOutput()

    # ------------------------------------------------------------
    # Binary variables
    # ------------------------------------------------------------

    z = [
        model.addVar(
            name=f"z_{j}",
            vtype="B",
        )
        for j in range(n)
    ]

    # ------------------------------------------------------------
    # Linear constraints:
    #
    #     A z <= b
    # ------------------------------------------------------------

    for i in range(m):
        start = A.indptr[i]
        end = A.indptr[i + 1]

        expr = quicksum(float(A.data[k]) * z[A.indices[k]] for k in range(start, end))

        model.addCons(
            expr <= float(b[i]),
            name=f"constraint_{i}",
        )

    # ------------------------------------------------------------
    # Linear objective
    # ------------------------------------------------------------

    objective_expr = quicksum(
        float(objective[j]) * z[j] for j in range(n) if objective[j] != 0.0
    )

    model.setObjective(
        objective_expr,
        sense,
    )

    return model, z


def _solution_array(
    model: Model,
    variables: list,
) -> np.ndarray:
    """Extract the current SCIP solution."""

    solution = model.getBestSol()

    if solution is None:
        return np.zeros(
            len(variables),
            dtype=float,
        )

    return np.asarray(
        [
            model.getSolVal(
                solution,
                var,
            )
            for var in variables
        ],
        dtype=float,
    )


# ----------------------------------------------------------------------
# Stage 1
# ----------------------------------------------------------------------


def solve_stage1_milp(
    model: AssignmentModel,
) -> dict:
    """
    Stage 1:

        maximize objective^T z

    subject to

        A z <= b
        z binary
    """

    start = time.perf_counter()

    scip, z = _build_scip_model(
        A=model.A,
        b=model.b,
        objective=model.objective,
        sense="maximize",
    )

    scip.optimize()

    status = _scip_status(scip)

    solution = _solution_array(
        scip,
        z,
    )

    objective_value = float(model.objective @ solution)

    # SCIP statistics.
    nodes = scip.getNNodes()

    wall_time_ms = (time.perf_counter() - start) * 1000.0

    return {
        "status": status,
        "objective": objective_value,
        "solution": solution,
        "wall_time_ms": wall_time_ms,
        # SCIP is not a simplex solver, so this does not
        # correspond to HiGHS's simplex_iteration_count.
        "iterations": None,
        "nodes": nodes,
    }


# ----------------------------------------------------------------------
# Stage 2
# ----------------------------------------------------------------------


def solve_stage2_miqp(
    model: AssignmentModel,
    optimal_tasks: int,
) -> dict:
    """
    Stage 2:

        minimize sum_i load_i^2

    subject to:

        A z <= b
        objective^T z == optimal_tasks
        z binary

    where:

        load = P z

    and therefore:

        sum_i load_i^2
            = ||P z||_2^2

    This is a convex binary MIQP.
    """

    start = time.perf_counter()

    A = model.A.tocsr()

    P = model.load_operator.tocsr()

    n = A.shape[1]
    # m = A.shape[0]

    if P.shape[1] != n:
        raise ValueError(f"load_operator has {P.shape[1]} columns; expected {n}")

    # ------------------------------------------------------------
    # Build SCIP model
    # ------------------------------------------------------------

    scip = Model()

    # Keep SCIP quiet.
    scip.hideOutput()

    # ------------------------------------------------------------
    # Binary variables
    # ------------------------------------------------------------

    z = [
        scip.addVar(
            name=f"z_{j}",
            vtype="B",
        )
        for j in range(n)
    ]

    # ------------------------------------------------------------
    # Original constraints:
    #
    #     A z <= b
    # ------------------------------------------------------------

    for i in range(A.shape[0]):
        start_idx = A.indptr[i]
        end_idx = A.indptr[i + 1]

        expr = quicksum(
            float(A.data[k]) * z[A.indices[k]] for k in range(start_idx, end_idx)
        )

        scip.addCons(
            expr <= float(model.b[i]),
            name=f"constraint_{i}",
        )

    # ------------------------------------------------------------
    # Preserve Stage 1 optimum:
    #
    #     objective^T z == optimal_tasks
    #
    # THIS IS THE IMPORTANT LEXICOGRAPHIC CONSTRAINT.
    # ------------------------------------------------------------

    primary_expr = quicksum(
        float(model.objective[j]) * z[j] for j in range(n) if model.objective[j] != 0.0
    )

    scip.addCons(
        primary_expr == float(optimal_tasks),
        name="stage1_optimum",
    )

    # ------------------------------------------------------------
    # Load expressions
    #
    #     load_i = sum_j P[i,j] z_j
    # ------------------------------------------------------------

    load_exprs = [
        quicksum(float(P[i, j]) * z[j] for j in range(n)) for i in range(P.shape[0])
    ]

    quadratic_objective = quicksum(load_expr * load_expr for load_expr in load_exprs)

    set_nonlinear_objective(
        scip,
        quadratic_objective,
        sense="minimize",
    )

    # print(
    #     "Stage 2:"
    #     f" vars={n},"
    #     f" rows={A.shape[0] + 1},"
    #     f" load_vars={P.shape[0]},"
    #     f" load_nnz={P.nnz},"
    #     f" optimal_tasks={optimal_tasks}"
    # )

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------

    scip.optimize()

    status = _scip_status(scip)

    # print(
    #     f"Stage 2 SCIP status: {status}"
    # )

    solution = _solution_array(
        scip,
        z,
    )

    # ------------------------------------------------------------
    # Recompute loads from the returned binary solution.
    #
    # This is deliberately done from P @ solution rather than
    # relying on SCIP's internal expression values.
    # ------------------------------------------------------------

    loads = np.asarray(
        P @ solution,
        dtype=np.float64,
    )

    load_objective = float(loads @ loads)

    primary_objective = float(model.objective @ solution)

    nodes = scip.getNNodes()

    wall_time_ms = (time.perf_counter() - start) * 1000.0

    return {
        "status": status,
        # Primary objective remains the number
        # of assigned tasks.
        "objective": primary_objective,
        "primary_objective": primary_objective,
        "load_objective": load_objective,
        "solution": solution,
        "capacitor_loads": loads,
        "max_capacitor_load": (float(np.max(loads)) if loads.size else 0.0),
        "wall_time_ms": wall_time_ms,
        "iterations": None,
        "nodes": nodes,
    }


# ----------------------------------------------------------------------
# Complete lexicographic solve
# ----------------------------------------------------------------------


def solve_assignment(
    model: AssignmentModel,
) -> dict:
    """
    Lexicographic assignment:

        Stage 1:
            maximize number of assigned tasks.

        Stage 2:
            among all Stage 1-optimal solutions,
            minimize sum of squared capacitor loads.
    """

    # ------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------

    stage1 = solve_stage1_milp(model)

    if stage1["status"] not in (
        "OPTIMAL",
        "FEASIBLE",
    ):
        print("returning stage 1; stage 1 not optimal or feasible")

        return stage1

    optimal_tasks = round(stage1["objective"])

    # print(f"Stage 1 optimal assignments: {optimal_tasks}")

    # ------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------

    stage2 = solve_stage2_miqp(
        model,
        optimal_tasks,
    )

    if stage2["status"] not in (
        "OPTIMAL",
        "FEASIBLE",
    ):
        print("returning stage 1; stage 2 not optimal or feasible")

        return stage1

    # ------------------------------------------------------------
    # Preserve Stage 1 metadata
    # ------------------------------------------------------------

    stage2["stage1_objective"] = stage1["objective"]

    stage2["stage1_wall_time_ms"] = stage1["wall_time_ms"]

    stage2["stage2_wall_time_ms"] = stage2["wall_time_ms"]

    stage2["wall_time_ms"] = stage1["wall_time_ms"] + stage2["wall_time_ms"]

    # print(
    #     "Stage 2 selected."
    #     f" loads={stage2['capacitor_loads']},"
    #     f" load_objective={stage2['load_objective']}"
    # )

    return stage2


class TaskAssigner:
    def __init__(
        self,
        caps,
        tasks,
        cap_limit,
        replan_depth,
        energy_constraint
    ):
        self.caps = caps
        self.tasks = tasks
        self.N = len(tasks)
        self.K = len(caps)
        self.M = cap_limit
        self.replan_depth = replan_depth
        self.energy_constraint = energy_constraint
        self.task_idx = {t.name: i for i, t in enumerate(tasks)}

    def presolve(self):
        # make list of fully-charged cap energies for all caps in self.caps.
        tmp = []
        for cap in self.caps:
            c = LeacsCapacitor(
                cap.farads,
                v_min=cap.v_min,
                v_max=cap.v_max,
            )
            c.voltage = cap.v_max
            tmp.append(c)

        energy_costs = [
            [(c.leakage * c.voltage + t.cost) * t.duration for c in tmp] for t in self.tasks
        ]
        model = build_model(
            N=self.N,
            K=self.K,
            M=self.M,
            caps=tmp,
            energy_costs=energy_costs,
            energy_constraint=self.energy_constraint
        )
        return solve_assignment(model)

    def assign(
        self,
        time,
    ):
        # c.leakage is current -> p = IV = 0.01*C*V^2
        # c.energy = 0.5 * v**2 * c
        #   ==> p_leak = 0.01 C * V**2 = c.energy / 0.01 / 2
        energy_costs = [
            [(c.leakage * c.voltage + t.cost) * t.duration for c in self.caps] for t in self.tasks
        ]

        model = build_model(
            N=self.N,
            K=self.K,
            M=self.M,
            caps=self.caps,
            energy_costs=energy_costs,
            energy_constraint=self.energy_constraint
        )

        return solve_assignment(model)

    def should_schedule(self, config=None):

        if config is None:
            return False

        if self.replan_depth <= 0:
            return False

        return not config.depth % self.replan_depth


class LeacSimConfig(CapacitorStorageSimConfig):
    def __init__(
        self,
        tasks,
        cap_limit,
        replan_depth,
        energy_constraint,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.cap_limit = cap_limit

        self.assignment = None

        self.assigner = TaskAssigner(
            self.caps,
            tasks,
            cap_limit,
            replan_depth,
            energy_constraint
        )

        result = self.assigner.presolve()
        self.assignment = result["solution"]

        # set initial sm params
        pairs = self._decode_assignment_result(self.assignment)
        self.sm = self.sink.sm
        self.sm.cap = self.caps[pairs[0][0]]
        self.sm.task_start = 0.0
        self.sm.time = 0.0

        self.last_substate = None
        self.depth = 0

        # get the last substate for all states
        # gets updated at every timestep
        self.last_substate_all = None


    def callback(self, time: float):

        event = None

        active_substate, child_substates = self.sm._get_task_state_id()

        if self.sm.time == 0.0:
            for i, cap in enumerate(self.caps):
                if self.assignment is not None:
                    # the following check evaluates any nonzero as True
                    if i in np.argwhere(self.assignment[:len(self.caps)]):
                        cap.connect(0)

            self.src.connect(0)
            self.sink.connect(1)

        self.sm.time = time

        # Apply previous assignment to the state machine.
        if self.assignment is not None:
            pairs = self._apply_assignment_to_SM(self.assignment)
            assigned_cap_idx = None
            child_cap_idx = []

            if active_substate is not None:
                if active_substate == 'sleep' and self.last_substate:
                    last_task = self.assigner.task_idx[self.last_substate]
                    for cap_idx, task_idx in pairs:
                        if task_idx == last_task:
                            child_cap_idx.append(cap_idx)
                elif active_substate == 'sleep' and not self.last_substate:
                    measure = self.assigner.task_idx['measure']
                    child_cap_idx.extend(c for c, t in pairs if t == measure)
                else:
                    for cap_idx, task_idx in pairs:
                        if task_idx == self.assigner.task_idx[active_substate]:
                            assigned_cap_idx = cap_idx
                        elif task_idx in child_substates:
                            child_cap_idx.append(cap_idx)

            if self.last_substate and active_substate != self.last_substate:
                self.depth += 1

            # Assigned capacitor -> sink.
            # All others -> source.
            for i, cap in enumerate(self.caps):
                if i == assigned_cap_idx:
                    cap.disconnect(0)
                    cap.connect(1)
                elif i in child_cap_idx:
                    cap.connect(0)
                    cap.disconnect(1)
                else:
                    cap.disconnect(0)   # unselected caps never sit on the charge line
                    cap.disconnect(1)

        # Solve a new assignment when scheduling
        # becomes possible.
        if self.assigner.should_schedule(self):
            try:
                result = self.assigner.assign(time)
            except ValueError:                      # some task fits no capacitor now
               result = None
            if result is not None and result["status"] in ("OPTIMAL", "FEASIBLE"):
               self.assignment = result["solution"]

        if active_substate != 'sleep':
            self.last_substate = active_substate

        # check if state is different from lsat timestep
        if active_substate != self.last_substate_all:
            # store event
            if active_substate == "sleep":
                event = Event(active_substate, time, ignore=True)
            else:
                event = Event(active_substate, time)
            # update last event
            self.last_substate_all = active_substate


        # run the state machine
        self._update_sink()

        return event

    def _decode_assignment_result(
        self,
        assignment,
    ):
        """
        Decode

            z = [x; y]

        into

            [(capacitor_index, task_index), ...].
        """

        assignment = np.asarray(
            assignment,
            dtype=float,
        )

        expected_size = self.assigner.K * (self.assigner.N + 1)

        if assignment.size != expected_size:
            raise ValueError(
                f"Invalid assignment size: "
                f"expected {expected_size}, "
                f"got {assignment.size}"
            )

        K = self.assigner.K
        N = self.assigner.N

        y = assignment[K:].reshape(
            N,
            K,
        )

        threshold = 0.5

        pairs = [
            (cap_idx, task_idx)
            for task_idx in range(N)
            for cap_idx in range(K)
            if y[task_idx, cap_idx] >= threshold
        ]
        return pairs

    def _apply_assignment_to_SM(
        self,
        assignment,
    ):
        pairs = self._decode_assignment_result(assignment)

        active_substate, _ = self.sm._get_task_state_id()
        if active_substate not in (None, 'sleep'):
            for cap_idx, task_idx in pairs:
                if task_idx == self.assigner.task_idx[active_substate]:
                    if self.caps[cap_idx].voltage > self.caps[cap_idx].v_min:
                        self.sm.cap = self.caps[cap_idx]

                        # print(
                        #     f"{self.sm.time:.6f}: "
                        #     f"{active_substate}, "
                        #     f"Cap{cap_idx} "
                        #     f"Volts:"
                        #     f"{self.sm.cap.voltage:.6f}"
                        # )

                    break

        elif self.sm.sleep.is_active:
            pass

        else:
            pass

        return pairs

    def _update_sink(self):
        self.sink.run_sm()


if __name__ == "__main__":
    M = 2
    replan_depth = 3
    energy_constraint = True

    CONST_VOLTAGE = 3.3

    cap_values = [
        6e-3,
        6e-4,
        6e-3,
    ]

    # src = ConstantSource(
    #     1,
    #     duration=2,
    #     dt=0.01,
    # )

    src = SineSource(
        1.70e-2,
        1.70e-2,
        0.50,
        0.00,
        duration=100,
        dt=0.01,
    )

    # src = BonitoSource(duration=3600, downsample=30000)

    caps = [
        LeacsCapacitor(
            c,
            v_min=1.7,
            v_max=3.2,
        )
        for c in cap_values
    ]

    tasks = [  # bookkeeping for LeacSimConfig; SMSink is hardcoded with identical Tasks
        Task(cost=11.68e-4 * CONST_VOLTAGE, duration=0.511, name="measure"),
        Task(cost=86.52e-4 * CONST_VOLTAGE, duration=0.285, name="tx"),
        Task(cost=20.03e-4 * CONST_VOLTAGE, duration=0.937, name="rx"),
    ]

    sink = SMSink(init_MTRSinkSM(caps[0]))

    config = LeacSimConfig(
        tasks,
        M,
        0,
        replan_depth,
        energy_constraint,
        src=src,
        caps=caps,
        sink=sink,
        p_lines=2
    )

    sim = CapacitorStorageSim(config)

    start = time.time()

    sim.run()

    print(f"runtime: {time.time() - start}")

    sim.plot()
    sim.show()
