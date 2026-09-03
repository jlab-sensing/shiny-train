"""
Matrix-form assignment model using HiGHS.

The mathematical model is

    Stage 1:
        maximize    objective^T z
        subject to  A z <= b
                    z in {0,1}^n

    Stage 2:
        minimize    z^T Q z
        subject to  A z <= b
                    objective^T z == T*
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
    Capacitor,
    CapacitorStorageSim,
    CapacitorStorageSimConfig,
    ConstantSource,
    SMSink,
)
from .state_machines import Task, init_SinkSM

_TASK_IDX = {"measure": 0, "tx": 1, "rx": 2}


@dataclass
class AssignmentModel:
    """
    Solver-independent sparse representation of the assignment problem.
    """

    A: sp.csr_matrix
    b: np.ndarray

    # Primary objective:
    #
    # maximize objective @ z
    #
    # This is the number of assigned tasks.
    objective: np.ndarray

    # P maps z -> tasks-per-capacitor.
    #
    #     loads = P @ z
    #
    load_operator: sp.csr_matrix

    f: np.ndarray
    c: np.ndarray
    leakage: np.ndarray

    N: int
    K: int
    M: int


def build_operators(N: int, K: int):
    """Construct M1, M2, M3, M4 and P as sparse matrices."""

    n = K * (N + 1)

    # ------------------------------------------------------------
    # M1 projects y from z = [x; y].
    # ------------------------------------------------------------

    M1 = sp.block_diag(
        (
            sp.csr_matrix((K, K)),
            sp.eye(K * N, format="csr"),
        ),
        format="csr",
    )

    # ------------------------------------------------------------
    # M2 projects x from z.
    # ------------------------------------------------------------

    M2 = sp.block_diag(
        (
            sp.eye(K, format="csr"),
            sp.csr_matrix((K * N, K * N)),
        ),
        format="csr",
    )

    # ------------------------------------------------------------
    # M3 replicates x into each assignment block.
    #
    # Shape: K(N+1) x K(N+1)
    # ------------------------------------------------------------

    rows = []
    cols = []
    data = []

    for i in range(N):
        row0 = K + i * K

        for j in range(K):
            rows.append(row0 + j)
            cols.append(j)
            data.append(1.0)

    M3 = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n, n),
    ).tocsr()

    # ------------------------------------------------------------
    # M4 = I_N kron 1_K^T
    #
    # Shape: N x KN
    #
    # Used to enforce:
    #
    #     sum_j y_ij <= 1
    # ------------------------------------------------------------

    M4 = sp.kron(
        sp.eye(N, format="csr"),
        np.ones((1, K)),
        format="csr",
    )

    # ------------------------------------------------------------
    # P maps z -> tasks-per-capacitor.
    #
    # z = [
    #     x_0, ..., x_K-1,
    #     y_00, ..., y_0,K-1,
    #     y_10, ..., y_1,K-1,
    #     ...
    # ]
    #
    # P @ z =
    #
    #     [
    #         sum_i y_i0,
    #         sum_i y_i1,
    #         ...
    #         sum_i y_i,K-1
    #     ]
    #
    # Since the y variables are task-major, this is
    #
    #     kron(1_N^T, I_K)
    # ------------------------------------------------------------

    P = sp.hstack(
        [
            sp.csr_matrix((K, K)),
            sp.kron(
                np.ones((1, N)),
                sp.eye(K, format="csr"),
                format="csr",
            ),
        ],
        format="csr",
    )

    return M1, M2, M3, M4, P


def build_model(
    N: int,
    K: int,
    M: int,
    caps: list[Capacitor],
    energy_costs: np.ndarray,
    leakage: np.ndarray,
    rewards: np.ndarray | None = None,
) -> AssignmentModel:
    """
    Build the sparse matrix representation of the assignment problem.
    """

    energy_costs = np.asarray(
        energy_costs,
        dtype=float,
    )

    leakage = np.asarray(
        leakage,
        dtype=float,
    )

    if energy_costs.shape != (N,):
        raise ValueError("energy_costs must have shape (N,)")

    if leakage.shape != (K,):
        raise ValueError("leakage must have shape (K,)")

    # ------------------------------------------------------------
    # Available energy in each capacitor.
    # ------------------------------------------------------------

    E_allowed = np.zeros(K, dtype=float)

    for i, cap in enumerate(caps):
        E_allowed[i] = cap.farads * (cap.voltage**2 - cap.v_min**2) / 2

    # ------------------------------------------------------------
    # Assignment energy:
    #
    #     E_task_i + leakage_j
    # ------------------------------------------------------------

    assignment_energy = energy_costs[:, np.newaxis] + leakage[np.newaxis, :]

    feasibility = assignment_energy <= E_allowed[np.newaxis, :]

    if feasibility.shape != (N, K):
        raise ValueError("feasibility must have shape (N, K)")

    if rewards is None:
        rewards = np.ones(N)

    rewards = np.asarray(
        rewards,
        dtype=float,
    )

    if rewards.shape != (N,):
        raise ValueError("rewards must have shape (N,)")

    M1, M2, M3, M4, P = build_operators(N, K)

    # ------------------------------------------------------------
    # z = [x; y]
    #
    # x_j = whether capacitor j is selected
    #
    # y_ij = whether task i is assigned to capacitor j
    # ------------------------------------------------------------

    c = np.concatenate(
        [
            np.zeros(K),
            np.repeat(energy_costs, K),
        ]
    )

    # ------------------------------------------------------------
    # f = [0_K; feasibility]
    # ------------------------------------------------------------

    f = np.concatenate(
        [
            np.zeros(K),
            feasibility.reshape(-1),
        ]
    )

    # ------------------------------------------------------------
    # Primary objective:
    #
    # maximize number of task assignments.
    #
    # objective^T z
    # ------------------------------------------------------------

    objective = np.asarray(np.ones(K * (N + 1)) @ M1).ravel()

    # ------------------------------------------------------------
    # Constraint 1:
    #
    #     M4 M1 z <= 1
    #
    # Each task is assigned to at most one capacitor.
    # ------------------------------------------------------------

    A_task = M4 @ M1[K:, :]

    b_task = np.ones(N)

    # ------------------------------------------------------------
    # Constraint 2:
    #
    #     (M1 - M3) z <= 0
    #
    # y_ij <= x_j
    # ------------------------------------------------------------

    A_supply_assignment = M1 - M3

    b_supply_assignment = np.zeros(K * (N + 1))

    # ------------------------------------------------------------
    # Constraint 3:
    #
    #     M1 z <= f
    #
    # y_ij <= feasibility_ij
    # ------------------------------------------------------------

    A_feasibility = M1

    b_feasibility = f

    # ------------------------------------------------------------
    # Constraint 4:
    #
    #     sum_j x_j <= M
    # ------------------------------------------------------------

    supply_count_row = np.ones(K * (N + 1)) @ M2

    A_supply_count = sp.csr_matrix(supply_count_row)

    b_supply_count = np.array([M])

    # ------------------------------------------------------------
    # Assemble A and b.
    # ------------------------------------------------------------

    A = sp.vstack(
        [
            A_task,
            A_supply_assignment,
            A_feasibility,
            A_supply_count,
        ],
        format="csr",
    )

    b = np.concatenate(
        [
            b_task,
            b_supply_assignment,
            b_feasibility,
            b_supply_count,
        ]
    )

    return AssignmentModel(
        A=A,
        b=b,
        objective=objective,
        load_operator=P,
        f=f,
        c=c,
        leakage=leakage,
        N=N,
        K=K,
        M=M,
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

    # print(
    #     f"Stage 1 optimal assignments: "
    #     f"{optimal_tasks}"
    # )

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
    ):
        self.caps = caps
        self.tasks = tasks
        self.N = len(tasks)
        self.K = len(caps)
        self.M = cap_limit

    def assign(
        self,
        time,
        floor=3e-6,
    ):
        leakage = [c.leakage for c in self.caps]

        energy_costs = [t.cost * t.duration for t in self.tasks]

        model = build_model(
            N=self.N,
            K=self.K,
            M=self.M,
            caps=self.caps,
            energy_costs=energy_costs,
            leakage=leakage,
        )

        return solve_assignment(model)

    def should_schedule(self, config=None):
        # Return True if any capacitor has enough energy for
        # at least one task.

        if config is None:
            return False

        min_task_energy = min(t.cost * t.duration for t in self.tasks)

        for cap in config.caps:
            if cap.voltage >= cap.v_min:
                available_energy = cap.farads * (cap.voltage**2 - cap.v_min**2) / 2

                if available_energy >= min_task_energy:
                    return True

        return False


class LeacSimConfig(CapacitorStorageSimConfig):
    def __init__(
        self,
        src,
        caps,
        sink,
        plines,
        tasks,
        cap_limit,
    ):
        super().__init__(
            src,
            caps,
            sink,
            plines,
        )

        self.cap_limit = cap_limit

        self.sm = self.sink.sm

        self.assignment = None

        self.assigner = TaskAssigner(
            caps,
            tasks,
            cap_limit,
        )

    def callback(self, time: float):

        if self.sm.time == 0.0:
            for cap in self.caps:
                cap.connect(0)

            self.src.connect(0)
            self.sink.connect(1)

        self.sm.time = time

        # Apply previous assignment to the state machine.
        if self.assignment is not None:
            pairs = self._apply_assignment_to_SM(self.assignment)
            assigned_cap_idx = None

            active_substate = self.sm._get_task_state_id()

            if active_substate is not None:
                for cap_idx, task_idx in pairs:
                    if task_idx == _TASK_IDX[active_substate]:
                        assigned_cap_idx = cap_idx
                        break

            # Assigned capacitor -> sink.
            # All others -> source.
            for i, cap in enumerate(self.caps):
                if i == assigned_cap_idx:
                    cap.disconnect(0)
                    cap.connect(1)

                else:
                    cap.disconnect(1)
                    cap.connect(0)

        # Solve a new assignment when scheduling
        # becomes possible.
        if self.assigner.should_schedule(self):
            result = self.assigner.assign(time)

            self.assignment = result["solution"]

        self._update_sink()

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

        active_substate = self.sm._get_task_state_id()

        if active_substate is not None:
            for cap_idx, task_idx in pairs:
                if task_idx == _TASK_IDX[active_substate]:
                    if self.caps[cap_idx].voltage > self.caps[cap_idx].v_min:
                        self.sm.cap = self.caps[cap_idx]

                        print(
                            f"{self.sm.time:.6f}: "
                            f"{active_substate}, "
                            f"Cap{cap_idx} "
                            f"Volts:"
                            f"{self.sm.cap.voltage:.6f}"
                        )

                    break

        elif self.sm.sleep.is_active:
            pass

        else:
            pass

        return pairs

    def _update_sink(self):
        self.sink.run_sm()


if __name__ == "__main__":
    M = 3

    CONST_VOLTAGE = 3.3

    cap_values = [
        4e-3,
        4e-3,
        4e-3,
    ]

    src = ConstantSource(
        3.3,
        0.05,
        duration=3,
        dt=1,
    )

    caps = [
        Capacitor(
            c,
            v_min=0.5,
            v_max=3.2,
        )
        for c in cap_values
    ]

    tasks = [
        Task(
            cost=-11.68e-3 * CONST_VOLTAGE,
            duration=0.511,
        ),
        Task(
            cost=-86.52e-3 * CONST_VOLTAGE,
            duration=0.285,
        ),
        Task(
            cost=-20.03e-3 * CONST_VOLTAGE,
            duration=0.937,
        ),
    ]

    sink = SMSink(init_SinkSM(caps[0]))

    config = LeacSimConfig(
        src,
        caps,
        sink,
        2,
        tasks,
        M,
    )

    sim = CapacitorStorageSim(config)

    start = time.time()

    sim.run()

    print(f"runtime: {time.time() - start}")

    sim.plot()
