"""
Matrix-form assignment model, solved with SCIP.

The constraint set A z <= b (the paper's Eq. (program)): each task on
exactly one capacitor (the equality, as two inequality rows), y_ij <= x_j,
y_ij <= f_ij, and sum_j x_j <= M.

Thimble solves one binary program with a convex objective:

    minimize ||P z||^2  s.t.  A z <= b

Faithful LEACS is not solved here: it is a fixed plan, chosen by simulating
every feasible static plan and keeping the best (see harness/runner.py), and
run through LeacSimConfig with set_plan and no replanning.

qp_weighting (operator P in the QP objective):

    "unweighted"  P = M5               tasks per capacitor
    "weighted"    P = M5 diag(d) / s   energy per capacitor, d_ij of Eq. (costs),
                                       s = max_ij d_ij (scaling leaves the
                                       minimizer unchanged; see build_model)
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
)
from .state_machines import Task, MATRSinkSM, init_MATRSinkSM
from .leaf_distribution import leaf_distribution

REPLAN_TRIGGERS = ("depth", "forecast")
VISIT_RULES = ("one", "ceil", "max")
CERTIFICATES = ("current", "capacity", "mixed")
REPLAN_FALLBACKS = ("presolve", "keep")
WAKE_RULES = ("energy", "vmax")
QP_WEIGHTINGS = ("unweighted", "weighted")
_OK = ("OPTIMAL", "FEASIBLE")


def tasks_from_sm(sm_cls=MATRSinkSM) -> list[Task]:
    """Planner tasks read from the state machine, so the two cannot drift."""
    return [getattr(sm_cls.task, name).value for name in sm_cls.TASK_NAMES]


@dataclass
class AssignmentModel:
    """
    Solver-independent sparse representation of the assignment problem.
    """

    A: sp.csr_matrix
    b: np.ndarray

    # objective @ z = 1^T M1 z is the number of assigned tasks; used to check
    # that a solution assigns every task.
    objective: np.ndarray

    # P maps z -> per-capacitor loads for the QP objective ||P z||^2:
    # M5 (tasks per capacitor) or M5 diag(d) / s (energy per capacitor).
    load_operator: sp.csr_matrix

    # Feasibility mask f = [0_K; f_1; ...; f_N] and assignment costs
    # d = [0_K; d_1; ...; d_N], both laid out like z.
    f: np.ndarray
    d: np.ndarray

    N: int
    K: int
    M: int

    E_allowed: np.ndarray | None = None
    qp_weighting: str = "unweighted"

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


def visit_counts(P, start: int, depth: int, rule: str = "ceil") -> np.ndarray | None:
    """
    n_i for the D + 1 tasks starting at `start` (the root plus `depth` more).

        "one"   None: every task counted once (the costs are not scaled)
        "ceil"  ceil(expected visits), from expected_visits()
        "max"   the most visits any reachable trajectory makes, from
                leaf_distribution with a one-hot energy vector

    Because the most visits is an integer no smaller than the mean,
    ceil(expected) <= max for every task: "max" is the worst-case bound,
    "ceil" sits between the mean and it. A small tolerance keeps float
    error (2.0000000000000004) from rounding up.
    """

    if rule not in VISIT_RULES:
        raise ValueError(f"visit rule must be one of {VISIT_RULES}, got {rule!r}")
    if rule == "one":
        return None
    if rule == "ceil":
        return np.ceil(expected_visits(np.asarray(P, dtype=float), start, depth) - 1e-9)

    n = len(P)
    out = np.zeros(n)
    for i in range(n):
        onehot = [1.0 if k == i else 0.0 for k in range(n)]
        _, marginal = leaf_distribution(P, onehot, start=start, steps=depth)
        out[i] = max(marginal)
    return out


def trajectories(P, start: int, depth: int) -> list[tuple[int, ...]]:
    """Every task sequence of length depth + 1 from `start` with nonzero probability."""
    paths = [(start,)]
    for _ in range(depth):
        paths = [p + (j,) for p in paths for j, pij in enumerate(P[p[-1]]) if pij > 0]
    return paths


def board_rows(K: int, N: int, board, cover: bool):
    """
    Rows in z-space that freeze the hardware to `board` (x = board) and,
    with cover=True, make every board capacitor hold at least one task
    (x_j - sum_i y_ij <= 0).
    """
    board = np.asarray(board, dtype=float)
    X = sp.hstack([sp.eye(K, format="csr"), sp.csr_matrix((K, K * N))], format="csr")
    rows = [(X, board), (-X, -board)]
    if cover:
        _, _, _, _, M5 = build_operators(N, K)
        rows.append((X - M5, np.zeros(K)))
    return rows


def near_term_rows(K: int, N: int, paths, costs, stored):
    """
    One row per (trajectory, capacitor): the energy every task on the
    trajectory draws from capacitor j must fit what j stores now,

        sum_{i in path} count_path(i) * costs[i][j] * y_ij - stored_j * x_j <= 0.

    Exact per trajectory, so taking one branch never pays for the other.
    """
    rows, cols, vals = [], [], []
    r = 0
    for path in paths:
        counts = np.bincount(path, minlength=N)
        for j in range(K):
            for i in np.flatnonzero(counts):
                rows.append(r); cols.append(K + i * K + j); vals.append(counts[i] * costs[i][j])
            rows.append(r); cols.append(j); vals.append(-stored[j])
            r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, K * (N + 1)))
    return [(A, np.zeros(r))]


def forecast_brownout(P, plan, costs, available, start: int, horizon: int, risk: float = 0.0):
    """
    Forecast capacitor overdraw along the task tree, assuming no harvest.

    P[i][j]       transition probabilities, task order
    plan[i]       capacitor of task i
    costs[i][j]   energy one execution of task i draws from capacitor j
    available[j]  usable energy stored in capacitor j now

    Energy drawn from capacitor j along a trajectory is the sum of costs of
    its tasks that run on j, so leaf_distribution tracks it exactly with
    e_i = costs[i][j] for tasks on j and 0 otherwise. Depth 0 is the root
    task itself.

    For each capacitor, the earliest depth in 0..horizon at which the
    probability of overdraw exceeds `risk` (0 means "some reachable
    trajectory overdraws it"). With risk > 0 each capacitor is tested
    separately, a lower bound on the probability that any is overdrawn.

    Returns (depth, risky): the earliest such depth over all capacitors, or
    None; and the set of tasks that tip a capacitor over at its earliest
    overdraw depth (the last task of each overdrawing trajectory there).
    """

    first, risky = None, set()
    for j in sorted(set(plan)):
        e = [costs[i][j] if plan[i] == j else 0.0 for i in range(len(plan))]
        dist, marginal = leaf_distribution(P, e, start=start, steps=0)
        for D in range(horizon + 1):
            if D > 0:
                dist, marginal = leaf_distribution(P, e, steps=1, dist=dist)
            p_over = sum(p for E, p in marginal.items() if E > available[j])
            if p_over > risk:
                first = D if first is None else min(first, D)
                risky |= {s for (s, E), p in dist.items() if E > available[j] and p > 0}
                break
    return first, risky


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
    feasibility_costs: np.ndarray | None = None,
    qp_weighting: str = "unweighted",
    feasibility_mask: np.ndarray | None = None,
    extra_constraints: list | None = None,
) -> AssignmentModel:
    """
    Build the sparse matrix representation of the constraints, Eq. (program).

    energy_costs:
        d_ij of Eq. (costs), shape (N, K). Build it with
        assignment_costs(...). With visits = 1 and no switching cost this
        is E_i + l_ij, i.e. what you passed before.

    feasibility_costs:
        Optional (N, K) costs for Eq. (f) only. None (default) uses d_ij,
        as the paper is written. With fractional visit counts (n_i < 1),
        pass the one-execution costs, assignment_costs(..., visits=None),
        so that f_ij = 1 still means one run of u_i fits in c_j.

    qp_weighting:
        "unweighted"  load operator M5 (tasks per capacitor).
        "weighted"    load operator M5 diag(d) / max(d) (energy per capacitor).

    feasibility_mask:
        Optional (N, K) boolean f_ij that replaces the mask computed from
        energies, e.g. a mixed certificate (see TaskAssigner.assign).

    extra_constraints:
        Optional list of (A_rows, b_rows) pairs in z-space, appended to
        A z <= b (frozen hardware, board coverage, near-term trajectories).
    """

    if qp_weighting not in QP_WEIGHTINGS:
        raise ValueError(f"qp_weighting must be one of {QP_WEIGHTINGS}, got {qp_weighting!r}")

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

    if feasibility_mask is None:
        feasibility = f_costs <= E_allowed[np.newaxis, :]
    else:
        feasibility = np.asarray(feasibility_mask, dtype=bool)
        if feasibility.shape != (N, K):
            raise ValueError(f"feasibility_mask must have shape ({N},{K})... has {feasibility.shape}")

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
    # 1^T M1 z counts assigned tasks (checked after every solve).
    # ------------------------------------------------------------

    ones = np.ones(K * (N + 1))
    objective = np.asarray(ones @ M1).ravel()

    # ------------------------------------------------------------
    # Eq. (program), row by row.
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

    for A_extra, b_extra in extra_constraints or []:
        blocks_A.append(sp.csr_matrix(A_extra))
        blocks_b.append(np.asarray(b_extra, dtype=float))

    # 1^T M2 z - M <= 0: at most M capacitors.
    A_supply_count = sp.csr_matrix(np.asarray(ones @ M2).ravel())
    b_supply_count = np.array([M])

    blocks_A.append(A_supply_count)
    blocks_b.append(b_supply_count)

    A = sp.vstack(blocks_A, format="csr")
    b = np.concatenate(blocks_b)

    # ------------------------------------------------------------
    # QP load operator P, objective ||P z||^2.
    #
    # The weighted operator is divided by the largest cost. That leaves the
    # minimizer unchanged but keeps the objective O(1): in joules it is
    # ~1e-4 J^2 with gaps of ~1e-6 between plans, close enough to SCIP's
    # tolerances that it occasionally returned a suboptimal plan in testing.
    # ------------------------------------------------------------

    if qp_weighting == "weighted":
        load_operator = (M5 @ sp.diags(d / d_mat.max())).tocsr()
    else:
        load_operator = M5

    return AssignmentModel(
        A=A,
        b=b,
        objective=objective,
        load_operator=load_operator,
        f=f,
        d=d,
        N=N,
        K=K,
        M=M,
        E_allowed=E_allowed,
        qp_weighting=qp_weighting,
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
# The QP
# ----------------------------------------------------------------------


def solve_qp(
    model: AssignmentModel,
) -> dict:
    """
    QP:

        minimize sum_i load_i^2

    subject to:

        A z <= b
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

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------

    scip.optimize()

    status = _scip_status(scip)

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
        "n_assigned": primary_objective,
        "load_objective": load_objective,
        "solution": solution,
        "capacitor_loads": loads,
        "max_capacitor_load": (float(np.max(loads)) if loads.size else 0.0),
        "wall_time_ms": wall_time_ms,
        "iterations": None,
        "nodes": nodes,
    }


def solve_assignment(
    model: AssignmentModel,
) -> dict:
    """
    Solve the QP. Returns the solver's result dict (status, solution,
    wall_time_ms, ...). A successful solve that does not assign every task
    raises, because the equality rows should make that impossible.
    """

    result = solve_qp(model)

    if result["status"] in _OK:
        n_assigned = int(round(float(model.objective @ result["solution"])))
        result["n_assigned"] = n_assigned
        if n_assigned != model.N:
            raise RuntimeError(
                f"QP assigned {n_assigned} of {model.N} tasks; "
                "the assignment equality is not holding"
            )

    return result


class TaskAssigner:
    def __init__(
        self,
        caps,
        tasks,
        cap_limit,
        replan_depth,
        qp_weighting="unweighted",
        certificate="mixed",
        freeze_board=True,
        cover_board=True,
    ):
        if certificate not in CERTIFICATES:
            raise ValueError(f"certificate must be one of {CERTIFICATES}, got {certificate!r}")
        self.certificate = certificate
        self.freeze_board = freeze_board
        self.cover_board = cover_board
        self.board = None   # 0/1 per capacitor, set by presolve (or set_board)
        if qp_weighting not in QP_WEIGHTINGS:
            raise ValueError(f"qp_weighting must be one of {QP_WEIGHTINGS}, got {qp_weighting!r}")
        self.qp_weighting = qp_weighting
        self.caps = caps
        self.tasks = tasks
        self.N = len(tasks)
        self.K = len(caps)
        self.M = cap_limit
        self.replan_depth = replan_depth
        self._planned_at = 0
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
            qp_weighting=self.qp_weighting,
        )
        result = solve_assignment(model)
        if result["status"] not in _OK:
            raise RuntimeError(
                f"presolve is {result['status']}: no plan satisfies the constraints "
                f"(M={self.M}, "
                f"caps={[c.farads for c in self.caps]})"
            )
        self.set_board(result["solution"])
        return result

    def set_board(self, z):
        """The capacitors the plan z uses become the board: the hardware that
        exists at run time. Taken from y, not x, so a capacitor the solver
        selected but gave no task is not counted."""
        y = np.asarray(z, dtype=float)[self.K:].reshape(self.N, self.K)
        self.board = (y.max(axis=0) >= 0.5).astype(float)

    def assign(
        self,
        time,
        visits=None,
        root=None,
        P=None,
        horizon=1,
    ):
        """
        Replan the task-to-capacitor map from the current charge state.

        certificate:
            "current"   every task must fit what its capacitor stores now;
                        replans then favor charged capacitors and may put
                        every task on one of them
            "capacity"  every task must fit a full charge (diagnosis F6)
            "mixed"     tasks reachable within `horizon` transitions of
                        `root` must fit stored energy along every trajectory
                        (no harvest assumed); later tasks must fit a full
                        charge, since they will be recharged first. Needs
                        root and P.
        freeze_board: x is fixed to the presolve's capacitors, so replans
            change only which task uses which capacitor.
        cover_board: every board capacitor keeps at least one task. False lets
            replans concentrate tasks on fewer board capacitors, down to one.
        """
        # c.leakage is current -> p = IV = 0.01*C*V^2
        # c.energy = 0.5 * v**2 * c
        #   ==> p_leak = 0.01 C * V**2 = 0.02 * c.energy / 1 s
        one_run = np.array([
            [(c.leakage * c.voltage + t.cost) * t.duration for c in self.caps] for t in self.tasks
        ])

        # d_ij = n_i (E_i + l_ij), Eq. (costs). The per-pair mask keeps
        # one-execution costs, so f_ij = 1 still means one run fits.
        if visits is None:
            energy_costs = one_run
        else:
            energy_costs = np.asarray(visits, dtype=float)[:, np.newaxis] * one_run

        stored = np.array([max(c.energy - c.min_energy, 0.0) for c in self.caps])
        full = np.array([c.farads * (c.v_max**2 - c.v_min**2) / 2 for c in self.caps])
        extra = []

        if self.certificate == "current":
            mask = one_run <= stored
        elif self.certificate == "capacity":
            mask = one_run <= full
        else:  # "mixed"
            if root is None or P is None:
                raise ValueError('certificate="mixed" needs root and P')
            paths = trajectories(P, root, horizon)
            near = np.zeros(self.N, dtype=bool)
            near[[i for p in paths for i in p]] = True
            mask = np.where(near[:, np.newaxis], one_run <= stored, one_run <= full)
            extra += near_term_rows(self.K, self.N, paths, one_run, stored)

        if self.freeze_board and self.board is not None:
            mask = mask & (self.board[np.newaxis, :] > 0)
            extra += board_rows(self.K, self.N, self.board, self.cover_board)

        model = build_model(
            N=self.N,
            K=self.K,
            M=self.M,
            caps=self.caps,
            energy_costs=energy_costs,
            qp_weighting=self.qp_weighting,
            feasibility_mask=mask,
            extra_constraints=extra,
        )

        return solve_assignment(model)

    def should_schedule(self, config=None):
        if config is None or self.replan_depth <= 0:
            return False
        if config.depth % self.replan_depth or config.depth == getattr(self, "_planned_at", 0):
            return False
        self._planned_at = config.depth
        return True



class LeacSimConfig(CapacitorStorageSimConfig):
    def __init__(
        self,
        # tasks,
        cap_limit=2,
        replan_depth=3,
        harvest_routing="fallback",
        charge_band=0.05,
        qp_weighting="unweighted",
        replan_trigger="depth",
        forecast_horizon=1,
        forecast_risk=0.0,
        forecast_gate=True,
        visit_rule="one",
        visit_depth=3,
        certificate="mixed",
        freeze_board=True,
        cover_board=True,
        replan_fallback="presolve",
        wake_rule="energy",
        wake_margin=1.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.cap_limit = cap_limit

        self.replan_depth = replan_depth

        # "children": charge only the caps of the active task's possible next
        #     tasks (harvest idles when none needs charge).
        # "fallback": if no child cap needs charge, feed the active cap; if it
        #     is full too, charge one other board cap.
        # Two rules keep the source from flipping between lines every step,
        # which stalls the SPICE solver:
        # charge_band (V): while awake, a cap that reaches v_max counts as full
        #     until it falls charge_band below v_max.
        # The running cap is fed at most until it first fills during a task:
        #     the load would otherwise pull it back through the band in a few
        #     steps and the source would keep moving.
        if harvest_routing not in ("children", "fallback"):
            raise ValueError(f"unknown harvest_routing {harvest_routing!r}")
        self.harvest_routing = harvest_routing
        self.charge_band = charge_band
        self._full = set()          # caps inside the band after reaching v_max
        self._fed_task = None       # task during which the running cap filled

        # qp_weighting: "unweighted" (M5) or "weighted" (M5 diag(d)), for the
        # presolve and for replans.
        self.qp_weighting = qp_weighting

        # replan_trigger:
        #   "depth"     every replan_depth task transitions (previous behavior)
        #   "forecast"  at each task start, enumerate trajectories from the
        #               active task with leaf_distribution and replan if, with
        #               no harvest, some capacitor can be overdrawn within
        #               forecast_horizon transitions (1 = by the next task)
        #               with probability above forecast_risk (0 = "can")
        # forecast_gate: when the forecast fires, replan only if some task that
        #   can tip a capacitor over has another capacitor that holds enough
        #   stored energy for one run of it now. Otherwise no new plan can
        #   avoid the wait for charge, and the replan is skipped.
        # visit_rule: n_i for replans over visit_depth transitions from the
        #   active task: "one" (previous behavior), "ceil" (ceil of expected
        #   visits) or "max" (worst case over reachable trajectories). The
        #   counts scale d_ij, so they act through the weighted objective.
        if replan_trigger not in REPLAN_TRIGGERS:
            raise ValueError(f"replan_trigger must be one of {REPLAN_TRIGGERS}, got {replan_trigger!r}")
        if visit_rule not in VISIT_RULES:
            raise ValueError(f"visit_rule must be one of {VISIT_RULES}, got {visit_rule!r}")
        self.replan_trigger = replan_trigger
        self.forecast_horizon = forecast_horizon
        self.forecast_risk = forecast_risk
        self.forecast_gate = bool(forecast_gate)
        self.visit_rule = visit_rule
        self.visit_depth = visit_depth
        # Replans (see TaskAssigner.assign): certificate "current", "capacity"
        #   or "mixed" (tasks within forecast_horizon against stored energy,
        #   later ones against a full charge); freeze_board keeps the presolve's
        #   capacitors as the hardware; cover_board keeps a task on each of
        #   them. replan_fallback: on a stranded or infeasible replan, go back
        #   to the presolve plan ("presolve") or keep the current one ("keep").
        # wake_rule: "energy" wakes once the charging capacitor holds
        #   wake_margin x the resumed task's energy; "vmax" waits for v_max,
        #   as LEACS assumes.
        if replan_fallback not in REPLAN_FALLBACKS:
            raise ValueError(f"replan_fallback must be one of {REPLAN_FALLBACKS}, got {replan_fallback!r}")
        if wake_rule not in WAKE_RULES:
            raise ValueError(f"wake_rule must be one of {WAKE_RULES}, got {wake_rule!r}")
        self.replan_fallback = replan_fallback
        self.wake_rule = wake_rule
        self.wake_margin = wake_margin
        self.replan_fallbacks = 0
        self._forecast_key = None
        self.forecast_checks = 0
        self.forecast_triggers = 0
        self.forecast_gated = 0
        self.last_forecast = None

        self.assignment = None


        # save the startup hash
        self.save_startup_hash()


        tasks = tasks_from_sm(self.sink.sm)

        self.assigner = TaskAssigner(
            self.caps,
            tasks,
            cap_limit,
            replan_depth,
            qp_weighting=qp_weighting,
            certificate=certificate,
            freeze_board=freeze_board,
            cover_board=cover_board,
        )

        result = self.assigner.presolve()
        self.assignment = result["solution"]
        self.presolve_assignment = np.array(result["solution"], dtype=float)

        # set initial sm params
        pairs = self._decode_assignment_result(self.assignment)
        self.sm = self.sink.sm
        self.sm.cap = self.caps[pairs[0][0]]
        self.sm.task_start = 0.0
        self.sm.time = 0.0

        self.last_substate = None
        self.depth = 0
        self.full_lap = 0

    def callback(self, time):
        active, children = self.sm._get_task_state_id()

        if self.sm.time == 0.0:
            self.sink.connect(1)
            self.sink.disconnect(0)
            self.src.disconnect(1)
        self.sm.time = time

        pairs = self._decode_assignment_result(self.assignment)
        cap_of = {t: c for c, t in pairs}

        # never run a task on a cap that cannot power the sink: pause and charge it
        if active not in (None, "sleep"):
            j = cap_of[self.assigner.task_idx[active]]
            if self.caps[j].voltage <= self.caps[j].v_min:
                self.sm.send("pause")
                self.last_substate, active = active, "sleep"
            else:
                self.sm.cap = self.caps[j]

        assigned, charging = None, []
        if active == "sleep":
            resume = self.assigner.task_idx[self.last_substate or "measure"]
            j = cap_of[resume]
            charging = [j]
            self.sm.cap = self.caps[j]                       # wake test == charging cap
            task = self.assigner.tasks[resume]
            if self.wake_rule == "energy":
                # Capped at a full charge: with the margin, a tightly sized
                # capacitor could otherwise never wake (2.2 mF holds 8.27 mJ;
                # 1.1 x Transmit needs 8.69 mJ).
                cap = self.caps[j]
                full = 0.5 * cap.farads * (cap.v_max**2 - cap.v_min**2)
                self.sm.wake_energy = min(
                    self.wake_margin * (task.cost + cap.leakage * cap.voltage) * task.duration,
                    full,
                )
            else:
                self.sm.wake_energy = None   # MATRSinkSM.charged: V >= v_max
        elif active is not None:
            assigned = cap_of[self.assigner.task_idx[active]]
            charging = [cap_of[t] for t in np.ravel(children) if cap_of[t] != assigned]

        # depth counts task-to-task transitions and rx->measure for metrics
        if active not in (None, "sleep") and self.last_substate and active != self.last_substate:
            self.depth += 1
            if self.last_substate == 'rx' and active == 'measure':
                self.full_lap += 1

        charging, feed_active = self._route_harvest(assigned, charging, cap_of, active)

        for i, cap in enumerate(self.caps):
            if i == assigned:
                cap.disconnect(0); cap.connect(1)
            elif i in charging:
                cap.connect(0); cap.disconnect(1)
            else:
                cap.disconnect(0); cap.disconnect(1)

        # The source sits on exactly one line. With both switches closed, the
        # source node would join line 0 and line 1 and put the active cap in
        # parallel with the charging caps.
        if feed_active:
            self.src.disconnect(0); self.src.connect(1)
        else:
            self.src.disconnect(1); self.src.connect(0)

        if self._should_replan(active, cap_of):
            root = active if active not in (None, "sleep") else (self.last_substate or "measure")
            try:
                result = self.assigner.assign(
                    time,
                    visits=self._visits(active),
                    root=self.assigner.task_idx[root],
                    P=self._transition_rows(),
                    horizon=self.forecast_horizon,
                )
            except ValueError:
                result = None
            if result is not None and result["status"] in ("OPTIMAL", "FEASIBLE"):
                self.assignment = result["solution"]
            elif self.replan_fallback == "presolve":
                self.assignment = self.presolve_assignment.copy()
                self.replan_fallbacks += 1

        if active != "sleep":
            self.last_substate = active
        self._update_sink()

    def set_plan(self, z):
        """Replace the starting plan (e.g. an injected static plan). The plan
        also becomes the board and the replan fallback."""
        self.assignment = np.array(z, dtype=float)
        self.presolve_assignment = np.array(z, dtype=float)
        self.assigner.set_board(z)

    def _transition_rows(self):
        """Transition matrix rows in task order, from the state machine."""
        return [np.asarray(self.sm.transitions[t.name], dtype=float) for t in self.assigner.tasks]

    def _should_replan(self, active, cap_of):
        """Depth trigger or forecast trigger (see replan_trigger)."""
        fire = False
        if self.replan_trigger == "depth":
            fire = self.assigner.should_schedule(self)

        if self.replan_trigger == "forecast" and active not in (None, "sleep"):
            # Once per task start (including a restart after a pause), so the
            # check is edge-triggered like should_schedule. Checking at the
            # start rather than the last step gives the same answer under the
            # no-harvest assumption, and gives the new plan the whole task to
            # charge the capacitors it assigns to the next tasks.
            key = (active, self.sm.task_start)
            if key != self._forecast_key:
                self._forecast_key = key
                self.forecast_checks += 1
                tasks = self.assigner.tasks
                plan = [cap_of[i] for i in range(len(tasks))]
                costs = [[(c.leakage * c.voltage + t.cost) * t.duration for c in self.caps] for t in tasks]
                available = [max(c.energy - c.min_energy, 0.0) for c in self.caps]
                depth, risky = forecast_brownout(
                    self._transition_rows(),
                    plan=plan,
                    costs=costs,
                    available=available,
                    start=self.assigner.task_idx[active],
                    horizon=self.forecast_horizon,
                    risk=self.forecast_risk,
                )
                self.last_forecast = depth
                if depth is not None:
                    self.forecast_triggers += 1
                    board = self._board(cap_of)
                    if not self.forecast_gate or self._replan_can_help(risky, plan, costs, available, board):
                        fire = True
                    else:
                        self.forecast_gated += 1
        return fire

    @staticmethod
    def _replan_can_help(risky, plan, costs, available, board):
        """True if some task that can tip a capacitor over has another board
        capacitor holding enough stored energy for one run of it now."""
        return any(
            available[k] >= costs[i][k]
            for i in risky
            for k in board
            if k != plan[i]
        )

    def _board(self, cap_of):
        """Board capacitors: the presolve's (or injected plan's) hardware.
        Falls back to the caps the current plan uses if no board is set."""
        board = getattr(self.assigner, "board", None)
        if board is None:
            return sorted(set(cap_of.values()))
        return [int(j) for j in np.flatnonzero(board)]

    def _visits(self, active):
        """n_i from the active task over visit_depth transitions, or None."""
        if self.visit_rule == "one":
            return None
        root = active if active not in (None, "sleep") else (self.last_substate or "measure")
        return visit_counts(self._transition_rows(), self.assigner.task_idx[root],
                            self.visit_depth, self.visit_rule)

    def _route_harvest(self, assigned, charging, cap_of, active=None):
        """Decide where harvested energy goes this step.

        Returns (charging, feed_active): the caps to put on the charge line
        (line 0), and whether the source should move to the sink line (line 1)
        to feed the active cap instead.

        A cap needs charge while it is below v_max (with the charge_band
        hysteresis while awake). In priority order:
          1. children of the active task (or, asleep, the task to resume);
          2. the active cap, by connecting the source to the sink line, until
             it first fills during the current task;
          3. one other board capacitor (whether or not the current plan
             gives it a task), the one with the lowest voltage relative to v_max.
        With harvest_routing="children" only tier 1 applies.
        """
        band = self.charge_band if assigned is not None else 0.0

        def needs_charge(j):
            v, v_max = self.caps[j].voltage, self.caps[j].v_max
            if v >= v_max:
                self._full.add(j)
            elif v < v_max - band:
                self._full.discard(j)
            return j not in self._full

        # Tier 1. dict.fromkeys drops duplicates when two children share a cap.
        tier1 = [j for j in dict.fromkeys(charging) if needs_charge(j)]
        if tier1 or self.harvest_routing == "children":
            return tier1, False

        # Tier 2: the active cap. Planner certificates assume no harvest, so
        # feeding the running task only adds margin.
        if assigned is not None and self._fed_task != (active, self.sm.task_start):
            if self.caps[assigned].voltage < self.caps[assigned].v_max:
                return [], True
            self._fed_task = (active, self.sm.task_start)    # filled: stop for this task

        # Tier 3: bank energy in another board cap, whether or not the current
        # plan gives it a task, so no board capacitor is left to drain.
        others = [j for j in self._board(cap_of) if j != assigned and needs_charge(j)]
        if others:
            return [min(others, key=lambda k: self.caps[k].voltage / self.caps[k].v_max)], False
        return [], False

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

    CONST_VOLTAGE = 3.2

    cap_values = [
        6e-3,
        6e-4,
        6e-3,
    ]

    src = SineSource(
        1.70e-2,
        1.70e-2,
        0.50,
        0.00,
        duration=100,
        dt=0.01,
    )

    caps = [
        LeacsCapacitor(
            c,
            v_min=1.7,
            v_max=3.2,
        )
        for c in cap_values
    ]

    sink = SMSink(init_MATRSinkSM(caps[0]))

    config = LeacSimConfig(
        cap_limit=M,
        replan_depth=replan_depth,
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
