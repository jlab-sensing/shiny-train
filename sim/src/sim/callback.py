#!/usr/bin/env python3

"""
Matrix-form assignment model for OR-Tools.

The mathematical model is

    maximize    c_obj^T z
    subject to  A z <= b
                z in {0,1}^n       # MILP / CP-SAT
                or
                0 <= z <= 1        # LP / GLOP / PDLP

The matrix A is assembled from the operators M1--M4.

This file keeps the mathematical formulation in sparse-matrix form and
provides adapters for OR-Tools solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union
from ortools.linear_solver import pywraplp

import numpy as np
import scipy.sparse as sp

from sim.models import (
    Capacitor,
    CapacitorStorageSim,
    CapacitorStorageSimConfig,
    SMSink,
    ConstantSource,
)
from .state_machines import Task


START = 0


@dataclass
class AssignmentModel:
    A: sp.csr_matrix
    b: np.ndarray
    objective: np.ndarray
    f: np.ndarray
    c: np.ndarray
    leakage: np.ndarray
    N: int
    K: int
    M: int


def build_operators(N: int, K: int):
    """Construct M1, M2, M3, M4 as sparse matrices."""
    n = K * (N + 1)

    # M1 projects y from z = [x; y].
    M1 = sp.block_diag(
        (sp.csr_matrix((K, K)), sp.eye(K * N, format="csr")),
        format="csr",
    )

    # M2 projects x from z.
    M2 = sp.block_diag(
        (sp.eye(K, format="csr"), sp.csr_matrix((K * N, K * N))),
        format="csr",
    )

    # M3 replicates x into each of the N assignment blocks.
    # Shape: K(N+1) x K(N+1)
    rows = []
    cols = []
    data = []

    # Row block 0 is zero.
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

    # M4 = I_N kron 1_K^T
    # Shape: N x KN
    M4 = sp.kron(
        sp.eye(N, format="csr"),
        np.ones((1, K)),
        format="csr",
    )

    return M1, M2, M3, M4


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

    Parameters
    ----------
    N:
        Number of computational units.

    K:
        Number of candidate power supplies.

    M:
        Maximum number of selected power supplies.

    energy_costs:
        Shape (N,). E_i = execution energy for task i.

    leakage:
        Shape (K,). Leakage cost associated with selecting supply j.

    rewards:
        Shape (N,), optional task rewards. Defaults to all ones.

    Returns
    -------
    AssignmentModel
    """
    energy_costs = np.asarray(energy_costs, dtype=float)
    leakage = np.asarray(leakage, dtype=float)

    if energy_costs.shape != (N,):
        raise ValueError("energy_costs must have shape (N,)")
    if leakage.shape != (K,):
        raise ValueError("leakage must have shape (K,)")

    E_allowed = np.zeros(K, dtype=float)
    for i, cap in enumerate(caps):
        E_allowed[i] = cap.farads * (cap.voltage**2 - cap.v_min**2) / 2

    assignment_energy = (
        energy_costs[:, np.newaxis]
        + leakage[np.newaxis, :]
    )

    feasibility = assignment_energy <= E_allowed[np.newaxis, :]
    # print(feasibility)

    if feasibility.shape != (N, K):
        raise ValueError("feasibility must have shape (N, K)")

    if rewards is None:
        rewards = np.ones(N)
    rewards = np.asarray(rewards, dtype=float)

    if rewards.shape != (N,):
        raise ValueError("rewards must have shape (N,)")

    M1, M2, M3, M4 = build_operators(N, K)

    # z = [x; y]
    # c = [0_K; E_1 1_K; ...; E_N 1_K]
    c = np.concatenate(
        [np.zeros(K), np.repeat(energy_costs, K)]
    )

    # f = [0_K; f_1; ...; f_N]
    f = np.concatenate(
        [np.zeros(K), feasibility.reshape(-1)]
    )

    # Objective 1:
    # maximize the total number of task assignments.
    #
    # M1 z = y, so:
    #
    #     1_{KN}^T M1 z
    #
    objective = np.asarray(
        np.ones(K * (N+1)) @ M1
    ).ravel()

    # # Objective 2:
    # #
    # # reward_i is assigned to every y_ij for task i.
    # # Thus objective = rewards^T M4 M1 z.
    # #
    # # For binary y, this counts reward for each executed task.
    # objective = np.asarray(
    #     (M4 @ M1).T @ rewards
    # ).ravel()

    # Constraint 1:
    # M4 M1 z <= 1_N
    #
    # Each task is assigned to at most one supply.
    A_task = M4 @ M1[K:,:]
    b_task = np.ones(N)

    # Constraint 2:
    # (M1 - M3) z <= 0
    #
    # y_ij <= x_j
    A_supply_assignment = M1 - M3
    b_supply_assignment = np.zeros(K * (N + 1))

    # Constraint 3:
    # M1 z <= f
    #
    # y_ij <= f_ij
    A_feasibility = M1
    b_feasibility = f

    # # Constraint 4:
    # # c^T M1 z + leakage^T M2 z <= E_max
    # #
    # # This is one scalar linear constraint.
    # energy_row = c @ M1 + np.pad(
    #     leakage,
    #     (0, K * N),
    # ) @ M2
    # A_energy = sp.csr_matrix(energy_row)
    # b_energy = np.array([E_max])

    # Constraint 5:
    # 1^T M2 z <= M
    #
    # At most M supplies selected.
    supply_count_row = np.ones(K*(N+1)) @ M2
    A_supply_count = sp.csr_matrix(supply_count_row)
    b_supply_count = np.array([M])

    # Assemble A and b.
    A = sp.vstack(
        [
            A_task,
            A_supply_assignment,
            A_feasibility,
            # A_energy,
            A_supply_count,
        ],
        format="csr",
    )

    b = np.concatenate(
        [
            b_task,
            b_supply_assignment,
            b_feasibility,
            # b_energy,
            b_supply_count,
        ]
    )

    return AssignmentModel(
        A=A,
        b=b,
        objective=objective,
        f=f,
        c=c,
        leakage=leakage,
        N=N,
        K=K,
        M=M,
    )


def solve_with_ortools(
    model: AssignmentModel,
    solver_name: Literal[
        "CBC_MIXED_INTEGER_PROGRAMMING",
        "GLOP",
        "PDLP",
    ] = "GLOP",
):
    """
    Solve the matrix model using an OR-Tools linear solver backend.

    CBC -> binary MILP
    GLOP -> continuous LP relaxation
    PDLP -> continuous LP relaxation
    """
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"Could not create solver {solver_name!r}")

    n = model.A.shape[1]

    if solver_name == "CBC_MIXED_INTEGER_PROGRAMMING":
        z = [solver.IntVar(0, 1, f"z_{j}") for j in range(n)]
    else:
        z = [solver.NumVar(0, 1, f"z_{j}") for j in range(n)]

    # Add Az <= b row by row.
    A = model.A.tocsr()

    for i in range(A.shape[0]):
        row = A.getrow(i)
        indices = row.indices
        values = row.data

        constraint = solver.RowConstraint(
            -solver.infinity(),
            float(model.b[i]),
            f"constraint_{i}",
        )

        for j, value in zip(indices, values):
            constraint.SetCoefficient(z[j], float(value))

    objective = solver.Objective()

    for j, coefficient in enumerate(model.objective):
        if coefficient != 0:
            objective.SetCoefficient(z[j], float(coefficient))

    objective.SetMaximization()

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        stat_str = "OPTIMAL"
    elif status == pywraplp.Solver.FEASIBLE:
        stat_str = "FEASIBLE"
    else:
        stat_str = "NO_SOLUTION"

    solution = np.array([variable.solution_value() for variable in z])
    if solver_name=="CBC_MIXED_INTEGER_PROGRAMMING":
        nodes = nodes = solver.nodes()
    else:
        nodes = None

    return {
        "status": stat_str,
        "objective": objective.Value(),
        "solution": solution,
        "wall_time_ms": solver.wall_time(),
        "iterations": solver.iterations(),
        "nodes": nodes
    }


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

    def assign(self, time, solver_name, floor=3e-6):
        leakage = [c.leakage for c in self.caps]
        energy_costs = [t.cost * t.duration for t in self.tasks]
        model = build_model(
            N=len(self.tasks),
            K=len(self.caps),
            M=self.M,
            caps=self.caps,
            energy_costs=energy_costs,
            leakage=leakage,
        )

        if solver_name in [
            "CBC_MIXED_INTEGER_PROGRAMMING",
            "GLOP",
            "PDLP",
        ]:
            result = solve_with_ortools(model, solver_name)
            print(
                f"Wake @ time: {time:.06f}:",
                "\n     status =", result["status"],
                "\n     objective =", result["objective"],
                "\n     solution =", result["solution"],
                "\n     time_ms =", result["wall_time_ms"],
                "\n     iterations =", result["iterations"],
            )
            if solver_name == "CBC_MIXED_INTEGER_PROGRAMMING":
                print(
                    "     nodes =", result["nodes"]
                )
            return result

    def should_schedule(self, config=None):
        # Return True if any capacitor has enough energy for at least one task
        if config is None:
            return False
        # Check if any capacitor has voltage >= v_min and enough energy for smallest task
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
        solver_name
    ):
        super().__init__(src, caps, sink, plines)
        self.cap_limit = cap_limit
        self.solver_name = solver_name
        # Use the sink's SM instance (assumes SMSink)
        self.sm = self.sink.sm
        self.assignment = None
        self.assigner = TaskAssigner(
            caps,
            tasks,
            cap_limit,
        )

    _TASK_IDX = {"measure": 0, "tx": 1, "rx": 2}

    def _active_task_idx(self):
        """Return the index of the SM's active task substate, or None in sleep.

        The substates are the instance-bound states in ``self.sm.configuration``.
        Accessing ``self.sm.task.measure.is_active`` does NOT work: that path
        resolves to the class-template state, whose ``is_active`` is always
        False. ``_get_task_state_id`` already iterates the live configuration.
        """
        state_id = self.sm._get_task_state_id()
        return None if state_id is None else self._TASK_IDX[state_id]

    def callback(self, time: float):

        self.sm.time = time

        if time == START:
            result = self.assigner.assign(time, solver_name=self.solver_name)
            self.assignment = result["solution"]
            # Initial: all caps charging from source on line 0
            for cap in self.caps:
                cap.connect(0)
            self.src.connect(0)
            self.sink.connect(1)


        # Use assignment to switch capacitors between charge (line 0) and discharge (line 1)
        if self.assignment is not None:
            pairs = self._decode_assignment_result(self.assignment)
            # Find which capacitor is assigned to the active task
            assigned_cap_idx = None
            task_substates = [
                self.sm.task.measure,
                self.sm.task.tx,
                self.sm.task.rx,
            ]
            active_substate_idx = None
            for idx, substate in enumerate(task_substates):
                if substate.is_active:
                    active_substate_idx = idx
                    break

            if active_substate_idx is not None:
                for cap_idx, task_idx in pairs:
                    if task_idx == active_substate_idx:
                        assigned_cap_idx = cap_idx
                        break

            # Switch capacitors: assigned cap -> sink (line 1), others -> source (line 0)
            for i, cap in enumerate(self.caps):
                if i == assigned_cap_idx:
                    cap.disconnect(0)
                    cap.connect(1)
                else:
                    cap.disconnect(1)
                    cap.connect(0)

        if self.assigner.should_schedule(self):
            result = self.assigner.assign(time, solver_name=self.solver_name)
            self.assignment = result["solution"]
        self._apply_assignment(self.assignment)
        self._update_sink()


    def _decode_assignment_result(self, assignment):
        """
        Decode the solver decision vector into capacitor/task assignments.

        The decision vector has the form

            z = [x; y]

        where
            x.shape == (K,)
            y.shape == (N, K)

        and y[i, j] represents assignment of task i to capacitor j.

        Returns
        -------
        list[tuple[int, int]]
            List of (capacitor_index, task_index) pairs.
        """
        assignment = np.asarray(assignment, dtype=float)

        expected_size = self.assigner.K * (self.assigner.N + 1)
        if assignment.size != expected_size:
            raise ValueError(
                f"Invalid assignment size: expected {expected_size}, "
                f"got {assignment.size}"
            )

        K = self.assigner.K
        N = self.assigner.N

        # Discard x; the remaining variables are the task/capacitor
        # assignment variables.
        y = assignment[K:].reshape(N, K)

        # CBC produces binary values. GLOP/PDLP may produce fractional
        # values, so use a threshold when decoding.
        threshold = 0.5

        pairs = [
            (cap_idx, task_idx)
            for task_idx in range(N)
            for cap_idx in range(K)
            if y[task_idx, cap_idx] >= threshold
        ]

        return pairs

    def _apply_assignment(self, assignment):
        pairs = self._decode_assignment_result(assignment)
        # Map task_idx to SM substate: 0=measure, 1=tx, 2=rx
        task_substates = [
            self.sm.task.measure,
            self.sm.task.tx,
            self.sm.task.rx,
        ]
        # Find which substate is currently active
        active_substate_idx = None
        for idx, substate in enumerate(task_substates):
            if substate.is_active:
                active_substate_idx = idx
                break

        if active_substate_idx is not None:
            # Find capacitor assigned to this task
            for cap_idx, task_idx in pairs:
                if task_idx == active_substate_idx:
                    self.sm.cap = self.caps[cap_idx]
                    break
        elif self.sm.sleep.is_active:
            # In sleep, no capacitor assigned to task; cap recharges via source
            pass

    def _update_sink(self):
        self.sink.run_sm()

if __name__ == "__main__":
    # Small example.
    M = 2
    CONST_VOLTAGE = 3.7

    cap_values = [100e-6, 470e-6, 1e-3]  # Larger capacitors

    # 100 mW @ 3.3 V
    src = ConstantSource(3.3, 0.1, duration=2, dt=1)
    caps = [Capacitor(c, v_min=0.5, v_max=4.5) for c in cap_values]
    tasks = [
        Task(  # measure
            cost=-11.68e-3 * CONST_VOLTAGE,
            duration=0.511
        ),
        Task(  # TX
            cost=-86.52e-3 * CONST_VOLTAGE,
            duration=0.285
        ),
        Task(  # RX
            cost=-20.03e-3 * CONST_VOLTAGE,
            duration=0.937
        )
    ]
    sink = SMSink()
    solver_name = "CBC_MIXED_INTEGER_PROGRAMMING"

    config = LeacSimConfig(
        src,
        caps,
        sink,
        2,
        tasks,
        M,
        solver_name
    )

    sim = CapacitorStorageSim(config)
    sim.run()
    sim.plot()
