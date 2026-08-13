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
from typing import Literal, Callable
from ortools.linear_solver import pywraplp

import numpy as np
import scipy.sparse as sp

from sim.models import (
    Capacitor,
    CapacitorStorageSim,
    CapacitorStorageSimConfig,
    Sink,
    Source,
)


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
    E_max: float


def build_operators(N: int, K: int):
    """Construct M1, M2, M3, M4 as sparse matrices."""
    n = K * (N + 1)

    # M1 extracts y from z = [x; y].
    M1 = sp.block_diag(
        (sp.csr_matrix((K, K)), sp.eye(K * N, format="csr")),
        format="csr",
    )

    # M2 extracts x from z.
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
    E_max: float,
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

    E_max:
        Total energy budget.

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

    # TODO: define feasibility
    feasibility = np.asarray(feasibility, dtype=float)

    if energy_costs.shape != (N,):
        raise ValueError("energy_costs must have shape (N,)")
    if leakage.shape != (K,):
        raise ValueError("leakage must have shape (K,)")
    if feasibility.shape != (N, K):
        raise ValueError("feasibility must have shape (N, K)")

    # if rewards is None:
    #     rewards = np.ones(N)
    # rewards = np.asarray(rewards, dtype=float)

    # if rewards.shape != (N,):
    #     raise ValueError("rewards must have shape (N,)")

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
        E_max=E_max,
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

    solution = np.array([variable.solution_value() for variable in z])

    return {
        "status": status,
        "objective": objective.Value(),
        "solution": solution,
        "wall_time_ms": solver.wall_time(),
        "iterations": solver.iterations(),
        "nodes": solver.nodes() if solver_name=="CBC_MIXED_INTEGER_PROGRAMMING" else None,
    }


class LeacSimConfig(CapacitorStorageSimConfig):
    def callback(self, time: float):

        costs = self.get_costs(self.caps)

    # Call once at start and hold - avoid recomputing
    def get_costs(self, caps: list[Capacitor]):

        # TODO: define task_costs w/ John, Steve

        costs = [[float('inf')]*len(tasks)]*len(caps)
        for j, cap in enumerate(caps):
            for i, cost in enumerate(task_costs):
                costs[i,j] = cost
        return costs


if __name__ == "__main__":
    # Small example.
    N = 5
    K = 3
    M = 2
    E_max = 10.0

    energy_costs = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
    leakage = np.array([0.5, 0.2, 0.8])

    model = build_model(
        N=N,
        K=K,
        M=M,
        E_max=E_max,
        energy_costs=energy_costs,
        leakage=leakage,
    )

    print("A shape:", model.A.shape)
    print("A nonzeros:", model.A.nnz)

    for solver_name in [
        "CBC_MIXED_INTEGER_PROGRAMMING",
        "GLOP",
        "PDLP",
    ]:
        result = solve_with_ortools(model, solver_name)
        print(
            solver_name,
            "solution =", result["solution"], ","
            "objective =", result["objective"], ",",
            "time_ms =", result["wall_time_ms"]
        )
