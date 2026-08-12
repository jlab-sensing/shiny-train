#!/usr/bin/env python3

from typing import Callable
from ortools.linear_solver import pywraplp

from sim.models import Capacitor


SwitchState = tuple[float, list[float]]


# Call once at start and hold - avoid recomputing
def get_costs(caps: list[Capacitor]):

    # TODO: define task_costs w/ John, Steve

    costs = [[float('inf')]*len(tasks)]*len(caps)
    for j, cap in enumerate(caps):
        for i, cost in enumerate(task_costs):
            costs[i,j] = cost
    return costs


def optimize_assignment(
    time: float,
    caps: list[Capacitor],
) -> SwitchState:
    """
    Determine the connectivity of the capacitor array at a simulation
    timestep.

    Args:
        time: Current simulation time [s].
        caps: Capacitors with their current state/voltage information.

    Returns:
        load_switch: Control voltage for the load switch.
        switch_state: Control voltage for each capacitor switch.
    """

    # Data
    costs = get_costs(caps)
    num_capacitors = len(costs)
    num_tasks = len(costs[0])

    # Preallocate outputs in all-open configuration
    load_switch = 0.0
    switch_state = [0.0 for _ in caps]

    # Solver
    # Create a mip solver with a SCIP backend. Can also use 'CBC' by COIN-OR.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return

    # Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if Capacitor i is assigned to task j.
    x = {}
    for i in range(num_capacitors):
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, "")

    # Constraints
    # # Each Capacitor is assigned to at most 1 task.
    # for i in range(num_capacitors):
    #     solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

    # Each task is assigned to exactly one Capacitor.
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_capacitors)]) == 1)

    # Objective
    objective_terms = []
    for i in range(num_capacitors):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    # Solve
    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    # Print and return solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Total cost = {solver.Objective().Value()}\n")
        load_switch = 1.0
        for i in range(num_capacitors):
            for j in range(num_tasks):
                # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                if x[i, j].solution_value() > 0.5:
                    print(f"Capacitor {i} assigned to task {j}." + f" Cost: {costs[i][j]}")
                    switch_state[i,j] = 1.0
    else:
        print("No solution found.")

    return load_switch, switch_state
