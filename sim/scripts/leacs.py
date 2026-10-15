#!/usr/bin/env python

"""Run LEACS simulation."""

import time

from sim.callback import LeacSimConfig
from sim.models import (
    BonitoSource,
    RWSource,
    CVCapacitor,
    IdealCapacitor,
    SimulationRunner,
    SMSink,
    simulation_config_matrix,
)
from sim.state_machines import MATRSinkSM, Task


def create_tasks() -> list:
    CONST_VOLTAGE = 3.2

    tasks = [  # bookkeeping for LeacSimConfig; SMSink is hardcoded with identical Tasks
        Task(cost=11.68e-4 * CONST_VOLTAGE, duration=0.511, name="measure"),
        Task(cost=2 * 11.68e-4 * CONST_VOLTAGE, duration=0.511, name="analyze"),
        Task(cost=86.52e-4 * CONST_VOLTAGE, duration=0.285, name="tx"),
        Task(cost=20.03e-4 * CONST_VOLTAGE, duration=0.937, name="rx"),
    ]

    return tasks


if __name__ == "__main__":
    caps = [
        1e-3,
        4.7e-3,
        10e-3,
    ]

    configs = simulation_config_matrix(
        [IdealCapacitor, CVCapacitor],
        caps,
        # [BonitoSource, BonitoSource, BonitoSource, BonitoSource],
        [RWSource]*100,
        # [
        #     {"duration": 1200, "downsample": 30000, "name": "node0"},
        #     {"duration": 1200, "downsample": 30000, "name": "node1"},
        #     {"duration": 1200, "downsample": 30000, "name": "node2"},
        #     {"duration": 1200, "downsample": 30000, "name": "node3"},
        # ],
        [
            {"duration": 1000, "dt":1}
        ]*100,
        [SMSink],
        {"sink": MATRSinkSM()},
        [LeacSimConfig, LeacSimConfig],
        [
            {
                "p_lines": 2,
                "tasks": create_tasks(),
                "cap_limit": 2,
                "initial_cap_idx": 0,
                "replan_depth": 3,
            },
            {
                "p_lines": 2,
                "tasks": create_tasks(),
                "cap_limit": 2,
                "initial_cap_idx": 0,
                "replan_depth": -1,
            }
        ],
        # names=[
        #     "Ideal/node0",
        #     "Ideal/node1",
        #     "Ideal/node2",
        #     "Ideal/node3",
        #     "CV/node0",
        #     "CV/node1",
        #     "CV/node2",
        #     "CV/node3",
        # ],
        names=[f"Ideal/node{i}" for i in range(100)] + [f"CV/node{i}" for i in range(100)],
    )

    runner = SimulationRunner(configs)
    start = time.time()
    results = runner.run(progress_bar=True)
    print(f"runtime: {time.time() - start}")
    runner.plot()
    runner.show()
