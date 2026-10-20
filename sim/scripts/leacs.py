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
        Task(cost=20.03e-4 * CONST_VOLTAGE, duration=0.927, name="rx"),
    ]

    return tasks


if __name__ == "__main__":
    caps = [
        1e-3,
        4.7e-3,
        10e-3,
    ]

    configs = simulation_config_matrix(
        [CVCapacitor],
        caps,
        [BonitoSource, BonitoSource, BonitoSource, BonitoSource],
        # [RWSource],
        [
            {"duration": 1200, "downsample": 10000, "name": "node0"},
            {"duration": 1200, "downsample": 10000, "name": "node1"},
            {"duration": 1200, "downsample": 10000, "name": "node2"},
            {"duration": 1200, "downsample": 10000, "name": "node3"},
        ],
        # [
        #     {
        #         "duration": 200,
        #         "dt":0.01
        #     }
        # ],
        [SMSink],
        {"sink": MATRSinkSM()},
        # [LeacSimConfig,LeacSimConfig,LeacSimConfig,LeacSimConfig],
        [LeacSimConfig],
        # [
        #     {
        #         "p_lines": 2,
        #         "tasks": create_tasks(),
        #         "cap_limit": 2,
        #         "replan_depth": -1,
        #         "energy_constraint": False
        #     },
        #     {
        #         "p_lines": 2,
        #         "tasks": create_tasks(),
        #         "cap_limit": 2,
        #         "replan_depth": -1,
        #         "energy_constraint": True
        #     },
        #     {
        #         "p_lines": 2,
        #         "tasks": create_tasks(),
        #         "cap_limit": 2,
        #         "replan_depth": 1,
        #         "energy_constraint": True
        #     },
        #     {
        #         "p_lines": 2,
        #         "tasks": create_tasks(),
        #         "cap_limit": 2,
        #         "replan_depth": 2,
        #         "energy_constraint": True
        #     },
        # ],
        {
            "p_lines": 2,
            "tasks": create_tasks(),
            "cap_limit": 2,
            # "replan_depth": 2,
            "replan_depth": -1,
            "energy_constraint": False
        },
        # names=[
        #     "none/node0",
        #     "none/node1",
        #     "none/node2",
        #     "none/node3",
        #     "ec/node0",
        #     "ec/node1",
        #     "ec/node2",
        #     "ec/node3",
        #     "rp/node0",
        #     "rp/node1",
        #     "rp/node2",
        #     "rp/node3",
        #     "ecrp/node0",
        #     "ecrp/node1",
        #     "ecrp/node2",
        #     "ecrp/node3",
        # ],
        names=[f"LEACS/src{i}" for i in range(4)],
        # names=[f"Thimble/src{i}" for i in range(4)],
    )

    runner = SimulationRunner(configs)
    start = time.time()
    results = runner.run(progress_bar=True)
    print(f"runtime: {time.time() - start}")
    runner.plot()
    runner.show()
