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
from sim.state_machines import MATRSinkSM


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
        [LeacSimConfig],
        {
            "p_lines": 2,
            "cap_limit": 2,
            "replan_depth": -1,
            "energy_constraint": False,
            # "replan_depth": 2,
            # "energy_constraint": False
            "harvest_routing": 'fallback',
            "solver": 'qp',
            "qp_weighting": 'weighted'
            # "qp_weighting": 'unweighted'
        },
        # names=[f"LEACS/src{i}" for i in range(4)],
        names=[f"Thimble/src{i}" for i in range(4)],
    )

    runner = SimulationRunner(configs)
    start = time.time()
    results = runner.run(progress_bar=True)
    print(f"runtime: {time.time() - start}")
    runner.plot()
    runner.show()
