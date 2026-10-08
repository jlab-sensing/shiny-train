#!/usr/bin/env python

"""Example showing multiple simulation runs in parallel.

Deep copy with the `copy` module did not with the class inheritence
configuration. We recommend creating a function to return a specific
configuration based on parameters as shown here.
"""

import time

from sim.models import (
    Capacitor,
    CapacitorStorageSimConfig,
    ConstantSink,
    ConstantSource,
    CVCapacitor,
    SimulationRunner,
    SMSink,
    BonitoSource,
    IdealCapacitor,
)
from sim.callback import LeacSimConfig
from sim.state_machines import Task, init_SinkSM





def create_config(caps: list[Capacitor]):
    M = 2
    CONST_VOLTAGE = 3.3

    src = BonitoSource(
        '/home/jtmadden/repos/jlab/shiny-train/sim/data/bonito/pwr_cars.h5',
        duration=3600,
        downsample=30000
    )

    # src = ConstantSource(
    #     1,
    #     duration=2,
    #     dt=0.01,
    # )

    # src = SineSource(
    #     1.70e-2,
    #     1.70e-2,
    #     2.00,
    #     0.00,
    #     duration=100,
    #     dt=0.01,
    # )

    tasks = [  # bookkeeping for LeacSimConfig; SMSink is hardcoded with identical Tasks
        Task(
            cost=11.68e-4 * CONST_VOLTAGE,
            duration=0.511,
            name='measure'
        ),
        Task(
            cost=86.52e-4 * CONST_VOLTAGE,
            duration=0.285,
            name='tx'
        ),
        Task(
            cost=20.03e-4 * CONST_VOLTAGE,
            duration=0.937,
            name='rx'
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

    return config



if __name__ == "__main__":
    cap_values = [
        6e-3,
        6e-4,
        6e-3,
    ]

    ideal_caps = [IdealCapacitor(c) for c in cap_values]
    ideal_config = create_config(ideal_caps)

    cv_caps = [CVCapacitor(c) for c in cap_values]
    cv_config = create_config(cv_caps)

    #
    # Runner
    #

    configs = [
        ideal_config,
        cv_config,
    ]

    names = [
        "Ideal",
        "CV Capacitor",
    ]

    runner = SimulationRunner(configs, names)
    start = time.time()
    results = runner.run()
    print(f"runtime: {time.time() - start}")
    runner.plot()
    runner.show()



