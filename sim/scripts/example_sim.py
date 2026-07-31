#!/usr/bin/env python

"""Runs an example simulation"""

from sim.models import CapacitorStorageSim

last_load = 0

def callback(time: float, cap_voltages: list) -> tuple[float, list]:
    global last_load

    print(f"Simulation time: {time}")

    switch_state = [1 for _ in cap_voltages]

    if cap_voltages[0] > 0.5:
        print("load on")
        load = 1
    elif cap_voltages[0] < 0.2:
        print("load off")
        load = 0
    else:
        load = last_load

    last_load = load

    return load, switch_state

sim = CapacitorStorageSim(callback, [10e-6, 100e-6])
sim.run()
sim.plot()
