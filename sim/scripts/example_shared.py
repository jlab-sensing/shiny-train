#!/usr/bin/env python

"""Example file for shared ngspice models

Implements a basic voltage divider with a source controlled by python code
"""

from sim.models import create_example_shared_model, SineShared

from PySpice.Logging import Logging
from PySpice.Unit import *

import matplotlib.pyplot as plt

logger = Logging.setup_logging()

circuit = create_example_shared_model()

sine_shared = SineShared(10@u_V, 5@u_Hz)

simulator = circuit.simulator(
    temperature=25,
    nominal_temperature=25,
    simulator="ngspice-shared",
    ngspice_shared=sine_shared,
)

analysis = simulator.transient(
    step_time=1 @ u_ms,
    end_time=2 @ u_s,
)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(analysis["input"], label="Input (V)")
axs[1].plot(analysis["output"], label="Output (V)")
axs[2].plot(analysis["output"] / analysis["input"], label="Gain")

axs[-1].set_xlabel("Time")

for ax in axs:
    ax.set_ylabel("Voltage (V)")
    ax.grid()
    ax.legend()

plt.show()
