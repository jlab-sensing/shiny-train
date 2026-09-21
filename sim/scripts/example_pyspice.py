#!/usr/bin/env python


from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from PySpice.Logging import Logging
from PySpice.Unit import *

from sim.models import create_basic_model

logger = Logging.setup_logging()


C_types = ["C_real", "C_ideal"]
# Semi reasonable values take from ChatGPT
C_models = [
    {"Cval": 100e-6, "Resr": 0.2, "Rleak": 1e6, "fo": 150e3},
    {"Cval": 200e-6, "Resr": 0.12, "Rleak": 750e3, "fo": 90e3},
    {"Cval": 1e-3, "Resr": 0.05, "Rleak": 250e3, "fo": 35e3},
]

fig, axs = plt.subplots(len(C_types), len(C_models), figsize=(10, 8), sharex=True)

results = {}

for axs_row, C_type in zip(axs, C_types):
    results[C_type] = {}
    for ax, C_model in zip(axs_row, C_models):
        circuit = create_basic_model(C_type, **C_model)

        # Transient simulation setup
        simulator = circuit.simulator(
            temperature=25,
            nominal_temperature=25,
        )

        analysis = simulator.transient(
            step_time=1 @ u_ms,
            end_time=2 @ u_s,
        )

        ax.plot(analysis.source, label="Input (V)")
        ax.plot(analysis.load, label="Load Voltage (V)")
        ax.plot(analysis.C1, label="Capacitor (V)")

        ax.set_ylabel("Voltage (V)")
        ax.set_xlabel("Time (s)")

        ax.set_title(f"type: {C_type}, value: {C_model['Cval']}")

        ax.grid()
        ax.legend()

        # Total Energy
        power = (analysis.load**2) / (2.2 @ u_Ohm)
        energy = np.trapezoid(power, analysis.time)
        # print(f"type: {C_type}, value: {C_model["Cval"]}: {energy}")

        results[C_type][C_model["Cval"]] = energy


pprint(results)


# Capacitor values
caps = list(results["C_ideal"].keys())

# Energy values
ideal = [float(results["C_ideal"][c]) for c in caps]
real = [float(results["C_real"][c]) for c in caps]

x = np.arange(len(caps))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(x - width / 2, ideal, width, label="Ideal")
ax.bar(x + width / 2, real, width, label="Real")

ax.set_xticks(x)
ax.set_xticklabels([f"{c * 1e6:.0f} µF" for c in caps])
ax.set_ylabel("Energy [J]")
ax.set_xlabel("Capacitance")
ax.set_title("Stored Energy: Ideal vs Real Capacitor")
ax.legend()
ax.grid(axis="y")


plt.tight_layout()
plt.show()

# f# save_figure('figure', 'kicad-example.png')
