"""Implements different models that can be used for simulation.

Ideally every model would implement a input and output source such that they
can be controlled from outside the original function.
"""

import math

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


class SineShared(NgSpiceShared):
    def __init__(self, amplitude, frequency, **kwargs):

        super().__init__(**kwargs)

        self._amplitude = amplitude
        self._pulsation = float(frequency.pulsation)

    def get_vsrc_data(self, voltage, time, node, ngspice_id):
        self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))
        voltage[0] = self._amplitude * math.sin(self._pulsation * time)
        return 0

    def get_isrc_data(self, current, time, node, ngspice_id):
        self._logger.debug('ngspice_id-{} get_isrc_data @{} node {}'.format(ngspice_id, time, node))
        current[0] = 1.
        return 0

    def send_data(self, data, count, ngspice_id):
        # called at each simulation step
        print(f"Step {count}: {data.actual_vector_values}")
        return 0


def create_example_shared_model() -> Circuit:
    """Creates a voltage divider circuit as an example of shared model.

    Returns:
        Circuit model.
    """

    circuit = Circuit("Array of capacitor storage")

    circuit.V("input", "input", circuit.gnd, "dc 0 external")
    circuit.R(1, 'input', 'output', 10@u_kOhm)
    circuit.R(2, 'output', circuit.gnd, 1@u_kOhm)

    return circuit


def create_basic_model(model: str = "C_real", **kwargs) -> Circuit:
    """Creates a basic model to charge/dischard capacitor.

    Available fields for model is "C_real" and "C_ideal". At time of writing
    the capacitor model incorperates "Resr", "Rleak", "Cval", "fo".

    Args:
        cap: Model name

    Returns:
        Circuit model
    """

    circuit = Circuit("Leakage current of capacitors")

    # Include capacitor subcircuit library
    # TODO update to a relative path
    circuit.include("/home/jtmadden/repos/jlab/shiny-train/cap.lib")

    # Switch models
    circuit.model(
        "__S1",
        "SW",
        vt=1,
        ron=1,
    )

    circuit.model(
        "__S2",
        "SW",
        vt=1,
        ron=1,
    )

    # Voltage sources (PWL)
    circuit.PieceWiseLinearVoltageSource(
        "V2",
        "Net-_S1-C+_",
        circuit.gnd,
        values=[
            (0, 0),
            (199e-3, 0),
            (200e-3, 1),
        ],
    )

    # Pulse on charging capacitor
    # circuit.PulseVoltageSource(
    #    "V2",
    #    "Net-_S1-C+_",
    #    circuit.gnd,
    #    initial_value=0,
    #    pulsed_value=1,
    #    delay_time=1@u_s,
    #    pulse_width=200@u_ms,
    #    period=400@u_ms,
    # )

    circuit.PieceWiseLinearVoltageSource(
        "V1",
        "Net-_R3-Pad2_",
        circuit.gnd,
        values=[
            (0, 0),
            (99e-3, 0),
            (100e-3, 1),
        ],
    )

    # Resistors
    circuit.R(
        "3",
        "source",
        "Net-_R3-Pad2_",
        100 @ u_kΩ,
    )

    circuit.R(
        "1",
        "load",
        circuit.gnd,
        200 @ u_Ω,
    )

    # Subcircuit capacitor
    circuit.X(
        "C2",
        model,
        "C1",
        circuit.gnd,
        **kwargs,
    )

    # Voltage controlled switches
    circuit.S(
        "1",
        "source",
        "C1",
        "Net-_S1-C+_",
        circuit.gnd,
        model="__S1",
    )

    circuit.S(
        "2",
        "source",
        "load",
        "Net-_S2-C+_",
        circuit.gnd,
        model="__S2",
    )

    # Second switch control source
    # circuit.PieceWiseLinearVoltageSource(
    #    "V3",
    #    "Net-_S2-C+_",
    #    circuit.gnd,
    #    values=[
    #        (0, 0),
    #        (0.99, 0),
    #        (1, 1),
    #    ],
    # )

    circuit.PulseVoltageSource(
        "V3",
        "Net-_S2-C+_",
        circuit.gnd,
        initial_value=0,
        pulsed_value=1,
        delay_time=1 @ u_s,
        pulse_width=200 @ u_ms,
        period=400 @ u_ms,
    )

    return circuit
