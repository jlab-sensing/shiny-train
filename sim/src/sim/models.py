"""Implements different models that can be used for simulation.

Ideally every model would implement a input and output source such that they
can be controlled from outside the original function.
"""

import math

import matplotlib.pyplot as plt

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Spice.NgSpice.Shared import NgSpiceShared



class CapacitorStorageSim:
    class CustomShared(NgSpiceShared):
        """Class that takes in a callback and determines the current state of the
        switches.

        The functions `get_vsrc_data` / `get_isrc_data` are called for every
        source before `send_data` is called.

        The simulation starts at time zero.

        The callback function has parameters time and voltages of each
        capacitor. It returns a tuple for the load switch and switch state list
        that includes all capacitors.
        """

        def __init__(self, cb: Callable[[float, list], tuple[float, list]], caps: list, **kwargs):
            """Sets the callback function.

            Args:
                cb: Callback function
                caps: Capacitor array values
            """

            super().__init__(**kwargs)
            self.cb = cb
            self.caps = caps

            # initialize zero list for switches
            self.load_switch = 0
            self.switch_state = [0 for _ in caps]


        def get_vsrc_data(self, voltage, time, node, ngspice_id):
            self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))

            if node == "input":
                voltage[0] = 1
            elif node == "vload":
                voltage[0] = self.load_switch
            else:
                for idx, _ in enumerate(self.caps):
                    if node == f"v{idx}":
                        voltage[0] = self.switch_state[idx]

            return 0

        def get_isrc_data(self, current, time, node, ngspice_id):
            self._logger.debug('ngspice_id-{} get_isrc_data @{} node {}'.format(ngspice_id, time, node))
            current[0] = 1.
            return 0

        def send_data(self, data, count, ngspice_id):
            """Gets the data at each timestamp.

            Assming that this is sim id?
                ngspice_id = 0

            Number of parameters
                count = 16

            Actual data
                data = 
                {'vinput#branch': 0j, 'v0#branch': 0j, 'v1#branch': 0j,
                'l.x0.l1#branch': 0j, 'l.x1.l1#branch': 0j, 'x1.3': 0j, 'x1.2': 0j,
                'c1+': 0j, 'vsw1+': 0j, 'x0.3': 0j, 'x0.2': 0j, 'c0+': 0j, 'vsw0+':
                0j, 'output': 0j, 'input': 0j, 'time': (2e-05+0j)}
            """

            cap_voltages = [0 for _ in self.caps]
            for idx, _ in enumerate(self.caps):
                cap_voltages[idx] = data[f"c{idx}+"].real

            time = data["time"].real
            self.load_switch, self.switch_state = self.cb(time, cap_voltages)

            return 0


    def __init__(self, cb: Callable[list, list], caps: list):
        """Create a simulation instance with a given configuration.

        Args:
            cb: Callback function
            caps: Capacitor array values
        """

        self.cb = cb
        self.caps = caps
        self.shared = self.CustomShared(cb, caps, send_data=True)

    def _create_circuit(self, caps: list, model: str = "C_real") -> Circuit:
        """Creates the circuit model.

        Available fields for model is "C_real" and "C_ideal". At time of writing
        the capacitor model incorperates "Resr", "Rleak", "Cval", "fo".

        Args:
            caps: Capacitor array values
            model: Capacitor model

        Returns:
            Circuit model object
        """

        circuit = Circuit("Capacitor Array")

        # capacitor models
        circuit.include("/home/jtmadden/repos/jlab/shiny-train/cap.lib")

        # switch model
        circuit.model("S", "SW", vt=1, ron=1)

        # input source
        circuit.V("input", "input", circuit.gnd, "dc 0 external")
        circuit.R(1, "input", "output", 2.2@u_kOhm)

        # capacitor array
        for idx, cap in enumerate(self.caps):
            # switch
            circuit.V(idx, f"VSW{idx}+", circuit.gnd, "dc 0 external")
            circuit.S(idx, "output", f"C{idx}+", f"VSW{idx}+", circuit.gnd,
                      model="S")

            # capacitor
            circuit.X(idx, model, f"C{idx}+", circuit.gnd, Cval=cap)

        # load
        circuit.V("load", "VSWload", circuit.gnd, "dc 0 external")
        circuit.S("Sload", "output", "load", "VSWload", circuit.gnd, model="S")
        circuit.R(2, "load", circuit.gnd, 200@u_Ohm)

        return circuit

    def _simulate(self, circuit: Circuit):


        simulator = circuit.simulator(
            temperature=25,
            nominal_temperature=25,
            simulator="ngspice-shared",
            ngspice_shared=self.shared,
        )

        analysis = simulator.transient(
            step_time=1 @ u_ms,
            end_time=2 @ u_s,
        )

        return analysis

    def run(self):
        """Run the simulation on a set of capacitor values.

        Args:
            caps: Capacitor array values
        """

        circuit = self._create_circuit(self.caps)
        print(circuit)
        self.analysis = self._simulate(circuit)

    def _plot_capacitors(self):
        _, axs = plt.subplots(len(self.caps), 2)

        # Titles
        axs[0][0].set_title("Capacitor Voltage")
        axs[0][1].set_title("Control Voltage")

        for idx, cap in enumerate(self.caps):
            # Voltage plot (column 0)
            axs[idx][0].plot(self.analysis[f"C{idx}+"], label=f"{cap}")
            axs[idx][0].set_ylabel(f"{cap}")

            # Control signal plot (column 1)
            axs[idx][1].plot(self.analysis[f"VSW{idx}+"], label=f"{cap}")

        for ax_row in axs:
            for ax in ax_row:
                ax.grid()
                ax.legend()

    def _plot_input_output(self):
        _, axs = plt.subplots(2, 1)

        axs[0].plot(self.analysis["input"], label="input")
        axs[1].plot(self.analysis["output"], label="output")

        for ax in axs:
            ax.grid()
            ax.legend()

    def plot(self):
        self._plot_capacitors()
        self._plot_input_output()

        plt.show()


    def save(self):
        pass


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
