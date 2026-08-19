"""Implements different models that can be used for simulation.

Ideally every model would implement a input and output source such that they
can be controlled from outside the original function.
"""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd

from math import pi, sin
from abc import ABC, abstractmethod
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import *


caplib_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cap.lib"
)


class SwitchedComponent:
    """Metaclass for objects that are connected to n number of switches.
    Implements logic to connect, disconnect, and get the current state of each
    switch
    """

    def __init__(self, n: int):
        """Initializes array of switch states.

        Args:
            n: Number of swtiches
        """

        self._sw_arr = [0 for _ in range(n)]

    def connect(self, idx: int):
        """Connects a switch.

        Args:
            idx: Index of the switch
        """

        self._sw_arr[idx] = 1

    def disconnect(self, idx: int):
        """Disconnects a switch.

        Args:
            idx: Index of the switch
        """

        self._sw_arr[idx] = 0

    def state(self, idx: int) -> int:
        """Gets the current state of a switch.

        Args:
            idx: Index of the switch.

        Returns:
            bool, 1 for on, 0 for off
        """

        return self._sw_arr[idx]


class Capacitor(SwitchedComponent):
    def __init__(self, farads: float):
        self.farads = farads
        self.voltage = 0


class Source(SwitchedComponent, ABC):
    def __init__(self, **kwargs):
        self.data = None
        self.load_data()

    @abstractmethod
    def load_data(self):
        """
        Subclasses must implement load_data, subject to the file types
        they read from.

        Output is a pandas dataframe with 2 columns: time and power harvested,
        assigned to self.data
        """

        pass


class ConstantSource(Source):
    def __init__(self, source_am, time, sample_hz, **kwargs):
        self.source_am = source_am      # source amplitude in Volts
        self.time = time                # length of the power trace in seconds
        self.sample_hz = sample_hz      # sampling frequency in Hz

    def load_data(self):
        steps = self.time * self.sample_hz
        ts = np.linspace(0, self.time, steps)
        vs = np.ones_like(ts)
        vs *= self.source_amplitude
        data = np.vstack((ts, vs)).T
        self.data = pd.DataFrame(
            array,
            columns=['Time(s)', 'Potential(V)']
        )


class SineSource(Source):
    def __init__(
        self,
        source_am,
        source_os,
        source_hz,
        source_ph,
        time,
        sample_hz,
        **kwargs
    ):
        self.source_am = source_am      # source amplitude in Volts
        self.source_os = source_os      # source voltage offset in Volts
        self.source_hz = source_hz      # frequency of the source in Hz
        self.source_ph = source_ph      # phase offset of the source in radians
        self.time = time                # length of the power trace in seconds
        self.sample_hz = sample_hz      # sampling frequency in Hz

    def load_data(self):
        steps = int(self.time * self.sample_hz)

        # Elapsed time in seconds, used for generating the waveform
        ts = np.linspace(0, self.time, steps, endpoint=False)

        # Datetime timestamps
        timestamps = pd.date_range(
            start=pd.Timestamp.now(),
            periods=steps,
            freq=pd.Timedelta(seconds=1 / self.sample_hz)
        )

        # Generate voltage
        vs = np.sin(2 * np.pi * self.source_hz * ts + self.source_ph)
        vs *= self.source_am
        vs += self.source_os

        self.data = pd.DataFrame({
            'Timestamp': timestamps,
            'Potential(V)': vs
        })


class Sink(SwitchedComponent):
    def __init__(self):
        pass


class SMSink(Sink):
    def __init__(self, sm_states, **kwargs):
        """
        sm_states is a dict whose key:value pairs are:
        key:
            state
        value:
        [
            cost (float),
            duration (int?)
            [           # list of allowed state transitions
                [       # each item is dst_state key and list of conditions
                    dst_state_1,
                    [
                        condition_1,
                        ...
                    ]
                ]
            ]
        ]
        """
        self.sm_states = sm_states  # dict of state:[cost, (trans1, conds1), ...] pairs
        self.state = None           # name of state as a string
        self.cost = None            # this must be in units of Watts

    def run_sm(self):
        pass


class CapacitorStorageSimConfig:
    def __init__(
        self,
        src: Source,
        caps: list[Capacitor],
        sink: Sink,
        p_lines: int,
    ):
        """
        Power lines connect the power source to the sink. This allows for
        multiple capacitors to charge while one is discharging.

        Args:
            cb: Callback function
            src: Power source
            caps: Capacitor array values
            sink: Power sink
            p_lines: Number of available power lines
        """

        # save parameters
        self.src = src
        self.caps = caps
        self.sink = sink
        self.p_lines = p_lines

        # Initialize number of switches
        SwitchedComponent.__init__(src, p_lines)
        SwitchedComponent.__init__(sink, p_lines)
        for cap in caps:
            SwitchedComponent.__init__(cap, p_lines)

    def callback(self, time: float):
        """Callback function to perform actions during sim runtime.

        This function should be overwritten by the user.

        Args:
            time: Timestep of simulation
        """

        return


class CapacitorStorageSim:
    class CustomShared(NgSpiceShared):
        """
        Class that takes in a callback and updates the current state of the
        switches.

        The functions `get_vsrc_data` / `get_isrc_data` are called for every
        source before `send_data` is called.

        The simulation starts at time zero.

        The callback function has parameters time and voltages of each
        capacitor.
        It returns a tuple for the load switch and switch state list
        that includes all capacitors.
        """

        def __init__(self, config: CapacitorStorageSimConfig, **kwargs):
            """Sets the callback function."""

            super().__init__(**kwargs)
            self.config = config

        def get_vsrc_data(self, voltage, time, node, ngspice_id):
            self._logger.debug(
                f"ngspice_id-{ngspice_id} get_vsrc_data @{time} node {node}"
            )

            # TODO Update to configured power source
            if node == "v_src":
                voltage[0] = 1

            # TODO Update for constant sink power
            if node == "sink":
                pass

            # configure switches
            for n in range(self.config.p_lines):
                # Set capacitor switches
                for idx, cap in enumerate(self.config.caps):
                    if node == f"v_ctrl_pwr{n}_c{idx}":
                        voltage[0] = cap.state(n)

                # Set source switches
                if node == f"v_ctrl_src_pwr{n}":
                    voltage[0] = self.config.src.state(n)

                # Set sink switches
                if node == f"v_ctrl_pwr{n}_sink":
                    voltage[0] = self.config.sink.state(n)

            return 0

        def get_isrc_data(self, current, time, node, ngspice_id):
            self._logger.debug(
                f"ngspice_id-{ngspice_id} get_isrc_data @{time} node {node}"
            )
            current[0] = 1.0
            return 0

        def send_data(self, data, count, ngspice_id):
            """Gets the data at each timestamp.

            Assming that this is sim id?
                ngspice_id = 0

            Number of parameters
                count = 16

            Actual data example
                data =
                {'vinput#branch': 0j, 'v0#branch': 0j, 'v1#branch': 0j,
                'l.x0.l1#branch': 0j, 'l.x1.l1#branch': 0j, 'x1.3': 0j,
                'x1.2': 0j, 'c1+': 0j, 'vsw1+': 0j, 'x0.3': 0j, 'x0.2': 0j,
                'c0+': 0j, 'vsw0+': 0j, 'output': 0j, 'input': 0j,
                'time': (2e-05+0j)}
            """

            for idx, cap in enumerate(self.config.caps):
                cap.voltage = data[f"c{idx}_pos"].real

            time = data["time"].real

            self.config.callback(time)

            return 0

    def __init__(
        self,
        config: CapacitorStorageSimConfig,
    ):
        """Create a simulation instance with a given configuration.


        Args:
            config: Configuration
        """

        self.config = config

        self.shared = self.CustomShared(config, send_data=True)

    def _create_circuit(self, model: str = "C_real") -> Circuit:
        """Creates the circuit model.

        Available fields for model is "C_real" and "C_ideal".
        At time of writing the capacitor model incorperates:
            "Resr", "Rleak", "Cval", "fo".

        Args:
            model: Capacitor model

        Returns:
            Circuit model object
        """

        circuit = Circuit("Capacitor Array")

        # capacitor models
        circuit.include(caplib_path)

        # switch model
        # threshold = 1 V
        # on resistance = 1 Ohm
        circuit.model("S", "SW", vt=1, ron=1)

        # input source
        circuit.V("_src", "v_src_pos", circuit.gnd, "dc 0 external")
        circuit.R(1, "v_src_pos", "src", 2.2 @ u_kOhm)

        # circuit.V("src") creates an element like Vsrc

        # input power switches
        for n in range(self.config.p_lines):
            circuit.V(
                f"_ctrl_src_pwr{n}",
                f"ctrl_src_pwr{n}_pos",
                circuit.gnd,
                "dc 0 external",
            )
            circuit.S(
                f"_src_pwr{n}",
                "src",
                f"pwr{n}",
                f"ctrl_src_pwr{n}_pos",
                circuit.gnd,
                model="S",
            )

        # capacitor array
        for idx, cap in enumerate(self.config.caps):
            # connections to power lines
            for n in range(self.config.p_lines):
                # switch
                circuit.V(
                    f"_ctrl_pwr{n}_c{idx}",
                    f"ctrl_pwr{n}_c{idx}_pos",
                    circuit.gnd,
                    "dc 0 external",
                )
                circuit.S(
                    f"_pwr{n}_c{idx}",
                    f"pwr{n}",
                    f"c{idx}_pos",
                    f"ctrl_pwr{n}_c{idx}_pos",
                    circuit.gnd,
                    model="S",
                )

            # capacitor
            circuit.X(idx, model, f"c{idx}_pos", circuit.gnd, Cval=cap.farads)

        # output power switches
        for n in range(self.config.p_lines):
            # load
            circuit.V(
                f"_ctrl_pwr{n}_sink",
                f"ctrl_pwr{n}_sink_pos",
                circuit.gnd,
                "dc 0 external",
            )
            circuit.S(
                f"_pwr{n}_sink",
                f"pwr{n}",
                "sink",
                f"ctrl_pwr{n}_sink_pos",
                circuit.gnd,
                model="S",
            )

        # resistive load
        circuit.R(2, "sink", circuit.gnd, 200 @ u_Ohm)

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

        circuit = self._create_circuit()
        print(circuit)
        self.analysis = self._simulate(circuit)

    def _plot_capacitors(self):
        _, axs = plt.subplots(len(self.config.caps), 1, sharex=True)

        # Titles
        axs[0].set_title("Capacitor Voltages")

        for idx, cap in enumerate(self.config.caps):
            # Voltage plot (column 0)
            axs[idx].plot(self.analysis[f"c{idx}_pos"], label=f"{cap.farads}")
            axs[idx].set_ylabel("Voltage (V)")

        axs[-1].set_xlabel("Time (ms)")

        for ax in axs:
            ax.grid()
            ax.legend()

    def _plot_cap_switches(self):
        num_switches = self.config.p_lines * len(self.config.caps)
        _, axs = plt.subplots(num_switches, sharex=True)

        axs[0].set_title("Capacitor Switch states")

        row = 0
        for idx, cap in enumerate(self.config.caps):
            for line in range(self.config.p_lines):
                axs[row].plot(
                    self.analysis[f"ctrl_pwr{line}_c{idx}_pos"],
                    label=f"C: {cap.farads} line: {line}",
                )
                axs[row].set_ylabel("Switch State")
                row += 1

        axs[-1].set_xlabel("Time (ms)")

        for ax in axs:
            ax.grid()
            ax.legend()

    def _plot_input_output(self):
        _, axs = plt.subplots(2, 1, sharex=True)

        axs[0].set_title("Source/Sink Voltages")

        axs[0].plot(self.analysis["src"], label="src")
        axs[1].plot(self.analysis["sink"], label="sink")

        for ax in axs:
            ax.set_ylabel("Voltage (V)")

        axs[1].set_xlabel("Time (ms)")

        for ax in axs:
            ax.grid()
            ax.legend()

    def _plot_io_switches(self):
        rows = 2 * self.config.p_lines
        _, axs = plt.subplots(rows, sharex=True)

        axs[0].set_title("Source/Sink Switches")

        axs[-1].set_xlabel("Time (ms)")

        ax_idx = 0

        for n in range(self.config.p_lines):
            axs[ax_idx].plot(
                self.analysis[f"ctrl_src_pwr{n}_pos"],
                label=f"source, line {n}"
            )
            ax_idx += 1

        for n in range(self.config.p_lines):
            axs[ax_idx].plot(
                self.analysis[f"ctrl_pwr{n}_sink_pos"],
                label=f"sink, line {n}"
            )
            ax_idx += 1

        for ax in axs:
            ax.set_ylabel("State")
            ax.grid()
            ax.legend()

    def _plot_power_lines(self):
        _, axs = plt.subplots(self.config.p_lines, sharex=True)

        axs[0].set_title("Power line voltages")

        axs[-1].set_xlabel("Time (ms)")

        for ax, n in zip(axs, range(self.config.p_lines)):
            ax.plot(self.analysis[f"pwr{n}"], label=f"line {n}")
            ax.set_ylabel("Voltage (V)")
            ax.grid()
            ax.legend()

    def plot(self):
        self._plot_capacitors()
        self._plot_cap_switches()
        self._plot_input_output()
        self._plot_io_switches()
        self._plot_power_lines()

        plt.show(block=False)
        input("Press enter to close figures...")

    def save(self):
        pass


class SineShared(NgSpiceShared):
    def __init__(self, amplitude, frequency, **kwargs):

        super().__init__(**kwargs)

        self._amplitude = amplitude
        self._pulsation = float(frequency.pulsation)

    def get_vsrc_data(self, voltage, time, node, ngspice_id):
        self._logger.debug(
            f"ngspice_id-{ngspice_id} get_vsrc_data @{time} node {node}"
        )
        voltage[0] = self._amplitude * math.sin(self._pulsation * time)
        return 0

    def get_isrc_data(self, current, time, node, ngspice_id):
        self._logger.debug(
            f"ngspice_id-{ngspice_id} get_isrc_data @{time} node {node}"
        )
        current[0] = 1.0
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
    circuit.R(1, "input", "output", 10 @ u_kOhm)
    circuit.R(2, "output", circuit.gnd, 1 @ u_kOhm)

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
    circuit.include(caplib_path)

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
