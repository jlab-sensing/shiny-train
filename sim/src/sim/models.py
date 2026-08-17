"""Implements different models that can be used for simulation.

Ideally every model would implement a input and output source such that they
can be controlled from outside the original function.
"""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from math import pi, sin
from abc import ABC, abstractmethod
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import *

import PySpice
from cffi import FFI


caplib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cap.lib")


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

        self.n = n
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

    def reset(self):
        """Disconnects all switches."""

        self._sw_arr = [0 for _ in range(self.n)]

    def state(self, idx: int) -> int:
        """Gets the current state of a switch.

        Args:
            idx: Index of the switch.

        Returns:
            bool, 1 for on, 0 for off
        """

        return self._sw_arr[idx]


class Capacitor(SwitchedComponent):
    def __init__(self, farads: float, initial_voltage: float = 0.):
        """Initializes capacitor element.

        Args:
            farads: Farads
            initial_voltage: Forced voltage at start of sim
        """

        self.farads = farads
        self.initial_voltage = initial_voltage
        self.voltage = 0.


class Source(SwitchedComponent, ABC):
    def __init__(self, duration: float, dt: float):
        """Initialize power source.

        Args:
            duration: Length of power traces (s)
            dt: Sampling period (ms)

        """

        self.duration = duration
        self.dt = dt

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
    def __init__(self, voltage: float, **kwargs):
        """Initial data for a constant voltage source.

        Args:
            voltage: Voltage in volts
        """

        self.voltage = voltage

        # this init must be after setting variables that are used in load_data.
        # The Source derived class calls the abstract method load_data so they
        # have to be set before the super call.
        Source.__init__(self, **kwargs)

    def load_data(self):
        #steps = self.duration * self.sample_hz

        curr_time = pd.Timestamp.now()

        start = curr_time
        end = curr_time + pd.Timedelta(self.duration, "s")

        # Datetime timestamps
        timestamps = pd.date_range(
            start=start,
            end=end,
            #periods=steps,
            freq=pd.Timedelta(self.dt, "ms")
        )

        # Generate voltage
        vs = np.ones(len(timestamps))
        vs *= self.voltage

        self.data = pd.DataFrame({
            'Timestamp': timestamps,
            'Potential(V)': vs
        })


class SineSource(Source):
    def __init__(
        self,
        source_am,
        source_os,
        source_hz,
        source_ph,
        **kwargs
    ):
        """Initializes the SineSource

        Args:
            source_am: source amplitude (V)
            source_os: source voltage offset (V)
            source_hz: frequency of the source (Hz)
            source_ph: phase offset of the source in radians
        """

        self.source_am = source_am
        self.source_os = source_os
        self.source_hz = source_hz
        self.source_ph = source_ph

        # this init must be after setting variables that are used in load_data.
        # The Source derived class calls the abstract method load_data so they
        # have to be set before the super call.
        Source.__init__(self, **kwargs)


    def load_data(self):
        sample_hz = 1 / self.dt
        steps = int(self.duration * sample_hz)

        # Elapsed time in seconds, used for generating the waveform
        ts = np.linspace(0, self.duration, steps, endpoint=False)

        # Datetime timestamps
        timestamps = pd.date_range(
            start=pd.Timestamp.now(),
            periods=steps,
            freq=pd.Timedelta(seconds=1 / sample_hz)
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
    # TODO: subclass with computational state machine (states and costs)
    # TODO: step through states in callback
    def __init__(self):
        pass


class SMSink(Sink):
    def __init__(self, **kwargs):
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

    def reset(self):
        """Disconnect all switches."""

        self.src.reset()
        for cap in self.caps:
            cap.reset()
        self.sink.reset()

    def callback(self, time: float):
        """Callback function to perform actions during sim runtime.

        This function should be overwritten by the user.

        Args:
            time: Timestep of simulation
        """

        return


class CapacitorStorageSim:
    class CustomShared(NgSpiceShared):
        """Class that takes in a callback and updates the current state of the
        switches.

        The functions `get_vsrc_data` / `get_isrc_data` are called for every
        source before `send_data` is called.

        The simulation starts at time zero.

        The callback function has parameters time and voltages of each
        capacitor. It returns a tuple for the load switch and switch state list
        that includes all capacitors.
        """

        def __init__(self, config: CapacitorStorageSimConfig, **kwargs):
            """Sets the callback function."""

            # This line allows for mulitple instantiations of ngspice. See
            # following for source.
            # https://github.com/PySpice-org/PySpice/pull/94
            PySpice.Spice.NgSpice.Shared.ffi = FFI()

            super().__init__(**kwargs)
            self.config = config

        def get_vsrc_data(self, voltage, time, node, ngspice_id):
            self._logger.debug(
                f"ngspice_id-{ngspice_id} get_vsrc_data @{time} node {node}"
            )

            # TODO Update to configured power source
            # this is constant
            if node == "v_src":
                voltage[0] = self.config.source.get_voltage()

            # TODO update power source. Voltage equals power.
            if node == "v_pwr_src":
                voltage[0] = self.config.source.get_power(time)

            # TODO Update for constant sink power. Voltage equals power.
            if node == "v_pwr_sink":
                # TODO check sink power state and set deep sleep
                voltage[0] = self.config.sink.get_power(time)

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
                'l.x0.l1#branch': 0j, 'l.x1.l1#branch': 0j, 'x1.3': 0j, 'x1.2': 0j,
                'c1+': 0j, 'vsw1+': 0j, 'x0.3': 0j, 'x0.2': 0j, 'c0+': 0j, 'vsw0+':
                0j, 'output': 0j, 'input': 0j, 'time': (2e-05+0j)}
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

        # limited min/max resistance for load
        self.load_min_r = 1e-9
        self.load_max_r = 1e9

    def _create_circuit(self, model: str = "C_real") -> Circuit:
        """Creates the circuit model.

        Available fields for model is "C_real" and "C_ideal". At time of writing
        the capacitor model incorperates "Resr", "Rleak", "Cval", "fo".

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


        # old input model
        #circuit.V("_src", "v_src_pos", circuit.gnd, "dc 0 external")
        #circuit.R(1, "v_src_pos", "src", 2.2 @ u_kOhm)

        # input source
        # TODO Chance to the param for constant voltage source
        circuit.V("_src", "v_src_pos", circuit.gnd, "dc 0 external")

        circuit.V("_pwr_source", "v_pwr_source", circuit.gnd, "dc 0 external")
        circuit.raw_spice += "R1 v_src_pos src {V(output)**2/v_pwr_source}\n"


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

        # old resistive load
        #circuit.R(2, "sink", circuit.gnd, 200 @ u_Ohm)

        # new voltage controlled resistive load
        circuit.V("_pwr_sink", "v_pwr_sink", circuit.gnd, "dc 0 external")
        circuit.raw_spice += f"R1 sink gnd {{limit({self.load_min_r},V(v_pwr_sink)**2/0.01,{self.load_max_r})}}\n"

        return circuit

    def _simulate(self, circuit: Circuit):

        simulator = circuit.simulator(
            temperature=25,
            nominal_temperature=25,
            simulator="ngspice-shared",
            ngspice_shared=self.shared,
        )

        # Initial conditions
        ic_kwargs = {}
        for idx, cap in enumerate(self.config.caps):
            ic_kwargs[f"c{idx}_pos"] = cap.initial_voltage
        simulator.initial_condition(**ic_kwargs)

        analysis = simulator.transient(
            step_time=self.config.src.dt @ u_ms,
            end_time=self.config.src.duration @ u_s,
            use_initial_condition=True,
        )

        return analysis

    def run(self):
        """Run the simulation on a set of capacitor values.

        Args:
            caps: Capacitor array values
        """

        self.circuit = self._create_circuit()
        self.analysis = self._simulate(self.circuit)

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
                self.analysis[f"ctrl_src_pwr{n}_pos"], label=f"source, line {n}"
            )
            ax_idx += 1

        for n in range(self.config.p_lines):
            axs[ax_idx].plot(
                self.analysis[f"ctrl_pwr{n}_sink_pos"], label=f"sink, line {n}"
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

    def _plot_src_sink_power(self):
        _, axs = plt.subplots(2, 1, sharex=True)

        axs[0].set_title("Source/Sink Power")

        axs[0].plot(self.analysis["v_pwr_src"], label="src")
        axs[1].plot(self.analysis["v_pwr_sink"], label="sink")

        for ax in axs:
            ax.set_ylabel("Power (W)")

        axs[1].set_xlabel("Time (ms)")

        for ax in axs:
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
        self._logger.debug(f"ngspice_id-{ngspice_id} get_vsrc_data @{time} node {node}")
        voltage[0] = self._amplitude * math.sin(self._pulsation * time)
        return 0

    def get_isrc_data(self, current, time, node, ngspice_id):
        self._logger.debug(f"ngspice_id-{ngspice_id} get_isrc_data @{time} node {node}")
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
