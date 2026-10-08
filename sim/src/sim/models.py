"""Implements different models that can be used for simulation.

Ideally every model would implement a input and output source such that they
can be controlled from outside the original function.
"""

import math
import os
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Manager, Pool
from threading import Lock

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PySpice
from cffi import FFI
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.NgSpice.Simulation import TransientAnalysis
from PySpice.Unit import u_kOhm, u_ms, u_Ohm, u_s, u_V
from scipy.integrate import cumulative_trapezoid


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

    def connected(self) -> bool:
        """Checks if any swithes are connected

        Returns:
            bool
        """

        return any(self._sw_arr)


@dataclass
class BaseCapacitor(SwitchedComponent):
    """Initializes capacitor element.


    Available fields for model is "C_real" and "C_ideal".
    At time of writing the capacitor model incorperates the following as args:
        "Resr", "Rleak", "Cval", "fo".


    Attributes:
        farads: Farads
        v_min: Minimum voltage allowed
        v_max: Maximum voltage allowed
        leakage: Leakage current in A
        initial_voltage: Forced voltage at start of sim
        voltage: Internal updated voltage that should not be touched
        model: Spice model that is used
    """

    farads: float
    v_min: float = 1.6
    v_max: float = 3.3
    _leakage: float = 3e-6
    initial_voltage: float = 0.0
    voltage: float = 0.0
    model: str = ""
    library: str = ""

    @property
    def energy(self) -> float:
        return self.voltage**2 * self.farads / 2

    @property
    def min_energy(self) -> float:
        return self.v_min**2 * self.farads / 2

    @property
    def leakage(self) -> float:
        return self._leakage

    def library_realpath(self) -> str:
        """Get the resolved path of the library."""

        file_path = os.path.abspath(__file__)
        spice_path = os.path.join(os.path.dirname(file_path), "spice")
        lib_path = os.path.join(spice_path, self.library)

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library does not exist at {lib_path}")

        return lib_path

    def model_kwargs(self):
        """Gets kwargs needed for spice model parameters"""

        return {}


@dataclass
class Capacitor(BaseCapacitor):
    """Simplified real capacitor with adjustable characteristics.

    Attributes:
        r_esr: Equivalent series resistance.
        r_leak: Leakage resistance.
        fo: Resonant frequency.
    """

    r_esr: float = 0.03
    r_leak: float = 100e12
    fo: float = 1e6

    model: str = "C_real"
    library: str = "cap.lib"

    def model_kwargs(self):
        return {
            "Resr": self.r_esr,
            "Rleak": self.r_leak,
            "Cval": self.farads,
            "fo": self.fo,
        }


@dataclass
class IdealCapacitor(BaseCapacitor):
    model: str = "C_ideal"
    library: str = "cap.lib"

    def model_kwargs(self):
        return {
            "C": self.farads,
        }


@dataclass
class LeacsCapacitor(Capacitor):
    """Leacs style capacitor model.

    Assumes the leakage current is determined by `0.1 * C * V` with a minimum
    at the `leak_floor`.
    """

    leak_floor: float = 3e-6

    @property
    def leakage(self):
        return max(0.01 * self.farads * self.voltage, self.leak_floor)


@dataclass
class VeryLeakyCapacitor(Capacitor):
    """Super leaky capacitor for testing."""

    r_esr: float = 0.03
    r_leak: float = 10e6
    fo: float = 1e6

    leak_floor: float = 3e-6


@dataclass
class CVCapacitor(BaseCapacitor):
    """Capacitor with leakage current dependent on CV.

    Attributes:
        k: Scaling factor.
        Imin: Minimum leakage current.
        Rmin: Minimum leakage resistance.
        Rmax: Maximum leakage resistance.
    """

    # spice parameters
    k: float = 0.01
    i_min: float = 1e-6
    r_min: float = 1
    r_max: float = 1e9

    model: str = "C_cv"
    library: str = "cap.lib"

    def model_kwargs(self):
        return {
            "Cval": self.farads,
            "k": self.k,
            "Imin": self.i_min,
            "Rmin": self.r_min,
            "Rmax": self.r_max,
        }

    @property
    def leakage(self):
        return max(0.01 * self.farads * self.voltage, self.leak_floor)


class PanasonicCapacitors:
    """Collection of Panasonic capacitors.

    Collected from: https://industrial.panasonic.com/ww/downloads/simulation-data

    These were selected based on the following criteria:
        - For a given arch (Polymer Aluminium, Aluminium electrolytic, etc) the
          lowest ESR was chosen. This was done because ESR was given in the
          table and the leakage current was not.
        - The steps was determined by E6 values.
        - The max voltage had to be above 3.3V.
    """

    class SPCAP:
        """Conductive polymer aluminium capacitors."""

        @dataclass
        class C100uF(BaseCapacitor):
            """
            https://industrial.panasonic.com/cdbs/www-data/pdf/ABE0000/ABE0000C79.pdf
            """

            farads: float = 100e-6
            v_max: float = 4.0
            library: str = "EEFSX0G101ER.lib"
            model: str = "EEFSX0G101ER"

            @property
            def leakage(self) -> float:
                return 0.1 * self.farads * self.voltage

        @dataclass
        class C150uF(BaseCapacitor):
            """
            https://industrial.panasonic.com/cdbs/www-data/pdf/ABE0000/ABE0000C79.pdf
            """

            farads: float = 150e-6
            v_max: float = 4.0
            library: str = "EEFSX0G151E7.lib"
            model: str = "EEFSX0G151E7"

            @property
            def leakage(self) -> float:
                return 0.1 * self.farads * self.voltage

        @dataclass
        class C220uF(BaseCapacitor):
            """
            https://industrial.panasonic.com/cdbs/www-data/pdf/ABE0000/ABE0000C79.pdf
            """

            farads: float = 220e-6
            v_max: float = 4.0
            library: str = "EEFSX0G221ER.lib"
            model: str = "EEFSX0G221ER"

            @property
            def leakage(self) -> float:
                return 0.1 * self.farads * self.voltage

        @dataclass
        class C330uF(BaseCapacitor):
            """
            https://industrial.panasonic.com/cdbs/www-data/pdf/ABE0000/ABE0000C79.pdf
            """

            farads: float = 330e-6
            v_max: float = 4.0
            library: str = "EEFSX0G331XE.lib"
            model: str = "EEFSX0G331XE"

            @property
            def leakage(self) -> float:
                return 0.1 * self.farads * self.voltage

        @dataclass
        class C470uF(BaseCapacitor):
            """
            https://industrial.panasonic.com/cdbs/www-data/pdf/ABE0000/ABE0000C106.pdf
            """

            farads: float = 470e-6
            v_max: float = 4.0
            library: str = "ECGSY0G471R.lib"
            model: str = "ECGSY0G471R"

            @property
            def leakage(self) -> float:
                return 0.1 * self.farads * self.voltage

    class POSCAP:
        """Conductive Polymer Tantalum Solid Capacitors"""

        @dataclass
        class C100uF(BaseCapacitor):
            """
            https://industrial.panasonic.com/cdbs/www-data/pdf/AAA8000/AAA8000C74.pdf
            """

            farads: float = 100e-6
            v_max: float = 6.3
            _leakage: float = 63e-6
            library: str = "6TCE100MI.lib"
            model: str = "6TCE100MI"

        @dataclass
        class C150uF(BaseCapacitor):
            """
            https://industrial.panasonic.com/cdbs/www-data/pdf/AAA8000/AAA8000C74.pdf
            """

            farads: float = 150e-6
            v_max: float = 6.3
            _leakage: float = 94.5e-6
            library: str = "6TCE150MF.lib"
            model: str = "6TCE150MF"

        @dataclass
        class C220uF(BaseCapacitor):
            """
            https://industrial.panasonic.com/cdbs/www-data/pdf/AAA8000/AAA8000C74.pdf
            """

            farads: float = 220e-6
            v_max: float = 6.3
            _leakage: float = 138.6e-6
            library: str = "6TCF220ML.lib"
            model: str = "6TCF220ML"

        @dataclass
        class C330uF(BaseCapacitor):
            """
            https://industrial.panasonic.com/cdbs/www-data/pdf/AAA8000/AAA8000C74.pdf
            """

            farads: float = 330e-6
            v_max: float = 6.3
            _leakage: float = 207.9e-6
            library: str = "6TCF330M9L.lib"
            model: str = "6TCF330M9L"

        @dataclass
        class C470uF(BaseCapacitor):
            """
            https://industrial.panasonic.com/cdbs/www-data/pdf/AAA8000/AAA8000C74.pdf
            """

            farads: float = 470e-6
            v_max: float = 4.0
            _leakage: float = 188.0e-6
            library: str = "4TCF470ML.lib"
            model: str = "4TCF470ML"

        @dataclass
        class C680uF(BaseCapacitor):
            """
            https://industrial.panasonic.com/cdbs/www-data/pdf/AAA8000/AAA8000C74.pdf
            """

            farads: float = 680e-6
            v_max: float = 4.0
            _leakage: float = 272.0e-6
            library: str = "4TCF470ML.lib"
            model: str = "4TCF470ML"


class Source(SwitchedComponent, ABC):
    def __init__(self, duration: float, dt: float, voltage: float = 3.3):
        """Initialize power source.

        Args:
            duration: Length of power traces (s)
            dt: Sampling period (ms)

        """

        self.duration = duration
        self.dt = dt
        self.voltage = voltage

    def get_power(self, time: float) -> float:
        """Gets the power consumption at a given timestep.

        Args:
            time: Number of seconds since the start of the sim.

        Returns:
            Power in watts.
        """

        return 0.1

    def get_voltage(self) -> float:
        """Gets the constant voltage level of the source

        Returns:
            Voltage in volts.
        """

        return self.voltage

    def next(self):
        """Function called when the simulation time is advanced."""

        return


class ConstantSource(Source):
    def __init__(self, power: float, **kwargs):
        """Initial data for a constant voltage source.

        Args:
            voltage: Voltage in volts.
            power: Power supplied in watts.
        """

        self.power = power

        # this init must be after setting variables that are used in load_data.
        # The Source derived class calls the abstract method load_data so they
        # have to be set before the super call.
        Source.__init__(self, **kwargs)

    def get_power(self, time: float):
        return self.power


class SineSource(Source):
    def __init__(
        self,
        source_am: float,
        source_os: float,
        source_hz: float,
        source_ph: float,
        **kwargs,
    ):
        """Initializes the SineSource

        Args:
            source_am: source amplitude (V)
            source_os: source voltage offset (V)
            source_hz: frequency of the source (Hz)
            source_ph: phase offset of the source in radians
        """

        # Check offset is greater than amplitude to prevent "negative" power
        # readings
        if source_am > source_os:
            raise RuntimeError("Amplitude cannot be greater than offset.")

        self.source_am = source_am
        self.source_os = source_os
        self.source_hz = source_hz
        self.source_ph = source_ph

        # this init must be after setting variables that are used in load_data.
        # The Source derived class calls the abstract method load_data so they
        # have to be set before the super call.
        Source.__init__(self, **kwargs)

    def get_power(self, time: float):
        vs = math.sin(2.0 * math.pi * self.source_hz * time + self.source_ph)
        vs *= self.source_am
        vs += self.source_os

        return vs


class BonitoSource(Source):
    def __init__(
        self,
        filename: str,
        name: str = "node0",
        offset: float = 0,
        duration: float = 0,
        downsample: int = 1000,
        **kwargs,
    ):
        """Initializes a bonito energy source.

        Downsampling is implemented as a zero order hold of the index until the
        next data point is requested. It means we are skipping over data points
        rather than doing any type of averaging.

        Kwarg "voltage" is required.

        Passing kwargs "dt" and "duration" override calculated values from the
        dataset. DO NOT do unless you know what you're doing.

        Args:
            file: Path to h5 data file.
            name: Name of the node.
            offset: Dataset time at start of simulation.
            duration: Time to run simulation from offset.
            downsample: Ratio to downsample datapoints.

        Raises:
            IndexErorr when the combination of offset and duration exceeds the
            time for the given dataset.
        """

        self.filename = filename
        self.name = name

        # Open file
        self.file = h5py.File(filename, "r")

        self.offset = offset
        self.duration = duration
        self.downsample = downsample

        # Convert time inputs to indexes
        dt = (self.file["time"][1] - self.file["time"][0]) * self.downsample
        self.idx = int(offset / dt)
        self.max_idx = int(duration / dt) + self.idx

        # Check max index is not out of range of data
        if self.max_idx > len(self.file["time"]):
            raise IndexError("Max time exceeds input data.")

        # Initialize the source class
        Source.__init__(self, duration=duration, dt=dt, **kwargs)

    def __del__(self):
        """Closes the open file."""

        self.file.close()

    def get_power(self, time: float):
        """Gets the next power checking against the current sim time


        Args:
            time: Simulation time.
        """

        # iterate until we get to the next timestamp
        # TODO (jmadden173): Can implement some sort of binary search to make
        # this go after
        while self.file["time"][self.idx] < time:
            self.idx += self.downsample

        # check that simulation and data class are synced
        # sim_time = (self.file["time"][self.idx] - self.offset)
        # dt = sim_time - time
        # if abs(dt) >= self.dt:
        #    raise RuntimeError(f"Simulation ({sim_time}) and data timestamp ({time}) is not synced.")

        # get power from file
        power = self.file["data"][self.name][self.idx]
        return power


class Sink(SwitchedComponent):
    def __init__(self, v_min: float = 1.65):
        """Initializes a sink

        Any classes that inherit from this should use `Sink.__init(self,
        **kwargs)` at the bottom of their `__init__` function.

        The param `sink_min_v` sets the minimum voltage that the sink starts
        the draw power. Otherwise the sink goes into a high impedance state.
        This is meant to simulate microcontrollers that have a minimum voltage
        required to operate. It could also be a minimum voltage for a boost
        converter to function.

        Args:
            v_min: Minimum voltage required for sink.
        """

        self.v_min = v_min

    def get_power(self) -> float:
        """Gets the current power draw of the sink.

        Returns:
            Power in W. Negative.
        """

        return 0.0


class ConstantSink(Sink):
    def __init__(self, power: float, **kwargs):
        """Sets constant power draw on sink

        Args:
            power: Power in watts
        """

        self.power = power

        Sink.__init__(self, **kwargs)

    def get_power(self) -> float:
        """Gets power drain

        Returns:
            Power in watts.
        """

        return self.power


class SMSink(Sink):
    def __init__(self, sink, **kwargs):
        """
        sm_states is a state machine instance representing an intermittent
        computing platform with active and passive states

        the active state cycles through the substates measure, analyze, and
        save, each with a cost to execute, if power allows, else it sleeps

        the passive state recharges until the wake threshold is reached,
        then wakes
        """

        self.sm = sink

        Sink.__init__(self, **kwargs)

    def run_sm(self):
        self.sm.send("cycle")

        current_id = self.sm._get_current_state_id()
        if current_id == "sleep" and self.sm.charged():
            # energy = self.sm.cap.energy
            # min_energy = self.sm.cap.min_energy
            # projected_energy = energy + self.sm.load_value * self.sm.remaining_time
            #
            # print(f'{self.sm.time:.6f}: waking ({energy}, {projected_energy}, {min_energy})')
            self.sm.send("wake")

    def get_power(self) -> float:
        """Gets power consumption of current state.

        Returns:
            Power in W. Negative
        """

        return self.sm.load_value




class CapacitorStorageSimConfig:

    # number of config objects
    count = 0

    def __init__(
        self,
        src: Source,
        caps: list[Capacitor],
        sink: Sink,
        p_lines: int,
        name: str = "",
    ):
        """
        Power lines connect the power source to the sink. This allows for
        multiple capacitors to charge while one is discharging.

        If name is left default then it will get autogenerated based on
        a global config number.

        Args:
            cb: Callback function
            src: Power source
            caps: Capacitor array values
            sink: Power sink
            p_lines: Number of available power lines
            name: Configuration name
        """

        # save parameters
        self.src = src
        self.caps = caps
        self.sink = sink
        self.p_lines = p_lines

        if name:
            self.name = name
        else:
            self.name = f"Config {CapacitorStorageSimConfig.count}"
        CapacitorStorageSimConfig.count += 1

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


    def farads(self) -> list[float]:
        """Gets list of capacitor farad values.

        Returns:
            List of farad values.
        """

        return [c.farads for c in self.caps]

    def callback(self, time: float):
        """Callback function to perform actions during sim runtime.

        This function should be overwritten by the user.

        Args:
            time: Timestep of simulation
        """

        return


class SimulationResults:
    """Container for storing simulation results.

    """

    def __init__(
        self,
        ts: dict | None = None,
        config: CapacitorStorageSimConfig | None = None,
    ):
        """Initialize results.

        The timeseries param is a dict with keys as the name and the values a numpy
        array of the data. This is meant to mimic the names of the plots in the
        plotter.

        Args:
            ts: Simulation timeseries data
            config: Simulation config
        """



        # Sve for later

        #src_energy: Energy produced from the source.
        #sink_energy: Energy consumed by the sink.
        #diff_energy: Difference between produce and consumed energy.
        #cap_energy: Energy remaining in capacitor.

        #src_energy: float = 0.0,
        #sink_energy: float = 0.0,
        #diff_energy: float = 0.0,
        #cap_energy: float = 0.0,

        #self.src_energy = src_energy
        #self.sink_energy = sink_energy
        #self.diff_energy = diff_energy
        #self.cap_energy = cap_energy

        if ts:
            self.ts = ts
        else:
            self.ts = {}

        self.config = config


    def get_slice(self) -> "SimulationResults":
        """Get a slice of the results.

        The data from the copied instance is still referenced to the parent
        instance.

        Returns:
            Copied slice of this instasnce.

        """

        return self



    def from_analysis(
        self,
        analysis: TransientAnalysis,
        config: CapacitorStorageSimConfig | None = None,
    ):
        """Updates values from the analysis of a simulation.

        This function maps the spice simulation analysis into a dictionary of
        numpy arrays so that they can be pickled between processes.

        It is important that the analysis is not stored in this class.

        Args:
            analysis: Results from pyspice transient simulation
            config: Simulation config.
        """

        # save config if specified
        if config:
            self.config = config


        self.ts = {}

        self.ts["time"] = np.asarray(analysis.time)

        self.ts["capacitors"] = []
        for idx, cap in enumerate(self.config.caps):
            self.ts["capacitors"].append(
                np.asarray(analysis[f"c{idx}_pos"])
            )

        # first index is capacitors, second index is power lines
        self.ts["cap_switches"] = []
        for idx, cap in enumerate(self.config.caps):
            self.ts["cap_switches"].append([])
            for line in range(self.config.p_lines):
                self.ts["cap_switches"][idx].append(
                    np.asarray(analysis[f"ctrl_pwr{line}_c{idx}_pos"])
                )

        self.ts["v_src"] = analysis["src"]
        self.ts["v_sink"] = analysis["sink"]

        self.ts["sw_src"] = []
        for n in range(self.config.p_lines):
            self.ts["sw_src"].append(
                np.asarray(analysis[f"ctrl_src_pwr{n}_pos"])
            )

        self.ts["sw_sink"] = []
        for n in range(self.config.p_lines):
            self.ts["sw_sink"].append(
                np.asarray(analysis[f"ctrl_pwr{n}_sink_pos"])
            )

        self.ts["plines"] = []
        for n in range(self.config.p_lines):
            self.ts["plines"].append(
                np.asarray(analysis[f"pwr{n}"])
            )

        self.ts["pwr_src"] = np.asarray(analysis["v_pwr_source"])
        self.ts["pwr_sink"] = np.asarray(analysis["v_pwr_sink"])

        self.ts["r_src"] = np.asarray(analysis["v_r_source"])
        self.ts["r_sink"] = np.asarray(analysis["v_r_sink"])

        self.ts["e_src"] = self._get_src_energy(analysis)
        self.ts["e_sink"] = self._get_src_energy(analysis)
        self.ts["e_diff"] = self._get_diff_energy(analysis)
        self.ts["e_sum_caps"] = self._get_sum_cap_energy(analysis)

        # array of energies in each of the caps
        self.ts["e_caps"] = self._get_cap_energy(analysis)

        self.ts["cap_current"] = []
        for idx, _ in enumerate(self.config.caps):
            self.ts["cap_current"].append(
                np.asarray(analysis[f"v_sense_{idx}"])
            )

        # Save final energies
        self.src_energy = self._get_src_energy(analysis)[-1]
        self.sink_energy = self._get_sink_energy(analysis)[-1]
        self.diff_energy = self._get_diff_energy(analysis)[-1]
        self.cap_energy = self._get_sum_cap_energy(analysis)[-1]




    def _get_src_energy(self, analysis) -> ArrayLike:
        """Get source energy at each timestep.

        Returns:
            Cumulative sum of energy at each timestep.
        """

        energy = cumulative_trapezoid(
            analysis["v_pwr_source"], analysis.time, initial=0
        )

        return np.asarray(energy)

    def _get_sink_energy(self, analysis) -> ArrayLike:
        """Get sink energy at each timestep.

        Returns:
            Cumulative sum of energy at each timestep.
        """

        energy = cumulative_trapezoid(analysis["v_pwr_sink"], analysis.time, initial=0)

        return np.asarray(energy)

    def _get_diff_energy(self, analysis) -> ArrayLike:
        """Get difference between source and sink energy.

        Returns:
            Cumulative sum of energy at each timestep.
        """

        diff = self._get_src_energy(analysis) - self._get_sink_energy(analysis)
        return diff

    def _get_cap_energy(self, analysis) -> list[ArrayLike]:
        """Get energy in each of the capacitors.

        Returns:
            Energy at each timestamp.
        """

        # energy in capacitors
        energy_list = []
        for idx, cap in enumerate(self.config.caps):
            # calculate energy from 1/2 C V^2
            energy = 0.5 * cap.farads * (analysis[f"c{idx}_pos"] ** 2)
            energy_list.append(energy)

        return energy_list

    def _get_sum_cap_energy(self, analysis) -> ArrayLike:
        """Get the sum of energy in all capacitors.

        Returns:
            Energy at each timestamp.
        """

        energy_list = self._get_cap_energy(analysis)
        total_energy = np.sum(energy_list, axis=0)

        return total_energy





class SimulationPlotter:
    def __init__(self, layout: str = "moasic"):
        """Initialize a plotter.

        Valid options for layout is the following:
            "individual": Each plot gets it's own figure.
            "moasic": All plots shown in the same figure.

        Args:
            layout: Type of plot layout.
        """

        # list of valid plot names
        plot_names = [
            # plots that show multiple timeseries data
            "capacitors",
            "cap_switches",
            "input_output",
            "io_switches",
            "power_lines",
            "src_sink_power",
            "src_sink_r",
            "energy",
            "cap_energy",
            "cap_current",

            # plots that show summary of results
            "final_energy",
        ]

        if layout == "individual":
            figs = {n: plt.figure() for n in plot_names}

        elif layout == "moasic":
            ts_fig = plt.figure(figsize=(16, 12))
            ts_subfigs = ts_fig.subfigures(4, 3)


            fin_fig = plt.figure()


            figs = {
                "input_output": ts_subfigs[0][0],
                "power_lines": ts_subfigs[0][1],
                "capacitors": ts_subfigs[0][2],
                "io_switches": ts_subfigs[1][0],
                "cap_switches": ts_subfigs[1][1],
                "src_sink_r": ts_subfigs[2][0],
                "src_sink_power": ts_subfigs[2][1],
                "cap_current": ts_subfigs[2][2],
                "energy": ts_subfigs[3][0],
                "cap_energy": ts_subfigs[3][1],

                "final_energy": fin_fig,
            }

        else:
            raise NotImplementedError(f"Layout {layout} is not implemented.")

        # make sure every valid plot name has a figure
        missing = set(plot_names) - figs.keys()
        if missing:
            raise ValueError(
                f"Layout '{layout}' is missing figures for: {sorted(missing)}"
            )

        # save params
        self.layout = layout
        self.figs = figs

    def get_fig(self, name: str) -> Figure:
        """Gets the figure object based on a name.

        Args:
            name: Name of the figure.

        Returns:
            Figure instasnce.
        """

        # check that it exists
        if name not in self.figs:
            raise NotImplementedError(f"Figure {name} is not implemented in the layout")

        return self.figs[name]


    def plot_single_ts(self, res: SimulationResults):
        """Plots single simulation results on new figure.

        Args:
            res: Simulation results
        """

        self.plot_multiple_ts([res])


    def plot_multiple_ts(self, results: list[SimulationResults]):
        """Plots multiple timeseries simulation results on new figure.

        Args:
            results: List of simulation results
        """

        self._plot_capacitors(self.get_fig("capacitors"), results)
        self._plot_cap_switches(self.get_fig("cap_switches"), results)
        self._plot_input_output(self.get_fig("input_output"), results)
        self._plot_io_switches(self.get_fig("io_switches"), results)
        self._plot_power_lines(self.get_fig("power_lines"), results)
        self._plot_src_sink_power(self.get_fig("src_sink_power"), results)
        self._plot_src_sink_r(self.get_fig("src_sink_r"), results)
        self._plot_energy(self.get_fig("energy"), results)
        self._plot_cap_energy(self.get_fig("cap_energy"), results)
        self._plot_cap_current(self.get_fig("cap_current"), results)

    def _plot_capacitors(
        self,
        fig: Figure,
        results: list[SimulationResults],
    ):
        num_caps = self._max_caps(results)
        axs = fig.subplots(num_caps, 1, sharex=True)

        # Titles
        axs[0].set_title("Capacitor Voltages")

        for res in results:
            # mapping to data
            time = res.ts["time"]
            farads = res.config.farads()
            voltages = res.ts["capacitors"]

            for idx, (farad, voltage) in enumerate(zip(farads, voltages)):
                # Voltage plot (column 0)
                axs[idx].plot(
                    time,
                    voltage,
                    label=f"{farad}",
                )

        axs[-1].set_xlabel("Time (s)")

        for ax in axs:
            ax.grid()
            ax.legend()
            axs[idx].set_ylabel("Voltage (V)")



    def _plot_cap_switches(self, fig: Figure, results: list[SimulationResults]):
        num_switches = self._max_switches(results)
        axs = fig.subplots(num_switches, sharex=True)

        axs[0].set_title("Capacitor Switch states")

        for res in results:
            # mapping to data
            time = res.ts["time"]
            caps = res.config.caps
            switches = res.ts["cap_switches"]

            row = 0
            for idx, (cap, sw) in enumerate(zip(caps, switches)):
                for line, val in enumerate(sw):
                    axs[row].plot(
                        time,
                        val,
                        label=f"C: {cap.farads} line: {line}",
                    )
                    axs[row].set_ylabel("State")
                    row += 1

        axs[-1].set_xlabel("Time (s)")

        for ax in axs:
            ax.grid()
            ax.legend()
            ax.set_ylim(-0.1, 1.1)



    def _plot_input_output(self, fig: Figure, results: list[SimulationResults]):
        axs = fig.subplots(2, 1, sharex=True)

        axs[0].set_title("Source/Sink Voltages")

        for res in results:
            time = res.ts["time"]
            v_src = res.ts["v_src"]
            v_sink = res.ts["v_sink"]

            axs[0].plot(
                time, v_src, label="src"
            )
            axs[1].plot(
                time, v_sink, label="sink"
            )

        for ax in axs:
            ax.set_ylabel("Voltage (V)")

        axs[1].set_xlabel("Time (s)")

        for ax in axs:
            ax.grid()
            ax.legend()


    def _plot_io_switches(self, fig: Figure, results: list[SimulationResults]):
        rows = 2 * self._max_plines(results)
        axs = fig.subplots(rows, sharex=True)

        axs[0].set_title("Source/Sink Switches")

        axs[-1].set_xlabel("Time (s)")

        for res in results:
            ax_idx = 0

            time = res.ts["time"]
            sw_src = res.ts["sw_src"]
            sw_sink = res.ts["sw_sink"]


            for n, data in enumerate(sw_src):
                axs[ax_idx].plot(time, data,
                    label=f"source, line {n}",
                )
                ax_idx += 1

            for n, data in enumerate(sw_sink):
                axs[ax_idx].plot(time, data,
                    label=f"sink, line {n}",
                )
                ax_idx += 1

        for ax in axs:
            ax.set_ylabel("State")
            ax.grid()
            ax.legend()
            ax.set_ylim(-0.1, 1.1)

    def _plot_power_lines(self, fig: Figure, results: list[SimulationResults]):
        rows = self._max_plines(results)
        axs = fig.subplots(rows, sharex=True)

        axs[0].set_title("Power line voltages")

        axs[-1].set_xlabel("Time (s)")

        for res in results:
            time = res.ts["time"]
            plines = res.ts["plines"]

            for n, (ax, pline) in enumerate(zip(axs, plines)):
                ax.plot(
                    time, pline,
                    label=f"line {n}",
                )
                ax.set_ylabel("Voltage (V)")
                ax.grid()
                ax.legend()

    def _plot_src_sink_power(self, fig: Figure, results:
                             list[SimulationResults]):
        axs = fig.subplots(2, 1, sharex=True)

        axs[0].set_title("Source/Sink Power")


        for res in results:
            time = res.ts["time"]
            pwr_src = res.ts["pwr_src"]
            pwr_sink = res.ts["pwr_sink"]

            axs[0].plot(time, pwr_src,
                label="src",
            )
            axs[1].plot(
                time,
                pwr_sink,
                label="sink",
            )

        for ax in axs:
            ax.set_ylabel("Power (W)")

        axs[1].set_xlabel("Time (s)")

        for ax in axs:
            ax.grid()
            ax.legend()


    def _plot_src_sink_r(self, fig: Figure, results: list[SimulationResults]):
        axs = fig.subplots(2, 1, sharex=True)

        axs[0].set_title("Source/Sink Resistance")

        for res in results:
            time = res.ts["time"]
            r_src = res.ts["r_src"]
            r_sink = res.ts["r_sink"]

            axs[0].plot(time, r_src,
                label="src",
            )
            axs[1].plot(
                time,
                r_sink,
                label="sink",
            )



        for ax in axs:
            ax.set_ylabel("Resistance (R)")

        axs[1].set_xlabel("Time (s)")

        for ax in axs:
            ax.grid()
            ax.legend()


    def _plot_energy(self, fig: Figure, results: list[SimulationResults]):
        axs = fig.subplots(4, 1, sharex=True)

        axs[0].set_title("Energy over time")


        for res in results:
            time = res.ts["time"]
            e_src = res.ts["e_src"]
            e_sink = res.ts["e_sink"]
            e_diff = res.ts["e_diff"]
            e_caps = res.ts["e_sum_caps"]

            axs[0].plot(time, e_src, label="src")
            axs[1].plot(time, e_sink, label="sink")
            axs[2].plot(time, e_diff, label="diff")
            axs[3].plot(time, e_caps, label="caps")

        for ax in axs:
            ax.grid()
            ax.legend()

        axs[1].set_ylabel("Energy (J)")
        axs[-1].set_xlabel("Time (s)")


    def _plot_cap_energy(self, fig: Figure, results: list[SimulationResults]):
        # Number of capacitors + total energy
        num_plots = self._max_caps(results) + 1
        axs = fig.subplots(num_plots, 1, sharex=True)

        axs[0].set_title("Energy in capacitors")


        for res in results:
            time = res.ts["time"]
            total_energy_list = res.ts["e_caps"]
            total_energy = res.ts["e_sum_caps"]

            for idx, (cap, energy) in enumerate(zip(res.config.caps, total_energy_list)):
                axs[idx].plot(
                    time, energy, label=f"{cap.farads}"
                )

            axs[-1].plot(
                time, total_energy, label="Total"
            )

        for ax in axs:
            ax.grid()
            ax.legend()
            ax.set_ylabel("Energy (J)")

        axs[-1].set_xlabel("Time (s)")


    def _plot_cap_current(self, fig: Figure, results: list[SimulationResults]):
        num_plots = self._max_caps(results)
        axs = fig.subplots(num_plots, 1, sharex=True)

        # Titles
        axs[0].set_title("Capacitor Currents")

        for res in results:
            time = res.ts["time"]
            current = res.ts["cap_current"]

            for idx, (cap, curr) in enumerate(zip(res.config.caps, current)):
                # Voltage plot (column 0)
                axs[idx].plot(
                    time,
                    curr,
                    label=f"{cap.farads}",
                )

        axs[-1].set_xlabel("Time (s)")
        axs[0].set_ylabel("Current (A)")

        for ax in axs:
            ax.grid()
            ax.legend()

    def cum_energy(self, v_label: str, r_label) -> ArrayLike:
        """Calculates cumulative energy given voltage and resistance.

        Args:
            v_label: Voltage label.
            r_label: Resistance label.

        Returns:
            Cumulative energy.
        """

        dt = np.diff(self.analysis.time)

        volts = self.analysis[v_label]
        res = self.analysis[r_label]
        power = volts**2 / res

        cum_energy = np.concat(([0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * dt)))

        return cum_energy



    def _max_caps(self, results: list[SimulationResults]) -> int:
        """Calculate the max number of capacitors among results.

        Args:
            results: List of simulation results

        Returns:
            Max number of capacitors among any of the simulation results.
        """

        num_caps = 0
        for res in results:
            num_caps = max(len(res.config.caps), num_caps)
        return num_caps


    def _max_switches(self, results: list[SimulationResults]) -> int:
        """Calculate the max number of switch lines

        Args:
            results: List of simulation results

        Returns:
            Max number of switches among any of the simulation results.
        """

        max_sw = 0
        for res in results:
            max_sw = max(res.config.p_lines * len(res.config.caps), max_sw)
        return max_sw

    def _max_plines(self, results: list[SimulationResults]) -> int:
        """Calculate the max number of power lines

        Args:
            results: List of simulation results

        Returns:
            Max number of power lines among any of the simulation results.
        """

        max_plines = 0
        for res in results:
            max_plines = max(res.config.p_lines, max_plines)
        return max_plines


    def plot_final(self, results: list[SimulationResults]):
        """Plots the final results from the simulation."""

        self._plot_final_energy(self.get_fig("final_energy"), results)


    def _plot_final_energy(self, fig: Figure, results: list[SimulationResults]):
        # keys in results.ts to plot, in bar order
        keys = ["e_src", "e_sink", "e_diff", "e_sum_caps"]
        n_groups = len(results)
        n_bars = len(keys)

        names = [res.config.name for res in results]

        # Okabe-Ito palette: designed to be distinguishable under the
        # common forms of color vision deficiency
        colors = [
            "#0072B2",
            "#E69F00",
            "#009E73",
            "#CC79A7",
            "#56B4E9",
            "#D55E00",
            "#F0E442",
            "#000000",
        ]
        hatches = ["", "//", "..", "xx", "\\\\", "++", "oo", "--"]

        x = np.arange(n_groups)
        width = 0.8 / n_bars  # total group width of 0.8

        ax = fig.subplots()

        for i, key in enumerate(keys):
            # final value of each run's time series
            values = [r.ts[key][-1] for r in results]
            # Offset so the bars in each group are centered on the tick
            offset = (i - (n_bars - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=key,
                color=colors[i % len(colors)],
                hatch=hatches[i % len(hatches)],
                edgecolor="black",
                linewidth=0.8,
            )
            ax.bar_label(bars, fmt="%.2g", padding=2, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Energy (J)")
        ax.set_title("Final Energy")
        ax.axhline(0, color="black", linewidth=0.8)  # diff can be negative
        ax.legend()


    def show(self):
        plt.show(block=False)
        input("Press enter to close figures...")

    def save(self, path: str, overwrite: bool = True, prefix: str = ""):
        """Saves figures to the path.

        Names are determined by configuration definitions.

        By default it will overwrite any existing figures.

        Args:
            path: Path to save figure outputs.
            overwrite: If existing files should be overwritten.
            prefix: Prefix to add to filenames.
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
        capacitor.
        It returns a tuple for the load switch and switch state list
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
            # this is constant, see logic in _crease_circuit
            if node == "v_src":
                connected = self.config.src.connected()
                if connected:
                    voltage[0] = self.config.src.get_voltage()
                else:
                    voltage[0] = 0.0

            # control the source power
            # voltage is power here
            if node == "v_pwr_source":
                connected = self.config.src.connected()
                if connected:
                    power = self.config.src.get_power(time)
                    voltage[0] = power
                else:
                    voltage[0] = 1e-9
                    # voltage[0] = 100

            # contrtol the sink power
            # voltage is power, see logic in _Create_circuit
            if node == "v_pwr_sink":
                # prevent div / 0 by setting power to small number
                if self.config.sink.get_power():
                    voltage[0] = self.config.sink.get_power()
                else:
                    voltage[0] = 1e-9

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

            # advance to next power timestamp if needed
            self.config.src.next()

            return 0

    def __init__(
        self,
        config: CapacitorStorageSimConfig,
        plotter: SimulationPlotter | None = None,
        lock: Lock | None = None,
    ):
        """Create a simulation instance with a given configuration.

        Args:
            config: Configuration
            plotter: Plotter object
            lock: Mutex lock for multiprocessing
        """

        self.config = config

        # default to an instance of plotter
        if plotter:
            self.plotter = plotter
        else:
            self.plotter = SimulationPlotter()

        # default to an instance of lock
        if lock:
            self.lock = lock
        else:
            self.lock = Lock()


        self.shared = self.CustomShared(config, send_data=True)

        # limited min/max resistance for load
        self.load_min_r = 1e-9
        self.load_max_r = 1e9

    def _create_circuit(self) -> Circuit:
        """Creates the circuit model.

        Returns:
            Circuit model object
        """

        circuit = Circuit("Capacitor Array")

        # include necessary capacitor libraries
        included_libraries = []
        for cap in self.config.caps:
            if cap.library_realpath() not in included_libraries:
                circuit.include(cap.library_realpath())
                included_libraries.append(cap.library_realpath())

        # switch model
        # threshold = 1 V
        # on resistance = 1 Ohm
        circuit.model("S", "SW", vt=1, ron=1)

        # old input model
        # circuit.V("_src", "v_src_pos", circuit.gnd, "dc 0 external")
        # circuit.R(1, "v_src_pos", "src", 2.2 @ u_kOhm)

        # input source
        # TODO Chance to the param for constant voltage source
        circuit.V("_src", "v_src_pos", circuit.gnd, "dc 0 external")
        circuit.C("_src", "src", circuit.gnd, 100e-6)

        circuit.V("_pwr_source", "v_pwr_source", circuit.gnd, "dc 0 external")
        circuit.BehavioralSource(
            "_r_source",
            "v_r_source",
            circuit.gnd,
            voltage_expression="{min(max(abs(V(v_src_pos)**2/V(v_pwr_source)), 10), 1e9)}",
        )

        circuit.raw_spice += "R1 v_src_pos src R='{V(v_r_source)}'\n"

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

            # Ammeter
            circuit.V(f"_sense_{idx}", f"c{idx}_pos", f"c{idx}_cap", 0 @ u_V)

            # capacitor
            circuit.X(idx, cap.model, f"c{idx}_cap", circuit.gnd, **cap.model_kwargs())

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
        # circuit.R(2, "sink", circuit.gnd, 200 @ u_Ohm)

        # new voltage controlled resistive load (R = V^2 / P)
        circuit.V("_pwr_sink", "v_pwr_sink", circuit.gnd, "dc 0 external")

        #
        # Direct change
        #

        # circuit.BehavioralSource(
        #   "_r_sink",
        #   "v_r_sink",
        #   circuit.gnd,
        #   voltage_expression=f"min(max(v(sink) > {self.config.sink.v_min} ? v(sink)**2 / v(v_pwr_sink) : 1e9, 1000), 1e6)",
        # )

        #
        # Smooth transition
        #

        v_min = self.config.sink.v_min
        # volts over which the load fades in
        width = 0.001
        # 1 / 1e9 ohm
        r_max = 1e9
        g_off = 1 / r_max

        # G_on = P / V^2, guarded against P = 0 and V near 0
        g_on = f"max(v(v_pwr_sink), 1e-3) / max(v(sink), {v_min})**2"
        ramp = f"u2((v(sink) - {v_min}) / {width})"

        circuit.BehavioralSource(
            "_r_sink",
            "v_r_sink",
            circuit.gnd,
            voltage_expression=f"min(1 / ({g_off} + {g_on} * {ramp}), {r_max})",
        )

        # Old voltage expression for reference. The instantaneous transition
        # was too much for the simulation to handle.
        # voltage_expression=f"{{v(sink) > {self.config.sink.v_min} ? v(sink)**2 / v(v_pwr_sink) : 1e9}}",

        # if voltage at sink > 1.65
        #   R = V^2 / P
        # else
        #   R = 1e9 (OC equivalent)
        circuit.raw_spice += "R2 sink gnd {v(v_r_sink)}\n"

        # small buffer cap
        circuit.C("_sink", "sink", circuit.gnd, 1e-6)

        return circuit

    def _simulate(self, circuit: Circuit, end_time: float = 2.0):
        self.simulator = circuit.simulator(
            temperature=25,
            nominal_temperature=25,
            simulator="ngspice-shared",
            ngspice_shared=self.shared,
        )

        # Initial conditions
        ic_kwargs = {}
        for idx, cap in enumerate(self.config.caps):
            ic_kwargs[f"c{idx}_pos"] = cap.initial_voltage
        self.simulator.initial_condition(**ic_kwargs)

        # set options
        # default gear order is 2
        self.simulator.options(
            savecurrents=True,
            method="gear",
            maxord=2,
        )

        analysis = self.simulator.transient(
            step_time=self.config.src.dt @ u_s,
            end_time=self.config.src.duration @ u_s,
            use_initial_condition=True,
        )

        return analysis

    def run(self, end_time: float = 2.0) -> SimulationResults:
        """Run the simulation on a set of capacitor values.

        Args:
            caps: Capacitor array values
            end_time: Simulation end time in seconds
        """

        circuit = self._create_circuit()
        analysis = self._simulate(circuit)

        # Calculate results
        results = SimulationResults()
        results.from_analysis(analysis, self.config)

        # save to class
        self.circuit = circuit
        self.analysis = analysis
        self.results = results

        return results


    def show(self):
        """Helper function to show plots from a simulation."""

        self.plotter.show()

    def plot(self, sl: slice = slice(None)):
        """Plots multiple subplots on given axes.

        Args:
            sl: Slice of data to plot
        """

        self.plotter.plot_single_ts(self.results)


        #self.plot_with_lock(self._plot_capacitors, "capacitors", sl=sl)
        #self.plot_with_lock(self._plot_cap_switches, "cap_switches", sl=sl)
        #self.plot_with_lock(self._plot_input_output, "input_output", sl=sl)
        #self.plot_with_lock(self._plot_io_switches, "io_switches", sl=sl)
        #self.plot_with_lock(self._plot_power_lines, "power_lines", sl=sl)
        #self.plot_with_lock(self._plot_src_sink_power, "src_sink_power", sl=sl)
        #self.plot_with_lock(self._plot_src_sink_r, "src_sink_r", sl=sl)
        #self.plot_with_lock(self._plot_energy, "energy", sl=sl)
        #self.plot_with_lock(self._plot_cap_energy, "cap_energy", sl=sl)
        #self.plot_with_lock(self._plot_cap_current, "cap_current", sl=sl)

    def plot_time(self, start: float | None = None, end: float | None = None):
        """Plot based on time.

        Similar to `plot` but instead of slice you specify start and end. Start
        and end are specified based on time since simulation start.

        Args:
            start: Start time
            end: End time
        """

        time = self.analysis.time

        # start
        if start is None:
            start_idx = 0
        else:
            start_idx = np.searchsorted(time, start, side="left")

        if end is None:
            end_idx = len(time)
        else:
            end_idx = np.searchsorted(time, end, side="right")

        self.plot(slice(start_idx, end_idx))

    def plot_with_lock(self, fn: Callable[Figure, slice], name: str, **kwargs):
        """Calls a plotting function with a mutex lock.

        Args:
            fn: Plotting function.
            name: Name of the plot.
            **kwargs: Passed to fn.
        """

        fig = self.plotter.get_fig(name)
        if self.lock:
            with self.lock:
                fn(fig, **kwargs)
        else:
            fn(fig, **kwargs)
        self.plotter.set_fig(name, fig)


    def save(self):
        pass


class SimulationRunner:
    """Runs one or more instances of CapacitorStorageSim"""

    def __init__(
        self,
        configs: list[CapacitorStorageSimConfig],
        names: list[str],
        nproc: int | None = None,
        plotter: SimulationPlotter | None = None,
    ):
        """
        Initialize a runner to parallelize simulations.

        Names are used to label plots.

        Args:
            configs: Simulation configurations.
            names: Readable names for simulations.
            nproc: Number of processes to spawn.
            plotter: Simluation plotter.
        """

        self.configs = configs
        self.names = names

        self.nproc = nproc

        # propogate plotter to simulations
        if plotter:
            self.plotter = plotter
        else:
            self.plotter = SimulationPlotter()

    def add(self, configs: list[CapacitorStorageSimConfig]):
        """Adds additional simulation configurations to run.


        Args:
            configs: Set of simulation configurations.
        """

        # save to class
        self.configs = configs

        for c in configs:
            pass

    def run(self, end_time: float = 2.0) -> list[SimulationResults]:
        """Runs simulations concurrently.

        Args:
            end_time: Duration of simulation.

        Returns:
            Dictionary containing the results.
        """

        with Manager() as m:

            # TODO Add progress bar with thread here

            with Pool(self.nproc) as p:
                results = p.map(self._run_sim, self.configs)

            #plotter.show()

        self.results = results

        return results

    def _run_sim(
        self,
        config: CapacitorStorageSimConfig,
    ) -> SimulationResults:
        """Runs a single instance of the simulation.

        Args:
            config: Simulation config
            plotter: Figures to store plots
            lock: Mutex lock for multiprocessing

        Returns:
            Final simuliation results.
        """

        sim = CapacitorStorageSim(config)
        result = sim.run()

        return result

    def plot(self):
        self.plotter.plot_multiple_ts(self.results)
        self.plotter.plot_final(self.results)

    def show(self):
        self.plotter.show()



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
    caplib_path = os.path.join(os.path.dir(os.path.abspath(__file__)), "spice")
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
        100 @ u_kOhm,
    )

    circuit.R(
        "1",
        "load",
        circuit.gnd,
        200 @ u_Ohm,
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
