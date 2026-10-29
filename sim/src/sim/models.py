"""Implements different models that can be used for simulation.

Ideally every model would implement a input and output source such that they
can be controlled from outside the original function.
"""

import copy
import hashlib
import math
import os
import pickle
import tempfile
import threading
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from multiprocessing import Manager, Pool
from queue import Queue

import h5py

# import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np
import PySpice
from cffi import FFI
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from PySpice.Probe.WaveForm import TransientAnalysis
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import u_kOhm, u_ms, u_Ohm, u_s, u_V
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

# disable matplotlib warnings for max figs
plt.rcParams["figure.max_open_warning"] = 0

# Suppress warnings specifically from the NgSpiceShared class
import logging

logging.getLogger("PySpice.Spice.NgSpice.Shared.NgSpiceShared").setLevel(logging.ERROR)


@dataclass(kw_only=True)
class SpiceComponent:
    """Loads a spice simulation component."""

    model: str = ""
    library: str = ""

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
class BaseSwitch(SpiceComponent):
    """Base classes for switches inside of the simulation.

    Expects the ordering to be DGS

    Attributes:
        r_on: Resistance when switched on
        gate_voltage: Level of voltage source to drive gate
    """

    r_on: float = 1.0
    gate_voltage: float = 3.3

    def dissipation(self, current: float) -> float:
        """Find power dissipation of the switch

        Args:
            current: Current in amps

        Returns:
            Energy lost in the MOSFET.
        """

        return (current**2) * self.r_on


class Switches:
    @dataclass
    class IdealSwitch(BaseSwitch):
        """Ideal switch

        Note (jmadden173): Had to modify vh (hysteresis) to high value to get
        simulation functional.
        """

        r_on: float = 0.1

        model: str = "IDEAL_PFET"
        library: str = "pfet.lib"

    @dataclass
    class DMP1555UFA(BaseSwitch):
        """Diodes DMP1555UFA

        https://www.diodes.com/assets/Datasheets/DMP1555UFA.pdf
        """

        model: str = "DMP1555UFA"
        library: str = "DMP1555UFA.spice.txt"

    @dataclass
    class DMP21D2UFA(BaseSwitch):
        """Diodes DMP21D2UFA

        https://www.diodes.com/assets/Datasheets/DMP21D2UFA.pdf
        """

        model: str = "DMP21D2UFA"
        library: str = "DMP21D2UFA.spice.txt"

    @dataclass
    class RZM002P02(BaseSwitch):
        """ROHM RZM002P02

        https://www.rohm.com/products/mosfets/small-signal/single-pch/rzm002p02-product
        """

        model: str = "RZM002P02"
        library: str = "rzm002p02.lib"

    @dataclass
    class BSS223PW(BaseSwitch):
        """Infineon BSS 223PW

        https://www.infineon.com/assets/row/public/documents/24/49/infineon-bss223pw-ds-en.pdf
        """

        model: str = "BSS223PW_L1"
        library: str = "OptiMOS_p.lib"

    @dataclass
    class SSM3J36FS(BaseSwitch):
        """Panasonic SSM3J36FS

        https://toshiba.semicon-storage.com/ap-en/semiconductor/product/mosfets/detail.SSM3J36FS.html
        """

        r_on: float = 3.6

        model: str = "PMOS_SSM3J36FS"
        library: str = "SSM3J36FS_PSpice_20160915.lib"


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
class BaseCapacitor(SwitchedComponent, SpiceComponent):
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
    v_min: float = 1.65
    v_max: float = 3.2
    _leakage: float = 3e-6
    initial_voltage: float = 1.3
    voltage: float = 0.0

    @property
    def energy(self) -> float:
        return self.voltage**2 * self.farads / 2

    @property
    def min_energy(self) -> float:
        return self.v_min**2 * self.farads / 2

    @property
    def leakage(self) -> float:
        return self._leakage


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class IdealCapacitor(BaseCapacitor):
    model: str = "C_ideal"
    library: str = "cap.lib"

    def model_kwargs(self):
        return {
            "C": self.farads,
        }


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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
    leak_floor: float = 3e-6

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

    def pre_sim(self):
        """Callback before the simulation starts."""

    def post_sim(self):
        """Callback after the simulation ends."""


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
            source_am: source amplitude (W)
            source_os: source power offset (W)
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


class RWSource(Source):
    def __init__(
        self,
        initial_mean: float = 5e-3,
        stdev: float = 1e-4,
        power_roof: float = 1e-2,
        power_floor: float = 0.0,
        seed: int = 42,
        **kwargs,
    ):
        """
        Initializes the RWSource with an initial position and a variance

        This source starts at initial_mean

        Args:
            initial_mean: initial position for the random walker (watts, W)
            stdev: source random walk standard deviation (watts, W)
            power_roof:
            power_floor:
        """

        # Check if invalid distribution parameters were specified
        if initial_mean < 0:
            raise RuntimeError("Initial mean must be nonnegative.")
        if stdev < 0:
            raise RuntimeError("Standard deviation must be nonnegative.")
        if power_floor < 0:
            raise RuntimeError("Power floor must be nonnegative.")
        if power_roof < 0:
            raise RuntimeError("Power roof must be nonnegative.")
        if power_floor > power_roof:
            raise RuntimeError("Power roof must be greater than power floor.")
        if power_floor > initial_mean or power_roof < initial_mean:
            raise RuntimeError("Initial mean must be in [power_floor, power_roof].")
        self.mean = initial_mean
        self.stdev = stdev
        self.rng = np.random.default_rng(seed=seed)

        self.power_floor = power_floor
        self.power_roof = power_roof
        self.time = None

        # this init must be after setting variables that are used in load_data.
        # The Source derived class calls the abstract method load_data so they
        # have to be set before the super call.
        Source.__init__(self, **kwargs)

    def get_power(self, time: float):
        if time != self.time:
            while True:
                delta = self.rng.normal(loc=0.0, scale=self.stdev)
                if (
                    self.mean + delta < self.power_floor
                    or self.mean + delta > self.power_roof
                ):
                    continue
                else:
                    break
            self.mean += delta
            self.time = time
        return self.mean


class BonitoSource(Source):
    def __init__(
        self,
        filename: str = "",
        datapath: str = "",
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

        Filename defaults to `pwr_cars.h5`. Datapath defaults to `data/bonito/`

        Args:
            filename: Path to h5 data file.
            datapath: Path to data sources.
            name: Name of the node.
            offset: Dataset time at start of simulation.
            duration: Time to run simulation from offset.
            downsample: Ratio to downsample datapoints.

        Raises:
            IndexErorr when the combination of offset and duration exceeds the
            time for the given dataset.
        """

        if datapath:
            if os.path.exists(datapath):
                self.datapath = datapath
            else:
                raise FileNotFoundError(f"Folder {datapath} does not exist.")
        else:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            module_dir = os.path.join(this_dir, "..", "..")
            self.datapath = os.path.join(module_dir, "data", "bonito")

        if filename:
            self.filename = filename
        else:
            self.filename = "pwr_cars.h5"

        self.fullpath = os.path.join(self.datapath, self.filename)

        self.name = name

        self.offset = offset
        self.duration = duration
        self.downsample = downsample

        with h5py.File(self.fullpath, "r") as file:
            # Convert time inputs to indexes
            dt = (file["time"][1] - file["time"][0]) * self.downsample
            self.idx = int(offset / dt)
            self.max_idx = int(duration / dt) + self.idx

            # Check max index is not out of range of data
            if self.max_idx > len(file["time"]):
                raise IndexError("Max time exceeds input data.")

        # Initialize the source class
        Source.__init__(self, duration=duration, dt=dt, **kwargs)

    def get_power(self, time: float):
        """Gets the next power checking against the current sim time

        Args:
            time: Simulation time.
        """

        # Reuse the persistent file handle opened in pre_sim()
        # and cache dataset references to avoid repeated lookups
        time_ds = self.file["time"]
        power_ds = self.file["data"][self.name]

        # TODO (jmadden173): Can implement some sort of binary search to make
        # this go faster instead of a linear scan
        while time_ds[self.idx] < time:
            self.idx += self.downsample

        # Get power from the dataset using the persistent handle
        power = power_ds[self.idx]

        return power*64

    def pre_sim(self):
        """Open file once in simulation."""

        self.file = h5py.File(self.fullpath, "r")

    def post_sim(self):
        """Close file once simulation ends."""

        self.file.close()
        del self.file


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
        # print('cycle:', self.sm.charged())
        self.sm.send("cycle")

        current_id = self.sm._get_current_state_id()
        # print(current_id)
        if current_id == "sleep" and self.sm.charged():
            # print('wake')
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


class Event:
    """Label in time for results.

    All objects stored in this class MUST be able to be pickled.
    """

    def __init__(
        self,
        name: str,
        time: float,
        ignore: bool = False,
        failed: bool = False,
    ):
        """Initialize a results label with a name and time.

        Args:
            name: Name of event
            time: Time the event happened
        """

        self.name = name
        self.time = time
        self.ignore = ignore
        self.failed = failed

    def __eq__(self, other: object) -> bool:
        """Check that the names are equal."""

        if not isinstance(other, Event):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        """Has based on the name."""

        return hash(self.name)

    def __lt__(self, other: "Event") -> bool:
        """Comparison is done based on time."""

        if not isinstance(other, Event):
            return NotImplemented
        return self.time < other.time


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
        sw: BaseSwitch | None = None,
    ):
        """
        Power lines connect the power source to the sink. This allows for
        multiple capacitors to charge while one is discharging.

        If name is left default then it will get autogenerated based on
        a global config number.

        Args:
            src: Power source
            caps: Capacitor array values
            sink: Power sink
            p_lines: Number of available power lines
            name: Configuration name
            sw: Switch model
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

        if sw:
            self.sw = sw
        else:
            self.sw = Switches.IdealSwitch()

        # store initial configuration hash
        self.save_startup_hash()

        # Initialize number of switches
        SwitchedComponent.__init__(src, p_lines)
        SwitchedComponent.__init__(sink, p_lines)
        for cap in caps:
            SwitchedComponent.__init__(cap, p_lines)

    def save_startup_hash(self, excluded_attr: list[str] | None = None):
        """Saves the startup hash.

        Args:
            excluded_attr: Attributes to exclude during hashing
        """

        self.startup_hash = self._hash(excluded_attr)

    def _hash(self, excluded_attr: list[str] | None = None) -> str:
        """Generate a SHA-256 hash of the object's attributes.

        Attributes can be excluded by setting self.excluded_attr.

        Args:
            excluded_attr: Attributes to exclude during hashing

        Returns:
            SHA-256 hash hex string

        Raises:
            RuntimeError: When the configuration cannot be pickled.
        """

        # NOTE (jmadden173): This attr checking is really not ideal since it
        # does not extend to object instances. You can only use it on direct
        # kwargs to the classes.

        # NONE checking
        if not excluded_attr:
            excluded_attr = []

        if "p_lines" in excluded_attr:
            raise RuntimeError(
                "Cannot exclude p_lines as it changes the number of components"
            )

        # defaults to include, does not have to exist
        default_excluded_attr = ["startup_hash", "name"]

        # create copy of dict
        attr = vars(self).copy()

        # remove any user specified args
        for k in excluded_attr:
            attr.pop(k)

        # ignore if attr don't exist
        for k in default_excluded_attr:
            attr.pop(k, None)

        # catch pickling errors
        try:
            pkl = pickle.dumps(attr)
        except Exception:
            raise RuntimeError("Configuration cannot be pickled.")

        # generate hashstring
        sha256 = hashlib.sha256(pkl).hexdigest()

        return sha256

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

    def callback(self, time: float) -> Event | None:
        """Callback function to perform actions during sim runtime.

        This function should be overwritten by the user.

        Args:
            time: Timestep of simulation

        Returns:
            Event that happened at `time`
        """

        return None


class SimulationResults:
    """Container for storing simulation results."""

    def __init__(
        self,
        analysis: TransientAnalysis,
        config: CapacitorStorageSimConfig | None = None,
        events: list[Event] | None = None,
    ):
        """Initialize results.

        The timeseries param is a dict with keys as the name and the values a numpy
        array of the data. This is meant to mimic the names of the plots in the
        plotter.

        Args:
            analysis: Results from pyspice transient simulation
            config: Simulation config
        """

        if config:
            self._parse_config(config)
        else:
            self.caps: list[Capacitor] = []
            self.p_lines = 0
            self.farads: list[float] = []
            self.name = ""

        self.config = config

        self.ts: dict = {}
        self.from_analysis(analysis)

        if events:
            self.events = events
        else:
            self.events = []

    def start(self) -> float:
        """Start time of the simulation.

        Returns:
            Start time
        """

        return self.ts["time"][0]

    def end(self) -> float:
        """End time of the simulation.

        Returns:
            End time
        """

        return self.ts["time"][-1]

    def save(self, path: str = ""):
        """Save results to a path.

        Args:
            path: Save path.
        """

        if not path:
            path = self.name

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    def unique_events(self) -> list[str]:
        """Get the names of unique events.

        The list is sorted by their appearance in time.

        Returns:
            List of names
        """

        return sorted({e.name for e in self.events})

    def event_count(self, name: str = "") -> int:
        """Total number of events.

        Returns:
            Number of events
        """

        if name:
            return sum(ev.name == name for ev in self.events)

        return len(self.events)

    def event_cycles(self, start: str = "", end: str = "") -> int:
        """Number of transitions from end to start (cycles).

        Args:
            start: First event
            end: Last event

        Returns:
            Number of full cycles
        """

        events = self.unique_events()

        if not start:
            start = events[0]

        if not end:
            end = events[-1]

        print(f"start: {start}, end: {end}")

        sorted_events = sorted(self.events)
        return sum(
            (a.name == start) and (b.name == end) for a, b in pairwise(sorted_events)
        )

    def event_failures(self) -> int:
        """Number of events that failed.

        Returns:
            Number of failed events
        """

        return sum(e.failed for e in self.events)

    @staticmethod
    def load(path: str) -> "SimulationResults":
        """Load results from a path.

        Returns an instance of an object from a saved binary.

        Args:
            Load path

        Returns:
            Loaded object instance.
        """

        with open(path, "rb") as f:
            return pickle.load(f)

    def _parse_config(self, config: CapacitorStorageSimConfig):
        """Stores attributes from config to ensure objects are picklable.

        Args:
            config: Simulation config
        """

        self.caps = config.caps
        self.p_lines = config.p_lines
        self.farads = config.farads()
        self.name = config.name

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
    ):
        """Updates values from the analysis of a simulation.

        This function maps the spice simulation analysis into a dictionary of
        numpy arrays so that they can be pickled between processes.

        It is important that the analysis is not stored in this class.

        The function `_parse_config` is required to parse the configuration and
        set the internal parameters.

        Args:
            analysis: Results from pyspice transient simulation
        """

        self.ts = {}

        self.ts["time"] = np.asarray(analysis.time)

        self.ts["capacitors"] = []
        for idx, cap in enumerate(self.caps):
            self.ts["capacitors"].append(np.asarray(analysis[f"c{idx}_pos"]))

        # first index is capacitors, second index is power lines
        self.ts["cap_switches"] = []
        for idx, cap in enumerate(self.caps):
            self.ts["cap_switches"].append([])
            for line in range(self.p_lines):
                self.ts["cap_switches"][idx].append(
                    np.asarray(analysis[f"ctrl_pwr{line}_c{idx}_pos"])
                )

        self.ts["v_src"] = np.asarray(analysis["src"])
        self.ts["v_sink"] = np.asarray(analysis["sink"])

        self.ts["sw_src"] = []
        for n in range(self.p_lines):
            self.ts["sw_src"].append(np.asarray(analysis[f"ctrl_src_pwr{n}_pos"]))

        self.ts["sw_sink"] = []
        for n in range(self.p_lines):
            self.ts["sw_sink"].append(np.asarray(analysis[f"ctrl_pwr{n}_sink_pos"]))

        self.ts["plines"] = []
        for n in range(self.p_lines):
            self.ts["plines"].append(np.asarray(analysis[f"pwr{n}"]))

        self.ts["pwr_src"] = np.asarray(analysis["v_pwr_source"])
        self.ts["pwr_sink"] = np.asarray(analysis["v_pwr_sink"])

        self.ts["r_src"] = np.asarray(analysis["v_r_source"])
        self.ts["r_sink"] = np.asarray(analysis["v_r_sink"])

        self.ts["e_src"] = self._get_src_energy(analysis)
        self.ts["e_sink"] = self._get_sink_energy(analysis)
        self.ts["e_diff"] = self._get_diff_energy(analysis)
        self.ts["e_sum_caps"] = self._get_sum_cap_energy(analysis)

        # array of energies in each of the caps
        self.ts["e_caps"] = self._get_cap_energy(analysis)

        self.ts["cap_current"] = []
        for idx, _ in enumerate(self.caps):
            self.ts["cap_current"].append(np.asarray(analysis[f"v_sense_{idx}"]))

        # Save final energies
        self.src_energy = self._get_src_energy(analysis)[-1]
        self.sink_energy = self._get_sink_energy(analysis)[-1]
        self.diff_energy = self._get_diff_energy(analysis)[-1]
        self.cap_energy = self._get_final_cap_energy(analysis)

    def _get_src_energy(self, analysis) -> ArrayLike:
        """Get source energy at each timestep.

        Returns:
            Cumulative sum of energy at each timestep.
        """

        # from input power
        # energy = cumulative_trapezoid(
        #    analysis["v_pwr_source"], analysis.time, initial=0
        # )

        # from actual power floating through the load
        time = analysis.time
        voltage = analysis["src"]
        current = analysis["@br1[i]"]
        power = voltage * current
        energy = cumulative_trapezoid(power, time, initial=0)

        return np.asarray(energy)

    def _get_sink_energy(self, analysis) -> ArrayLike:
        """Get sink energy at each timestep.

        Returns:
            Cumulative sum of energy at each timestep.
        """

        energy = cumulative_trapezoid(analysis["v_pwr_sink"], analysis.time, initial=0)

        # actual power flowing through the load
        time = analysis.time
        voltage = analysis["sink"]
        current = analysis["@br2[i]"]
        power = voltage * current
        energy = cumulative_trapezoid(power, time, initial=0)

        return np.asarray(energy)

    def _get_diff_energy(self, analysis) -> ArrayLike:
        """Get difference between source, sink, and capacitor.

        Returns:
            Cumulative sum of energy at each timestep.
        """

        diff = self._get_src_energy(analysis)
        diff -= self._get_sink_energy(analysis)
        diff -= self._get_sum_cap_energy(analysis)

        return diff

    def _get_cap_energy(self, analysis) -> list[ArrayLike]:
        """Get energy in each of the capacitors.

        Returns:
            Energy at each timestamp.
        """

        # energy in capacitors
        energy_list = []
        for idx, cap in enumerate(self.caps):
            # calculate energy from 1/2 C V^2
            # ORDER matters here, see the following
            # https://github.com/jlab-sensing/shiny-train/pull/35#issuecomment-6066922555
            energy = (analysis[f"c{idx}_pos"] ** 2) * cap.farads * 0.5
            energy_list.append(np.asarray(energy))

        return energy_list

    def _get_sum_cap_energy(self, analysis) -> ArrayLike:
        """Get the sum of energy in all capacitors.

        Returns:
            Energy at each timestamp.
        """

        energy_list = self._get_cap_energy(analysis)
        total_energy = np.sum(energy_list, axis=0)

        return total_energy

    def _get_final_cap_energy(self, analysis) -> ArrayLike:
        """Get the sum of energy in all capacitors.

        Returns:
            Energy at each timestamp.
        """

        energy_list = self._get_cap_energy(analysis)

        final_energy = np.sum([arr[-1] for arr in energy_list])

        return final_energy


class SimulationPlotter:
    def __init__(self, layout: str = "mosaic"):
        """Initialize a plotter.

        Valid options for layout is the following:
            "individual": Each plot gets its own figure.
            "mosaic": All plots shown in the same figure.

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
            "final_counts",
            "final_transitions",
            # plots for events
            "event_vertical",
            "event_shaded",
            "event_line",
        ]

        if layout == "individual":
            figs = {n: plt.figure() for n in plot_names}

        elif layout == "mosaic":
            ts_fig = plt.figure(figsize=(16, 12))
            ts_subfigs = ts_fig.subfigures(4, 3)

            fin_fig = plt.figure()
            fin_subfigs = fin_fig.subfigures(3, 1)

            event_fig = plt.figure()
            event_subfigs = event_fig.subfigures(1, 3)

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
                "final_energy": fin_subfigs[0],
                "final_transitions": fin_subfigs[1],
                "final_counts": fin_subfigs[2],
                "event_vertical": event_subfigs[0],
                "event_shaded": event_subfigs[1],
                "event_line": event_subfigs[2],
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

    def plot(self, results: list[SimulationResults]):
        """Plot results

        Args:
            results: List of simulation results
        """

        self.plot_multiple_ts(results)

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
        self._plot_event_vertical(self.get_fig("event_vertical"), results)
        self._plot_event_shaded(self.get_fig("event_shaded"), results)
        self._plot_event_line(self.get_fig("event_line"), results)

        self._plot_final_energy(self.get_fig("final_energy"), results)
        self._plot_final_counts(self.get_fig("final_counts"), results)
        self._plot_final_transitions(self.get_fig("final_transitions"), results)

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
            farads = res.farads
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
            caps = res.caps
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

            axs[0].plot(time, v_src, label="src")
            axs[1].plot(time, v_sink, label="sink")

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
                axs[ax_idx].plot(
                    time,
                    data,
                    label=f"source, line {n}",
                )
                ax_idx += 1

            for n, data in enumerate(sw_sink):
                axs[ax_idx].plot(
                    time,
                    data,
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
                    time,
                    pline,
                    label=f"line {n}",
                )
                ax.set_ylabel("Voltage (V)")

        for ax in axs:
            ax.grid()
            ax.legend()

    def _plot_src_sink_power(self, fig: Figure, results: list[SimulationResults]):
        axs = fig.subplots(2, 1, sharex=True)

        axs[0].set_title("Source/Sink Power")

        for res in results:
            time = res.ts["time"]
            pwr_src = res.ts["pwr_src"]
            pwr_sink = res.ts["pwr_sink"]

            axs[0].plot(
                time,
                pwr_src,
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

            axs[0].plot(
                time,
                r_src,
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

            for idx, (cap, energy) in enumerate(zip(res.caps, total_energy_list)):
                axs[idx].plot(time, energy, label=f"{cap.farads}")

            axs[-1].plot(time, total_energy, label="Total")

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

            for idx, (cap, curr) in enumerate(zip(res.caps, current)):
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
            num_caps = max(len(res.caps), num_caps)
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
            max_sw = max(res.p_lines * len(res.caps), max_sw)
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
            max_plines = max(res.p_lines, max_plines)
        return max_plines

    def _plot_final_energy(self, fig: Figure, results: list[SimulationResults]):
        ax = fig.subplots()

        # Okabe-Ito colorblind-safe palette

        colors = {
            "Source": "tab:blue",
            "Sinked": "tab:orange",
            "Remaining": "tab:green",
            "Lost": "tab:red",
        }
        hatches = {
            "Source": "",
            "Sinked": "",
            "Remaining": "//",
            "Lost": "..",
        }
        bar_style = {
            "edgecolor": "black",
            "linewidth": 0.6,
        }

        run_names = [res.name for res in results]
        x = np.arange(len(run_names))
        width = 0.35
        gap = 0.0
        offset = (width + gap) / 2

        src = np.array([res.src_energy for res in results])
        sink_energies = {
            "Sinked": np.array([res.sink_energy for res in results]),
            "Remaining": np.array([res.cap_energy for res in results]),
            "Lost": np.array([res.diff_energy for res in results]),
        }

        ax.bar(
            x - offset,
            src,
            width,
            label="Source",
            color=colors["Source"],
            hatch=hatches["Source"],
            **bar_style,
        )

        bottom = np.zeros(len(results))
        for label, energy in sink_energies.items():
            ax.bar(
                x + offset,
                energy,
                width,
                bottom=bottom,
                label=label,
                color=colors[label],
                hatch=hatches[label],
                **bar_style,
            )
            bottom += energy

        ax.set_xticks(x)
        ax.set_xticklabels(run_names)
        ax.set_ylabel("Energy (J)")
        ax.axhline(0, color="black", linewidth=0.8)

        # Clean, publication-style axes
        ax.tick_params(direction="out")
        ax.legend()

    def _plot_final_counts(self, fig: Figure, results: list[SimulationResults]):
        """Plot the event counts per run."""

        if not self._all_events(results):
            self._plot_text(fig, "No events")
            return

        ax = fig.subplots()

        run_names = [res.name for res in results]

        # get unique names among all results
        event_names = set()
        for res in results:
            event_names |= set(res.unique_events())

        # check for no events
        if not event_names:
            return

        # build dictionary for grouped bar
        event_counts = {name: [] for name in event_names}
        for res in results:
            for name in res.unique_events():
                event_counts[name].append(res.event_count(name))

        bar = ax.grouped_bar(event_counts, tick_labels=run_names, group_spacing=1)

        # modify bars
        hatches = ["", "//", "\\\\", "xx", "..", "++", "oo", "--", "**"]
        for i, container in enumerate(bar.bar_containers):
            for rect in container:
                rect.set_hatch(hatches[i % len(hatches)])
                rect.set_edgecolor("black")
                rect.set_linewidth(0.5)
            ax.bar_label(container, padding=3)

        ax.set_ylabel("Count")
        # Account for space above the bars
        ax.margins(y=0.25)
        ax.legend()

    def _plot_final_transitions(self, fig: Figure, results: list[SimulationResults]):
        """Plot the number of transitions per run."""

        if not self._all_events(results):
            self._plot_text(fig, "No events")
            return

        ax = fig.subplots()

        run_names = [res.name for res in results]

        transitions = {
            "Total": [res.event_count() for res in results],
            "Cycles": [res.event_cycles() for res in results],
            "Failure": [res.event_failures() for res in results],
        }

        bar = ax.grouped_bar(transitions, tick_labels=run_names, group_spacing=1)

        # modify bars
        hatches = ["", "//", "\\\\", "xx", "..", "++", "oo", "--", "**"]
        for i, container in enumerate(bar.bar_containers):
            for rect in container:
                rect.set_hatch(hatches[i % len(hatches)])
                rect.set_edgecolor("black")
                rect.set_linewidth(0.5)
            ax.bar_label(container, padding=3)

        ax.set_ylabel("Count")
        # Account for space above the bars
        ax.margins(y=0.25)
        ax.legend()

    def _plot_event_vertical(self, fig: Figure, results: list[SimulationResults]):
        """Plots vertical lines at each event."""

        if not self._all_events(results):
            self._plot_text(fig, "No events")
            return

        num_plots = len(results)
        axs = fig.subplots(num_plots, sharex=True, squeeze=False)

        colors: dict[str, str] = {}
        for ax, result in zip(axs[:][0], results):
            seen = set()
            for event in result.events:
                # sets the color if it does not exist
                color = colors.setdefault(event.name, f"C{len(colors)}")
                label = event.name if event.name not in seen else None
                ax.axvline(x=event.time, label=label, color=color)
                seen.add(event.name)

            ax.set_ylabel(result.name)

        for ax in axs[:][0]:
            # remove ticks
            ax.set_yticks([])

            ax.set_xlim(self._start(results), self._end(results))

            ax.grid()
            ax.legend()

        axs[-1][0].set_xlabel("Time (s)")

    def _plot_event_shaded(self, fig: Figure, results: list[SimulationResults]):
        """Plots shaded regions in time between events."""

        if not self._all_events(results):
            self._plot_text(fig, "No events")
            return

        num_plots = len(results)
        axs = fig.subplots(num_plots, 1, sharex=True, squeeze=False)

        colors: dict[str, str] = {}
        for ax, result in zip(axs[:][0], results):
            seen = set()
            events = sorted(result.events)

            for start, end in zip(events, events[1:]):
                # skip if flagged to ignore
                if start.ignore:
                    continue

                color = colors.setdefault(start.name, f"C{len(colors)}")
                label = start.name if start.name not in seen else None
                ax.axvspan(start.time, end.time, color=color, alpha=0.3, label=label)
                seen.add(start.name)

            ax.set_ylabel(result.name)

        for ax in axs[:][0]:
            # remove ticks
            ax.set_yticks([])
            ax.set_xlim(self._start(results), self._end(results))
            ax.grid(axis="x")
            ax.legend()

        axs[-1][0].set_xlabel("Time (s)")

    def _plot_event_line(self, fig: Figure, results: list[SimulationResults]):
        """Plot lines between events with color-coded markers at each transition."""

        if not self._all_events(results):
            self._plot_text(fig, "No events")
            return

        num_plots = len(results)
        axs = fig.subplots(num_plots, 1, sharex=True, sharey=True, squeeze=False)[:, 0]

        # Discrete level and color per event name, shared across all subplots
        levels: dict[str, int] = {}
        for result in results:
            for event in sorted(result.events):
                levels.setdefault(event.name, len(levels))
        colors = {name: f"C{i}" for name, i in levels.items()}

        for ax, result in zip(axs, results):
            events = sorted(result.events)

            # Connecting line drawn first, underneath the markers
            ax.plot(
                [e.time for e in events],
                [levels[e.name] for e in events],
                color="lightgray",
                zorder=1,
            )

            # One scatter call per name gives one legend entry per name
            for name, level in levels.items():
                times = [e.time for e in events if e.name == name]
                if not times:
                    continue
                ax.scatter(
                    times,
                    [level] * len(times),
                    color=colors[name],
                    label=name,
                    s=50,
                    zorder=3,
                )

            ax.set_ylabel(result.name)
            ax.set_xlim(self._start(results), self._end(results))
            ax.grid(axis="x")
            ax.legend()

        # sharey=True, so setting ticks on the first axes applies to all
        axs[0].set_yticks(list(levels.values()), labels=list(levels.keys()))
        axs[0].set_ylim(-0.5, len(levels) - 0.5)

        axs[-1].set_xlabel("Time (s)")

    def _plot_text(self, fig: Figure, text: str):
        """Plots text in place of a figure.

        Args:
            fig: Figure object
            text: Text to display
        """

        ax = fig.subplots()

        # Hide the axes borders, ticks, and labels
        ax.axis("off")

        # Display the text in the center of the figure
        ax.text(
            0.5,
            0.5,
            text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=14,
            color="gray",
            style="italic",
        )

    def _all_events(self, results: list[SimulationResults]) -> set[Event]:
        """Check if results have events."""

        events = set()
        for res in results:
            events.update(res.events)

        return events

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

    def _start(self, results: list[SimulationResults]) -> float:
        """Minimum start time for all results."""

        return min([res.start() for res in results])

    def _end(self, results: list[SimulationResults]) -> float:
        """Maximum end time for all results."""

        return max([res.end() for res in results])


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

        def __init__(
            self,
            config: CapacitorStorageSimConfig,
            progress: Callable[[float], None] | None,
            **kwargs,
        ):
            """Sets the callback function.

            Args:
                config: Simulation configuration
                progress: Callback function to take progress
            """

            # This line allows for mulitple instantiations of ngspice. See
            # following for source.
            # https://github.com/PySpice-org/PySpice/pull/94
            PySpice.Spice.NgSpice.Shared.ffi = FFI()

            super().__init__(**kwargs)
            self.config = config
            self.progress = progress
            self.events: list[Event] = []

        def _drive_voltage(self, state: int) -> float:
            """Converts a switch state to drive voltage

            Args:
                state: Switch state

            Returns
                Drive voltage
            """

            return self.config.sw.gate_voltage * (not state)

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
                        voltage[0] = self._drive_voltage(cap.state(n))

                # Set source switches
                if node == f"v_ctrl_src_pwr{n}":
                    voltage[0] = self._drive_voltage(self.config.src.state(n))

                # Set sink switches
                if node == f"v_ctrl_pwr{n}_sink":
                    voltage[0] = self._drive_voltage(self.config.sink.state(n))

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

            # uesr defined callback
            event = self.config.callback(time)
            if event:
                self.events.append(event)

            # progress callback
            if self.progress:
                self.progress(time)

            # advance to next power timestamp if needed
            self.config.src.next()

            return 0

    def __init__(
        self,
        config: CapacitorStorageSimConfig,
        plotter: SimulationPlotter | None = None,
        progress: Callable[[float], None] | None = None,
        r_min: float = 1e-3,
        r_max: float = 1e9,
        c_bypass: float = 100e-6,
    ):
        """Create a simulation instance with a given configuration.

        Args:
            config: Configuration
            plotter: Plotter object
            progress: Callback function to take progress
            r_min: Minimum resistance for power source/sinks
            r_max: Maximum resistance for power source/sinks
            c_bypass: Bypass capacitors values
        """

        self.config = config

        # default to an instance of plotter
        if plotter:
            self.plotter = plotter
        else:
            self.plotter = SimulationPlotter()

        self.shared = self.CustomShared(config, progress, send_data=True)

        # limited min/max resistance for load
        self.r_min = r_min
        self.r_max = r_max
        self.c_bypass = c_bypass

    def _create_circuit(self) -> Circuit:
        """Creates the circuit model

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

        # include switch library
        sw = self.config.sw
        circuit.include(sw.library_realpath())

        #
        # Old not needed
        #
        # switch model
        # threshold = 1 V
        # on resistance = 1 Ohm
        # circuit.model("S", "SW", vt=1, ron=1)

        # old input model
        # circuit.V("_src", "v_src_pos", circuit.gnd, "dc 0 external")
        # circuit.R(1, "v_src_pos", "src", 2.2 @ u_kOhm)

        # input source
        # TODO Chance to the param for constant voltage source
        circuit.V("_src", "v_src_pos", circuit.gnd, "dc 0 external")
        circuit.C("_src", "src", circuit.gnd, self.c_bypass)

        circuit.V("_pwr_source", "v_pwr_source", circuit.gnd, "dc 0 external")
        circuit.BehavioralSource(
            "_r_source_raw",
            "v_r_source_raw",
            circuit.gnd,
            voltage_expression="{(V(src)*V(v_src_pos) - V(src)**2)/V(v_pwr_source)}",
        )
        circuit.BehavioralSource(
            "_r_source",
            "v_r_source",
            circuit.gnd,
            voltage_expression=f"{{min(max(abs(V(v_r_source_raw)), {self.r_min}), {self.r_max})}}",
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

            #
            # Old straight through model
            #
            # circuit.S(
            #    f"_src_pwr{n}",
            #    "src",
            #    f"pwr{n}",
            #    f"ctrl_src_pwr{n}_pos",
            #    circuit.gnd,
            #    model="S",
            # )

            # d g s
            circuit.X(
                f"_src_pwr{n}",
                sw.model,
                "src",
                f"ctrl_src_pwr{n}_pos",
                f"pwr{n}",
                **sw.model_kwargs(),
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

                #
                # Old
                #
                # circuit.S(
                #    f"_pwr{n}_c{idx}",
                #    f"pwr{n}",
                #    f"c{idx}_pos",
                #    f"ctrl_pwr{n}_c{idx}_pos",
                #    circuit.gnd,
                #    model="S",
                # )

                circuit.X(
                    f"_pwr{n}_c{idx}",
                    sw.model,
                    f"c{idx}_pos",
                    f"ctrl_pwr{n}_c{idx}_pos",
                    f"pwr{n}",
                    **sw.model_kwargs(),
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

            #
            # Old
            #
            # circuit.S(
            #    f"_pwr{n}_sink",
            #    f"pwr{n}",
            #    "sink",
            #    f"ctrl_pwr{n}_sink_pos",
            #    circuit.gnd,
            #    model="S",
            # )

            circuit.X(
                f"_pwr{n}_sink",
                sw.model,
                f"pwr{n}",
                f"ctrl_pwr{n}_sink_pos",
                "sink",
                **sw.model_kwargs(),
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
        g_off = 1 / self.r_max

        # G_on = P / V^2, guarded against P = 0 and V near 0
        g_on = f"max(v(v_pwr_sink), 1e-3) / max(v(sink), {v_min})**2"
        ramp = f"u2((v(sink) - {v_min}) / {width})"

        circuit.BehavioralSource(
            "_r_sink",
            "v_r_sink",
            circuit.gnd,
            voltage_expression=f"max(min(1 / ({g_off} + {g_on} * {ramp}), {self.r_max}), {self.r_min})",
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
        circuit.C("_sink", "sink", circuit.gnd, self.c_bypass)

        return circuit

    def _simulate(self, circuit: Circuit, end_time: float = 2.0) -> tuple:
        """Run the simulation and get the results.

        Args:
            circuit: Spice circuit
            end_time: Simulation duration

        Returns:
            Tuple with PySpice analysis and events during the simulation.

        Raises:
            Runtime error when values exceed the tolerance.
        """

        self.simulator = circuit.simulator(
            temperature=25,
            nominal_temperature=25,
            simulator="ngspice-shared",
            ngspice_shared=self.shared,
        )

        # Initial conditions. With UIC there is no DC solve, so the .ic must
        # sit on the capacitor's own node (c{idx}_cap), behind the V_sense
        # ammeter; an .ic on c{idx}_pos never reaches the capacitor.
        ic_kwargs = {}
        for idx, cap in enumerate(self.config.caps):
            ic_kwargs[f"c{idx}_cap"] = cap.initial_voltage
        self.simulator.initial_condition(**ic_kwargs)

        # set options
        # default gear order is 2
        self.simulator.options(
            savecurrents=True,
            method="gear",
            maxord=2,
        )

        self.config.src.pre_sim()

        analysis = self.simulator.transient(
            step_time=self.config.src.dt @ u_s,
            end_time=self.config.src.duration @ u_s,
            use_initial_condition=True,
        )

        self.config.src.post_sim()

        # check source resistance tolerance
        r_src = np.asarray(analysis["v_r_source"])
        if self._exceeded_r_limit(r_src):
            raise RuntimeError("Source resistance exceed tolerance bounds.")

        ## check sink resistance tolerance
        r_sink = np.asarray(analysis["v_r_sink"])
        if self._exceeded_r_limit(r_sink):
            raise RuntimeError("Sink resistance exceed tolerance bounds.")

        return analysis, self.shared.events

    def _exceeded_r_limit(
        self,
        res: np.ndarray,
        tol: float = 0.01,
    ):
        """CHecks the resistance limits for power.

        Args:
            analysis: Simulation analysis
            tol: Percent tolerance
        """

        # Allow 1% tolerance between the bonds
        tol_min = self.r_min * (1.0 - tol)
        tol_max = self.r_max * (1.0 + tol)

        return res.min() <= tol_min or tol_max <= res.max()

    def run(self, end_time: float = 2.0) -> SimulationResults:
        """Run the simulation on a set of capacitor values.

        Args:
            caps: Capacitor array values
            end_time: Simulation end time in seconds
        """

        circuit = self._create_circuit()
        analysis, events = self._simulate(circuit)

        # Calculate results
        results = SimulationResults(analysis, self.config, events)

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

    # def plot_with_lock(self, fn: Callable[Figure, slice], name: str, **kwargs):
    #     """Calls a plotting function with a mutex lock.
    #
    #     Args:
    #         fn: Plotting function.
    #         name: Name of the plot.
    #         **kwargs: Passed to fn.
    #     """
    #
    #     fig = self.plotter.get_fig(name)
    #     if self.lock:
    #         with self.lock:
    #             fn(fig, **kwargs)
    #     else:
    #         fn(fig, **kwargs)
    #     self.plotter.set_fig(name, fig)

    def save(self):
        pass


def create_caps_list(
    model: Callable,
    values: list[float],
) -> list[Capacitor]:
    """Create a list of capacitors objects based on values.

    Args:
        models: Capacitor model
        values: List of farads

    Returns:
        List of capacitor objects
    """

    return [model(c) for c in values]


def simulation_config_matrix(
    cap_classes: list[Callable],
    cap_farads: list[list[float]] | list[float],
    src_classes: list[Callable],
    src_kwargs: list[dict] | dict,
    sink_classes: list[Callable],
    sink_kwargs: list[dict] | dict,
    config_classes: list[Callable],
    config_kwargs: list[dict] | dict,
    cap_kwargs: list[dict] | dict | None = None,
    names: list[str] | None = None,
) -> list[CapacitorStorageSimConfig]:
    """Returns a matrix of configurations.

    Arguments caps, sources, and sink all must be separate objects and cannot
    share memory.

    All non-class arguments are made as single or list arguments. If a single
    kwargs list is passed, then all the models are created with those
    arguments. Otherwise, for example, different config_classes require
    different kwargs, you can specify per object.

    The names of the configurations is done by the order of arguments. First
    index will refer to the first capacitor, first source configuration. The
    second index will refer to the second capacitor config, first source
    configuration. The third index will refer to the first capacitor config,
    second source configuration.

    Args:
        cap_classes: List of capacitor classes
        cap_farads: List of farads for each simulation
        cap_kwargs: kwargs to pass to the constructor
        src_classes: List of source classes
        src_kwargs: kwargs to pass to the contructor
        sink_classes: List of sink classes
        sink_kwargs: kwargs to pass to the contructor
        config_classes: List of config classes
        config_kwargs: kwargs to pass to the constructor
        names: Configuration names

    Returns:
        List of configuration objects without any shared memory.

    """

    if not cap_kwargs:
        cap_kwargs = {}
    if isinstance(cap_kwargs, dict):
        cap_kwargs = [cap_kwargs for _ in cap_classes]

    if isinstance(cap_farads[0], float):
        cap_farads = [cap_farads]

    if isinstance(src_kwargs, dict):
        src_kwargs = [src_kwargs for _ in src_classes]

    if isinstance(sink_kwargs, dict):
        sink_kwargs = [sink_kwargs for _ in sink_classes]

    if isinstance(config_kwargs, dict):
        config_kwargs = [config_kwargs for _ in config_classes]

    num_configs = (
        len(cap_classes)
        * len(cap_farads)
        * len(src_classes)
        * len(sink_classes)
        * len(config_classes)
    )
    if not names:
        names = [f"Config {i}" for i in range(num_configs)]
    else:
        if num_configs != len(names):
            raise RuntimeError(
                f"Number of names ({len(names)}) does not match number of configs ({num_configs})"
            )

    config_idx = 0

    config_objs = []
    for cfg_class, cfg_kwargs in zip(config_classes, config_kwargs):
        for sink_class, si_kwargs in zip(sink_classes, sink_kwargs):
            for src_class, sr_kwargs in zip(src_classes, src_kwargs):
                for c_farads in cap_farads:
                    for cap_class, c_kwargs in zip(cap_classes, cap_kwargs):
                        # create copies of argument
                        c_kwargs = copy.deepcopy(c_kwargs)
                        sr_kwargs = copy.deepcopy(sr_kwargs)
                        si_kwargs = copy.deepcopy(si_kwargs)
                        cfg_kwargs = copy.deepcopy(cfg_kwargs)

                        caps = [cap_class(f, **c_kwargs) for f in c_farads]
                        src = src_class(**sr_kwargs)
                        sink = sink_class(**si_kwargs)

                        config_objs.append(
                            cfg_class(
                                src=src,
                                caps=caps,
                                sink=sink,
                                name=names[config_idx],
                                **cfg_kwargs,
                            )
                        )

                        config_idx += 1

    return config_objs


class SimulationRunner:
    """Runs one or more instances of CapacitorStorageSim"""

    def __init__(
        self,
        configs: list[CapacitorStorageSimConfig],
        nproc: int | None = None,
        plotter: SimulationPlotter | None = None,
        results_dir: str = "",
    ):
        """
        Initialize a runner to parallelize simulations.

        Names are used to label plots.

        `results_dir` is used for both saving and checking that results for
        a given configuration exist. If not specified then it will create a new
        tmp directory to store them.

        Args:
            configs: Simulation configurations
            nproc: Number of processes to spawn
            plotter: Simluation plotter
            results_dir: Path to save results
        """

        self.configs = configs

        self.nproc = nproc

        if results_dir:
            self.results_dir = results_dir
        else:
            self.results_dir = tempfile.mkdtemp()

        # propogate plotter to simulations
        if plotter:
            self.plotter = plotter
        else:
            self.plotter = SimulationPlotter()

    def run(
        self,
        end_time: float = 2.0,
        progress_bar: bool = True,
    ) -> list[SimulationResults]:
        """Runs simulations concurrently.

        Args:
            end_time: Duration of simulation
            progress_bar: Show progress bar

        Returns:
            Dictionary containing the results.
        """

        with Manager() as m:
            q = m.Queue()

            if progress_bar:
                # create bars
                bars = []
                for idx, cfg in enumerate(self.configs):
                    bars.append(
                        tqdm(
                            total=cfg.src.duration,
                            desc=cfg.name,
                            position=idx,
                            bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                        )
                    )

                # function to update bars from queue
                def listener():
                    while True:
                        msg = q.get()
                        # handle sentinel indicating end of process
                        if msg is None:
                            break
                        idx, sim_time = msg
                        bars[idx].n = min(sim_time, bars[idx].total)
                        bars[idx].refresh()

                t = threading.Thread(target=listener, daemon=True)
                t.start()

            try:
                with Pool(self.nproc) as p:
                    params = []
                    for idx, cfg in enumerate(self.configs):
                        params.append((idx, cfg, q))
                    results = p.starmap(self._run_sim, params)
            finally:
                if progress_bar:
                    q.put(None)
                    t.join()
                    for b in bars:
                        b.n = b.total
                        b.refresh()
                        b.close()

        self.results = results

        return results

    def save(self):
        """Save collection of results to directory.

        Args:
            directory: Directory containing pickles
        """

        # check results exist
        if not self.results:
            raise RuntimeError("No results to save.")

        # save to directory
        for res in self.results:
            path = os.path.join(self.results_dir, f"{res.config.startup_hash}.pkl")
            res.save(path)

    @staticmethod
    def load(directory: str) -> list[SimulationResults]:
        """Load collection of results from a directory.

        Args:
            directory: Directory containing pickles

        Returns:
            List of simulation results
        """

        results = []
        for root, _, files in os.walk(directory):
            for f in files:
                path = os.path.join(root, f)
                res = SimulationResults.load(path)
                results.append(res)

        return results

    def _check_results_exist(
        self, config: CapacitorStorageSimConfig
    ) -> SimulationResults | None:
        """Checks if results already exist.

        Args:
            config: Simulation configuration

        Returns:
            Results if they exist, otherwise none.
        """

        for file in os.listdir(self.results_dir):
            basename = os.path.splitext(file)[0]
            if file.endswith(".pkl") and basename == config.startup_hash:
                return SimulationResults.load(os.path.join(self.results_dir, file))

        return None

    def _run_sim(
        self, idx: int, config: CapacitorStorageSimConfig, queue: Queue
    ) -> SimulationResults:
        """Runs a single instance of the simulation.

        Args:
            idx: Simulation index
            config: Simulation config
            queue: Message queue for progress

        Returns:
            Final simuliation results.
        """

        # load results if they exist
        result = self._check_results_exist(config)
        if result:
            return result

        last = -1.0
        step = config.src.duration / 200

        def progress(time: float):
            nonlocal last
            if time - last >= step:
                last = time
                queue.put((idx, time))

        sim = CapacitorStorageSim(config, progress=progress)
        result = sim.run()

        return result

    def plot(self):
        self.plotter.plot_multiple_ts(self.results)

    def show(self):
        self.plotter.show()


def capacitor_list(_min: float, _max: float, decade: str = "E6") -> list[float]:
    """Creates a E6 capacitor value list between two values.

    Args:
        _min: Minimum farads
        _max: Maximum farads

    Returns:
        List of capacitor values.
    """

    if decade == "E3":
        values = [10.0, 22.0, 47.0]
    elif decade == "E6":
        values = [10.0, 15.0, 22.0, 33.0, 47.0, 68.0]
    elif decade == "E12":
        values = [
            10.0,
            12.0,
            15.0,
            18.0,
            22.0,
            27.0,
            33.0,
            39.0,
            47.0,
            56.0,
            68.0,
            82.0,
        ]
    else:
        raise NotImplementedError(f"Decade {decade} is not implemented.")

    if _min <= 0 or _max <= 0:
        raise ValueError("_min and _max must be positive.")
    if _min > _max:
        raise ValueError("_min must not exceed _max.")

    # Base values span 10..82, so each one is value * 10**k.
    # Pick a range of exponents wide enough to cover [_min, _max].
    low_exp = math.floor(math.log10(_min)) - 2
    high_exp = math.ceil(math.log10(_max))

    result = []
    for exp in range(low_exp, high_exp + 1):
        for v in values:
            # Build from an integer exponent via string-free scaling;
            # round to avoid float noise like 4.7000000000000004e-06
            c = float(f"{v}e{exp}")
            if _min <= c <= _max:
                result.append(c)

    return result


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
