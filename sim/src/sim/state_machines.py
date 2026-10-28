import warnings
from dataclasses import dataclass

import numpy as np
from statemachine import HistoryState, State, StateChart

from .models import Capacitor

warnings.filterwarnings("ignore", category=DeprecationWarning)


DC_VOLTS = 3.2


def make_transition_matrix(a, b):
    # rows: source task; columns in TASK_NAMES order (measure, analyze, tx, rx)
    return {
        "measure": np.array([0, 1, 0, 0]),
        "analyze": np.array([a, 0, 1 - a, 0]),
        "tx":      np.array([0, 0, 0, 1]),
        "rx":      np.array([1 - b, 0, b, 0]),
    }


@dataclass(frozen=True)
class Task:
    cost: float
    duration: float
    name: str


class MATRSinkSM(StateChart):
    load_value = 4.64e-4 * DC_VOLTS  # OUTPUT: the energy consumption for next timestep
    TASK_NAMES = ['measure', 'analyze', 'tx', 'rx']
    wake_energy = None


    class task(State.Compound):
        measure = State(
            "measure",
            value=Task(cost=11.68e-4 * DC_VOLTS, duration=0.511, name="measure"),
        )
        analyze = State(
            "analyze",
            value=Task(cost=2 * 11.68e-4 * DC_VOLTS, duration=0.511, name="analyze"),
        )
        tx = State(
            "tx", value=Task(cost=86.52e-4 * DC_VOLTS, duration=0.285, name="tx")
        )
        rx = State(
            "rx", value=Task(cost=20.03e-4 * DC_VOLTS, duration=0.927, name="rx")
        )
        h = HistoryState(type="deep")

    sleep = State("sleep", initial=True, value=Task(cost=4.64e-4 * DC_VOLTS, duration=0.01, name="sleep"))
    # replan = State("replan", value=4.64e-4 * DC_VOLTS)

    # Self-transitions while executing, advance when done
    cycle = (
        task.measure.to.itself(cond="executing")
        | task.measure.to(task.analyze)
        | task.analyze.to.itself(cond="executing")
        | task.analyze.to(task.tx, cond="analyze_pass")
        | task.analyze.to(task.measure)  # analyze fail
        | task.tx.to.itself(cond="executing")
        | task.tx.to(task.rx)
        | task.rx.to.itself(cond="executing")
        | task.rx.to(task.measure, cond="rx_pass")
        | task.rx.to(task.tx)  # rx fail
    )

    pause = task.to(sleep)
    wake = sleep.to(task.measure, cond="no_history") | sleep.to(task.h)

    # plan = task.to(replan)
    # resume = replan.to(task.measure, cond="no_history") | replan.to(task.h)

    def __init__(self, cap=None, a=0.95, b=0.02, seed=None, **kwargs):
        # Set before super().__init__(), which enters the initial state
        self.cap = cap
        self.time = 0.0
        self.task_start = 0.0
        self.remaining_time = 0.0
        self.a = a
        self.b = b
        self.transitions = make_transition_matrix(a, b)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        super().__init__(**kwargs)


    def __getstate__(self):
        """Serialize only simulation state and configuration parameters."""
        state = self.__dict__.copy()

        # 1. Drop python-statemachine engine internals containing local closures & adapters
        engine_keys = [
            "rng",
            "_callbacks",
            "_engine",
            "_config",
            "_listeners",
            "_state_instances",
            "_substates",
            # State objects are class descriptors, not per-instance data
            "task",
            "measure",
            "analyze",
            "tx",
            "rx",
            "sleep",
            "h",
        ]
        for key in engine_keys:
            state.pop(key, None)

        # 2. Record the active state name so it can be restored on unpickle
        state["_saved_active_state_id"] = self._get_current_state_id()

        # 3. Convert transitions dictionary of ndarrays to nested lists/tuples for clean pickling
        if "transitions" in state and isinstance(state["transitions"], dict):
            state["transitions"] = {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in state["transitions"].items()
            }

        return state

    def __setstate__(self, state):
        """Restore state and reinitialize the statemachine engine."""
        saved_state_id = state.pop("_saved_active_state_id", None)

        # 1. Restore transitions back to ndarrays
        if "transitions" in state and isinstance(state["transitions"], dict):
            state["transitions"] = {
                k: np.array(v) for k, v in state["transitions"].items()
            }

        # 2. Restore instance attributes
        self.__dict__.update(state)

        # 3. Rebuild the statemachine engine, callbacks, and registries
        super().__init__()

        # 4. Restore the random number generator
        self.rng = np.random.default_rng(self.seed)

        # 5. Restore active state if needed
        if saved_state_id and saved_state_id != "sleep":
            # If it was in an active task, transition or activate the corresponding state
            try:
                target_state = getattr(self.task, saved_state_id, None)
                if target_state:
                    self.current_state = target_state
            except Exception:
                pass


    def _p(self, src, dst):
        return self.transitions[src][self.TASK_NAMES.index(dst)]

    def analyze_pass(self, **kwargs):
        return self.rng.random() < self._p("analyze", "tx")   # 1 - a

    def rx_pass(self, **kwargs):
        return self.rng.random() < self._p("rx", "measure")   # 1 - b

    def _get_task_state_id(self):
        for state in self.configuration:
            if (
                state.is_active
                and hasattr(state, "value")
                and isinstance(state.value, Task)
                and state.id == 'sleep'
            ):
                id = state.id
                return id, np.array([[0]])
            elif (
                state.is_active
                and hasattr(state, "value")
                and isinstance(state.value, Task)
            ):
                id = state.id
                child_substates = np.argwhere(self.transitions[id])
                return id, child_substates
        return None, None

    def _get_current_state_id(self):
        for state in self.configuration:
            if state.is_active:
                return state.id
        return None

    def on_enter_measure(self, target, source=None, **kwargs):
        if source is None or source.id != "measure":
            # print(f"{time:.6f}:    Exiting {source.id} and entering {target.id}")
            self.task_start = self.time
            self.load_value = target.value.cost

    def on_enter_analyze(self, target, source=None, **kwargs):
        if source is None or source.id != "analyze":
            # print(f"{time:.6f}:    Exiting {source.id} and entering {target.id}")
            self.task_start = self.time
            self.load_value = target.value.cost

    def on_enter_tx(self, target, source=None, **kwargs):
        if source is None or source.id != "tx":
            # print(f"{time:.6f}:    Exiting {source.id} and entering {target.id}")
            self.task_start = self.time
            self.load_value = target.value.cost

    def on_enter_rx(self, target, source=None, **kwargs):
        if source is None or source.id != "rx":
            # print(f"{time:.6f}:    Exiting {source.id} and entering {target.id}")
            self.task_start = self.time
            self.load_value = target.value.cost

    def on_enter_sleep(self, target, source=None):
        if source is None or source.id != "sleep":
            # print(f"{time:.6f}:    Exiting {source.id} and entering {target.id}")
            self.task_start = self.time
            self.load_value = 0.0

    def on_enter_replan(self, target, source=None):
        if source is None or source.id != "replan":
            # print(f"{time:.6f}:    Exiting {source.id} and entering {target.id}")
            self.task_start = self.time
            self.load_value = 0.0

    def executing(self, source=None, **kwargs):
        if source and hasattr(source, "value") and isinstance(source.value, Task):
            return (self.time - self.task_start) < source.value.duration
        return False  # fallback default

    def charged(self):
        need = getattr(self, "wake_energy", None)
        if need is None:
            return self.cap.voltage >= self.cap.v_max
        return self.cap.energy - self.cap.min_energy >= need

    def recharging(self):
        return self.cap.voltage < self.cap.v_max

    def no_history(self):
        return self.history_values == {}

    def on_cycle(self, source=None, target=None, **kwargs):
        if source and hasattr(source, "value") and isinstance(source.value, Task):
            task = source.value
            energy = self.cap.energy
            min_energy = self.cap.min_energy
            self.remaining_time = task.duration - self.time + self.task_start
            projected_energy = energy - task.cost * self.remaining_time

            if projected_energy <= min_energy:
                self.raise_("pause")
            else:
                pass
        else:
            print(f"unexpected source.id: {source.id}")


def init_MATRSinkSM(cap, testing=False):
    sm = MATRSinkSM()
    sm.cap = cap
    sm.task_start = 0.0
    sm.time = 0.0
    return sm


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cap = Capacitor(20e-3)
    cap.voltage = 3.0
    cap.v_max = 3.3

    sm = init_MATRSinkSM(cap)

    # History for plotting
    times = [sm.time]
    voltages = [cap.voltage]
    loads = [sm.load_value]
    states = [sm._get_task_state_id() or sm._get_current_state_id()]
    task_times = [sm.time - sm.task_start]

    print(
        f"{'Step':>7} | {'State':>8} | {'Load(mW)':>8} | {'TaskTime(s)':>10} | {'CapVolt':>8} | {'Energy(uJ)':>10} | {'Action'}"
    )
    print("-" * 75)

    energy_uJ = cap.energy * 1e6
    load_mW = sm.load_value * 1e3
    state_id = sm._get_task_state_id() or sm._get_current_state_id()
    action = "cycle"

    print(
        f"{0:>8} | "
        f"{state_id:>8} | "
        f"{load_mW:8.3f} | "
        f"{sm.time - sm.task_start:11.3f} | "
        f"{cap.voltage:7.3f} | "
        f"{energy_uJ:10.2f} | "
        f"{action}"
    )

    for i in range(500):
        sm.time += 0.05
        sm.send("cycle")

        current_id = sm._get_current_state_id()

        if current_id == "sleep" and sm.charged():
            sm.send("wake")

        # Store simulation history
        times.append(sm.time)
        voltages.append(cap.voltage)
        loads.append(sm.load_value)
        states.append(sm._get_task_state_id() or sm._get_current_state_id())
        task_times.append(sm.time - sm.task_start)

        energy_uJ = cap.energy * 1e6
        load_mW = sm.load_value * 1e3
        state_id = sm._get_task_state_id() or sm._get_current_state_id()

        action = "cycle"
        if state_id == "sleep":
            action = "recharge" if sm.recharging() else "wake"

        print(
            f"{(i + 1) * 0.05:8.2f} | "
            f"{state_id:>8} | "
            f"{load_mW:8.3f} | "
            f"{sm.time - sm.task_start:11.2f} | "
            f"{cap.voltage:7.3f} | "
            f"{energy_uJ:10.2f} | "
            f"{action}"
        )

    print("-" * 75)
    print(
        f"Final: voltage={cap.voltage:.3f}V, "
        f"state={sm._get_task_state_id() or sm._get_current_state_id()}"
    )

    # ------------------------------------------------------------------
    # Plot simulation results
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # 1. Capacitor voltage
    axes[0].plot(times, voltages)
    axes[0].axhline(cap.v_min, linestyle="--", label=f"V_min = {cap.v_min:.2f} V")
    axes[0].axhline(cap.v_max, linestyle="--", label=f"V_max = {cap.v_max:.2f} V")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("Capacitor State")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Load power
    axes[1].plot(times, [load * 1e3 for load in loads])
    axes[1].set_ylabel("Load (mW)")
    axes[1].set_title("Sink Load")
    axes[1].grid(True)

    # 3. State
    state_names = ["sleep", "measure", "tx", "rx"]
    state_to_num = {state: i for i, state in enumerate(state_names)}

    state_nums = [state_to_num.get(state, -1) for state in states]

    axes[2].step(times, state_nums, where="post")
    axes[2].set_yticks(range(len(state_names)))
    axes[2].set_yticklabels(state_names)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("State")
    axes[2].set_title("State Machine")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show(block=False)
    input("Press enter to close figures...")
