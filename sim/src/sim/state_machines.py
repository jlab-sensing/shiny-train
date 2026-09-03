#!/usr/bin/env python3

from statemachine import State, StateChart, HistoryState
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


DC_VOLTS = 3.3


class Capacitor:
    def __init__(self, farads: float, v_min: float = 1.6, v_max: float = 3.3):
        self.farads = farads
        self.voltage = 0.0
        self.v_min = v_min
        self.v_max = v_max

    @property
    def energy(self) -> float:
        return self.voltage**2 * self.farads / 2

    @property
    def min_energy(self) -> float:
        return self.v_min**2 * self.farads / 2


@dataclass(frozen=True)
class Task:
    cost: float
    duration: float


class SinkSM(StateChart):
    cap = None
    time = None
    task_start = 0.0
    remaining_time = 0.0
    load_value = -4.64e-3 * DC_VOLTS  # OUTPUT: the energy consumption for next timestep

    class task(State.Compound):
        measure = State("measure", value=Task(cost=11.68e-3 * DC_VOLTS, duration=0.511))
        tx = State("tx", value=Task(cost=86.52e-3 * DC_VOLTS, duration=0.285))
        rx = State("rx", value=Task(cost=20.03e-3 * DC_VOLTS, duration=0.927))
        h = HistoryState(type="deep")

    sleep = State("sleep", initial=True, value=4.64e-3 * DC_VOLTS)

    # Self-transitions while executing, advance when done
    cycle = (
        task.measure.to.itself(cond="executing")
        | task.measure.to(task.tx)
        | task.tx.to.itself(cond="executing")
        | task.tx.to(task.rx)
        | task.rx.to.itself(cond="executing")
        | task.rx.to(task.measure)
    )

    pause = task.to(sleep)
    wake = sleep.to(task.measure, cond="no_history") | sleep.to(task.h)

    def _get_task_state_id(self):
        for state in self.configuration:
            if (
                state.is_active
                and hasattr(state, "value")
                and isinstance(state.value, Task)
            ):
                return state.id
        return None

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

    def executing(self, source=None, **kwargs):
        if source and hasattr(source, "value") and isinstance(source.value, Task):
            return (self.time - self.task_start) < source.value.duration
        return False  # fallback default

    def charged(self):
        return self.cap.voltage >= self.cap.v_max

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
            projected_energy = energy + task.cost * self.remaining_time

            if projected_energy <= min_energy:
                print(
                    f"{self.time:.6f}: going to sleep ({energy}, {projected_energy}, {min_energy})"
                )
                self.raise_("pause")
            else:
                # self.load_value = task.cost
                print(
                    f"{self.time:.6f}:     working ({energy}, {projected_energy}, {min_energy})"
                )
                pass
        else:
            print(f"unexpected source.id: {source.id}")


class TestSinkSM(SinkSM):
    def recharging(self):
        new_v = ((self.cap.energy + 5e-3 * 0.05) * 2 / self.cap.farads) ** 0.5

        self.cap.voltage = min(new_v, self.cap.v_max)
        return self.cap.voltage < self.cap.v_max

    def on_cycle(self, source=None, target=None, **kwargs):
        if source and hasattr(source, "value") and isinstance(source.value, Task):
            task = source.value
            energy = self.cap.energy
            min_energy = self.cap.min_energy
            self.remaining_time = task.duration - self.time + self.task_start
            projected_energy = energy + task.cost * self.remaining_time

            if projected_energy <= min_energy:
                self.raise_("pause")
            else:
                self.cap.voltage = (
                    (energy - task.cost * 0.05) * 2 / self.cap.farads
                ) ** 0.5


def init_SinkSM(cap, testing=True):
    if testing:
        sm = TestSinkSM()
    else:
        sm = SinkSM()
    sm.cap = cap
    sm.task_start = 0.0
    sm.time = 0.0
    return sm


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cap = Capacitor(20e-3)
    cap.voltage = 3.0
    cap.v_max = 3.3

    sm = init_SinkSM(cap)

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
