#!/usr/bin/env python3

from statemachine import State, StateChart, HistoryState
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


DC_VOLTS = 3.7


class Capacitor:
    def __init__(self, farads: float, v_min: float = 1.6, v_max: float = 3.3):
        self.farads = farads
        self.voltage = 0.0
        self.v_min = v_min
        self.v_max = v_max

    @property
    def energy(self) -> float:
        return self.voltage ** 2 * self.farads / 2

    @property
    def min_energy(self) -> float:
        return self.v_min ** 2 * self.farads / 2


@dataclass(frozen=True)
class Task:
    cost: float
    duration: float


class SinkSM(StateChart):
    cap = None
    time = None
    task_start = 0.0
    load_value = 0.0     # OUTPUT: the energy consumption for next timestep

    class task(State.Compound):
        measure = State(
            "measure",
            initial=True,
            value=Task(
                cost=-11.68e-3 * DC_VOLTS,
                duration=0.511)
            )
        tx = State(
            "tx",
            value=Task(
                cost=-86.52e-3 * DC_VOLTS,
                duration=0.285)
            )
        rx = State(
            "rx",
            value=Task(
                cost=-20.03e-3 * DC_VOLTS,
                duration=0.927)
            )
        h = HistoryState(type="deep")

    sleep = State("sleep")

    # Self-transitions while executing, advance when done
    cycle = (
        task.measure.to.itself(cond="executing") |
        task.measure.to(task.tx) |
        task.tx.to.itself(cond="executing") |
        task.tx.to(task.rx) |
        task.rx.to.itself(cond="executing") |
        task.rx.to(task.measure)
    )

    pause = task.to(sleep)
    wake = sleep.to(task.h)

    def _get_task_state_id(self):
        for state in self.configuration:
            if state.is_active and hasattr(state, 'value') and isinstance(state.value, Task):
                return state.id
        return None

    def _get_current_state_id(self):
        for state in self.configuration:
            if state.is_active:
                return state.id
        return None

    def on_enter_measure(self, target, source=None, **kwargs):
        if source is None or source.id != "measure":
            self.task_start = self.time
            self.load_value = target.value.cost

    def on_enter_tx(self, target, source=None, **kwargs):
        if source is None or source.id != "tx":
            self.task_start = self.time
            self.load_value = target.value.cost

    def on_enter_rx(self, target, source=None, **kwargs):
        if source is None or source.id != "rx":
            self.task_start = self.time
            self.load_value = target.value.cost

    def on_enter_sleep(self, source=None):
        if source is None or source.id != "sleep":
            self.task_start = self.time
            self.load_value = 0.0

    def executing(self, source=None, **kwargs):
        if source and hasattr(source, 'value') and isinstance(source.value, Task):
            return (self.time - self.task_start) < source.value.duration
        return True   # fallback default

    def charged(self):
        return self.cap.voltage >= self.cap.v_max

    def recharging(self):
        # TEST FIXTURE: universal basic income
        self.cap.voltage = min(self.cap.voltage + 0.05, self.cap.v_max)
        return self.cap.voltage < self.cap.v_max

    def on_cycle(self, source=None, target=None, **kwargs):

        # TEST FIXTURE: universal basic income
        self.cap.voltage = min(self.cap.voltage + 0.05, self.cap.v_max)

        if source and hasattr(source, 'value') and isinstance(source.value, Task):
            task = source.value
            energy = self.cap.energy
            min_energy = self.cap.min_energy
            remaining_time = task.duration - self.time + self.task_start
            projected_energy = energy + task.cost * remaining_time
            # print(energy, projected_energy, min_energy)

            if projected_energy <= min_energy:
                self.raise_("pause")
            else:
                # TEST FIXTURE: simple discharge model
                tmp = energy + task.cost*0.05
                self.cap.voltage = (2 * tmp / cap.farads)**0.5
        elif source.id == 'sleep':
            # In sleep state - zzz
            pass
        else:
            print(f"unexpected source.id: {source.id}")


    def on_pause(self):
        # print("Going to sleep...")
        self.load_value = 0.0


def init_SinkSM(cap):
    sm = SinkSM()
    sm.cap = cap
    sm.task_start = 0.0
    sm.time = 0.0
    return sm


if __name__ == "__main__":
    cap = Capacitor(20e-3)
    cap.voltage = 3.

    sm = init_SinkSM(cap)
    print(sm.cap.voltage, sm.cap.farads, sm.time, sm.task_start, sm.load_value)

    print(f"{'Step':>7} | {'State':>8} | {'Load(mW)':>8} | {'TaskTime(s)':>10} | {'CapVolt':>8} | {'Energy(uJ)':>10} | {'Action'}")
    print("-" * 75)
    energy_uJ = cap.energy * 1e6
    load_mW = sm.load_value * 1e3
    state_id = sm._get_task_state_id() or sm._get_current_state_id()
    action = "cycle"
    print(f"{0:>8}| {state_id:>8} | {load_mW:8.3f} | {sm.time - sm.task_start:11.3f} | {cap.voltage:7.3f}  | {energy_uJ:10.2f} | {action}")

    for i in range(500):
        sm.time += 0.05
        sm.send("cycle")

        current_id = sm._get_current_state_id()
        if current_id == "sleep" and sm.charged():
            sm.send("wake")

        if (i+1) % 1 == 0:
            energy_uJ = cap.energy * 1e6
            load_mW = sm.load_value * 1e3
            state_id = sm._get_task_state_id() or sm._get_current_state_id()
            action = "cycle"
            if state_id == "sleep":
                action = "recharge" if sm.recharging() else "wake"
            print(f"{(i+1)*0.05:8.2f}| {state_id:>8} | {load_mW:8.3f} | {sm.time - sm.task_start:11.3f} | {cap.voltage:7.3f}  | {energy_uJ:10.2f} | {action}")

    print("-" * 75)
    print(f"Final: voltage={cap.voltage:.3f}V, state={sm._get_task_state_id() or sm._get_current_state_id()}")
