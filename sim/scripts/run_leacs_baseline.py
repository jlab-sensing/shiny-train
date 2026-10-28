#!/usr/bin/env python

"""LEACS baseline: rank every static plan on the training node, then run the
winner on the held-out nodes.

Faithful LEACS (Barjami et al., SenSys '26) picks its capacitors and its
task-to-capacitor map at compile time, by simulating every candidate plan and
keeping the best. It never replans, and it wakes at V_max.

This script covers one runtime cell (RUNTIME) at one workload (A, B). The
experiment harness (python -m harness.runner presolve) does every cell.
"""

import csv
import os
import time

import numpy as np

from sim.callback import LeacSimConfig, tasks_from_sm
from sim.models import (
    BonitoSource,
    CVCapacitor,
    Event,
    LeacsCapacitor,
    SimulationRunner,
    SMSink,
    capacitor_list,
)
from sim.state_machines import MATRSinkSM

# --- settings ---------------------------------------------------------------
CATALOG = capacitor_list(1e-3, 6.8e-3, decade="E6")   # 1.0, 1.5, 2.2, 3.3, 4.7, 6.8 mF
CAP_LIMIT = 2                                         # LEACS's M
A, B = 0.5, 0.5                                       # P(analyze -> measure), P(rx -> tx)
# RUNTIME = dict(wake_rule="vmax", harvest_routing="fallback")   # vmax is LEACS's own rule
RUNTIME = dict(wake_rule="energy", harvest_routing="fallback")   # vmax is LEACS's own rule
SOURCE = dict(downsample=10000)                       # add filename=... for another Bonito source
TRAIN_NODE, TEST_NODES = "node0", ["node1", "node2", "node3"]
# TRAIN_NODE, TEST_NODES = "node2", ["node0", "node1", "node3"]
TRAIN_SEEDS, TEST_SEEDS = [100, 101], [1, 2, 3]
TRAIN_DURATION, TEST_DURATION = 1200, 3600            # simulated seconds
# OUT = "/home/mwmaster/catch/thimble_sims/leacs_baseline_node0"
# OUT = "/home/mwmaster/catch/thimble_sims/leacs_baseline_node2"
OUT = "/home/mwmaster/catch/thimble_sims/leacs_energy_node0"
# OUT = "/home/mwmaster/catch/thimble_sims/leacs_energy_node2"


class LeacsPlanConfig(LeacSimConfig):
    """LeacSimConfig that runs one fixed static plan and never replans.

    `plan` gives each task's capacitor index, in task order (measure, analyze,
    tx, rx), e.g. "2222" = every task on the 2.2 mF capacitor. Each completed
    task is returned as an Event, so throughput survives the worker process.
    """

    def __init__(self, plan, **kwargs):
        super().__init__(replan_trigger="depth", replan_depth=-1, **kwargs)
        K, N = len(self.caps), len(plan)
        z = np.zeros(K * (N + 1))
        for i, j in enumerate(map(int, plan)):
            z[j] = z[K + i * K + j] = 1.0          # select capacitor j; assign task i to it
        self.set_plan(z)
        self.sm.cap = self.caps[int(plan[0])]      # the node starts with measure
        self._prev = None

    def callback(self, time):
        state = self.sm._get_task_state_id()[0]
        prev, self._prev = self._prev, state
        super().callback(time)
        if prev not in (None, "sleep") and state not in (prev, "sleep"):   # prev finished
            name = "report" if (prev, state) == ("rx", "measure") else f"done:{prev}"
            return Event(name=name, time=time)
        return None


def static_plans(catalog, max_caps):
    """Every plan LEACS may choose: each task fits a full charge (leakage at
    V_max), at most max_caps capacitors. Returns (plan, n_caps, leakage W)."""
    tasks = tasks_from_sm(MATRSinkSM)
    full = []
    for C in catalog:
        cv = CVCapacitor(C)
        c = LeacsCapacitor(C, v_min=cv.v_min, v_max=cv.v_max)
        c.voltage = c.v_max
        full.append(c)
    fits = [[(c.leakage * c.voltage + t.cost) * t.duration <= c.energy - c.min_energy for c in full]
            for t in tasks]
    plans = []
    for p in np.ndindex(*[len(catalog)] * len(tasks)):
        used = set(p)
        if len(used) <= max_caps and all(fits[i][j] for i, j in enumerate(p)):
            leak = sum(full[j].leakage * full[j].voltage for j in used)
            plans.append(("".join(map(str, p)), len(used), leak))
    return plans


def make_config(plan, node, seed, duration):
    return LeacsPlanConfig(
        plan=plan,
        src=BonitoSource(name=node, duration=duration, **SOURCE),
        caps=[CVCapacitor(C) for C in CATALOG],
        sink=SMSink(MATRSinkSM(a=A, b=B, seed=seed)),
        p_lines=2,
        name=f"leacs_{plan}/{node}/seed{seed}",
        cap_limit=CAP_LIMIT,
        **RUNTIME,
    )


def tasks_per_hour(results, duration):
    return [sum(e.name == "report" or e.name.startswith("done:") for e in r.events) * 3600 / duration
            for r in results]


def rank_plans():
    """Simulate every static plan on the training node with every training seed,
    in batches so the waveforms don't pile up in memory."""
    plans = static_plans(CATALOG, CAP_LIMIT)
    jobs = [(p, seed) for p, _, _ in plans for seed in TRAIN_SEEDS]
    print(f"ranking {len(plans)} plans x {len(TRAIN_SEEDS)} seeds on {TRAIN_NODE}")
    scores, batch = {}, os.cpu_count() or 4
    for k in range(0, len(jobs), batch):
        chunk = jobs[k:k + batch]
        results = SimulationRunner([make_config(p, TRAIN_NODE, s, TRAIN_DURATION) for p, s in chunk]).run(
            progress_bar=True)
        for (p, _), tph in zip(chunk, tasks_per_hour(results, TRAIN_DURATION)):
            scores.setdefault(p, []).append(tph)
        print(f"  {min(k + batch, len(jobs))}/{len(jobs)}")
    # highest mean throughput, then fewer capacitors, then less leakage
    ranked = sorted(plans, key=lambda r: (-np.mean(scores[r[0]]), r[1], r[2]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ranking.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["plan", "n_caps", "leak_W", "tasks_per_h"] + [f"seed{s}" for s in TRAIN_SEEDS])
        for p, n, leak in ranked:
            w.writerow([p, n, f"{leak:.3e}", f"{np.mean(scores[p]):.1f}"] + [f"{v:.1f}" for v in scores[p]])
    for p, n, _ in ranked[:5]:
        print(f"  {p}  {n} cap(s)  {np.mean(scores[p]):7.1f} tasks/h")
    return ranked[0][0]


def main():
    start = time.time()
    best = rank_plans()
    print(f"LEACS plan: {best}")

    configs = [make_config(best, node, seed, TEST_DURATION) for node in TEST_NODES for seed in TEST_SEEDS]
    runner = SimulationRunner(configs)
    results = runner.run(progress_bar=True)
    for cfg, tph in zip(configs, tasks_per_hour(results, TEST_DURATION)):
        print(f"  {cfg.name}: {tph:.1f} tasks/h")
    print(f"mean: {np.mean(tasks_per_hour(results, TEST_DURATION)):.1f} tasks/h; runtime {time.time() - start:.0f} s")
    runner.save(OUT)
    runner.plot()
    runner.show()


if __name__ == "__main__":
    main()
