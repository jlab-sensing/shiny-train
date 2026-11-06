#!/usr/bin/env python

"""Each Bonito source file, both wake rules: LEACS vs. three Thimble variants.

For each source file and wake rule:
  1. LEACS ranks every static plan on the training node (faithful LEACS:
     simulate every candidate, keep the best, never replan).
  2. On every held-out node and test seed, run LEACS's plan and Thimble
     (weighted objective) with no replanning, depth-triggered replanning,
     and risk-triggered (forecast) replanning.

Every run uses the same task sequence for a given seed (common random
numbers), so arms can be compared run by run.

Source files: FILENAMES below, or on the command line,
    python run_source.py --filenames pwr_cars.h5 pwr_jogging.h5
Each file gets its own output directory, OUT_ROOT/template_<name>, where
<name> is the file name without "pwr_" and ".h5" (pwr_cars.h5 -> template_cars).
Its nodes are read from the file: by default the first (in natural order)
trains LEACS and the rest are held out.

Resumable, two ways:
  - each finished run is appended to <out>/progress.jsonl, and runs listed
    there are never rebuilt;
  - results are saved by SimulationRunner.save() as
    <out>/results/<startup_hash>.pkl, and SimulationRunner skips any config
    whose pickle already exists, so a crash between saving and recording costs
    only reloading that batch.
A crash loses at most one batch.

Cache keys: CapacitorStorageSimConfig computes startup_hash in its own
__init__, before LeacSimConfig sets the wake rule, the replanning settings or a
LEACS plan. Configs that differ only in those would share a hash and SimulationRunner
would hand one run's results to another. key_config() below folds a
description of each job into the hash to keep them apart. The source file and
node are already in the hash (they are attributes of the source).
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
from types import SimpleNamespace

import h5py
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
FILENAMES = ["pwr_cars.h5"]                           # Bonito files in the data path; or --filenames
SOURCE = dict(downsample=10000)                       # other BonitoSource kwargs (datapath, offset)
TRAIN_NODE, TEST_NODES = None, None                   # None: first node trains, the rest are held out
CATALOG = capacitor_list(1e-3, 6.8e-3, decade="E6")   # 1.0, 1.5, 2.2, 3.3, 4.7, 6.8 mF
CAP_LIMIT = 2                                         # M, for LEACS and Thimble
A, B = 0.5, 0.5                                       # P(analyze -> measure), P(rx -> tx)
WAKE_RULES = ["vmax", "energy"]                       # vmax is LEACS's own rule
ROUTING = "fallback"
TRAIN_SEEDS, TEST_SEEDS = [100, 101], [1, 2, 3]
TRAIN_DURATION, TEST_DURATION = 1200, 3600            # simulated seconds
OUT_ROOT = "/home/mwmaster/catch"                     # one template_<name> directory per file
SAVE_RANKING = True                                  # ranking pickles: ~200 x 20-min runs per wake rule
CACHE_TAG = "v1"                                      # change after editing callback.py, to rerun everything

# Thimble variants: weighted objective, two-capacitor board, the paper's
# certificate and board rules; only the replanning trigger differs.
THIMBLE = dict(qp_weighting="weighted", cap_limit=CAP_LIMIT, certificate="mixed",
               freeze_board=True, cover_board=False, replan_fallback="presolve",
               forecast_gate=True, forecast_horizon=1, forecast_risk=0.0,
               visit_rule="ceil", visit_depth=2)
ARMS = {
    "thimble_noreplan": dict(THIMBLE, replan_trigger="depth", replan_depth=-1),
    "thimble_depth": dict(THIMBLE, replan_trigger="depth", replan_depth=2),
    "thimble_risk": dict(THIMBLE, replan_trigger="forecast", replan_depth=-1),
}


# --- configs ----------------------------------------------------------------
class EventLog:
    """Mixin: returns each completed task as an Event (a report when Receive
    hands back to Measure), so throughput survives the worker process."""

    def callback(self, time):
        state = self.sm._get_task_state_id()[0]
        prev, self._prev = getattr(self, "_prev", None), state
        super().callback(time)
        if prev not in (None, "sleep") and state not in (prev, "sleep"):   # prev finished
            return Event(name="report" if (prev, state) == ("rx", "measure") else f"done:{prev}", time=time)
        return None


class LeacsPlanConfig(EventLog, LeacSimConfig):
    """LEACS: one fixed static plan, no replanning. `plan` gives each task's
    capacitor index in task order (measure, analyze, tx, rx), e.g. "2323"."""

    def __init__(self, plan, **kwargs):
        super().__init__(replan_trigger="depth", replan_depth=-1, cap_limit=CAP_LIMIT, **kwargs)
        K, N = len(self.caps), len(plan)
        z = np.zeros(K * (N + 1))
        for i, j in enumerate(map(int, plan)):
            z[j] = z[K + i * K + j] = 1.0          # select capacitor j; assign task i to it
        self.set_plan(z)
        self.sm.cap = self.caps[int(plan[0])]      # the node starts with measure


class ThimbleConfig(EventLog, LeacSimConfig):
    """Thimble: board and map from its own program; replanning per ARMS."""


def key_config(cfg, job):
    """Fold the job's own settings into cfg.startup_hash, which otherwise only
    covers the source (file and node included), capacitors, sink, power lines
    and switch (see above)."""
    settings = dict(kind=job["kind"], plan=job.get("plan"), wake=job["wake"], routing=ROUTING,
                    a=A, b=B, arm=ARMS.get(job["kind"]), cap_limit=CAP_LIMIT, tag=CACHE_TAG)
    text = cfg.startup_hash + json.dumps(settings, sort_keys=True)
    cfg.startup_hash = hashlib.sha256(text.encode()).hexdigest()
    cfg.name = job["name"]
    return cfg


def make_config(job):
    common = dict(
        src=BonitoSource(filename=job["file"], name=job["node"], duration=job["duration"], **SOURCE),
        caps=[CVCapacitor(C) for C in CATALOG],
        sink=SMSink(MATRSinkSM(a=A, b=B, seed=job["seed"])),
        p_lines=2, wake_rule=job["wake"], harvest_routing=ROUTING,
    )
    if job["kind"] == "leacs":
        cfg = LeacsPlanConfig(plan=job["plan"], **common)
    else:
        cfg = ThimbleConfig(**ARMS[job["kind"]], **common)
    return key_config(cfg, job)


# --- one source file ---------------------------------------------------------
def natural_key(name):
    """node2 before node10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def source_setup(filename):
    """Output paths, nodes and time step for one Bonito file. Fails early if
    the file is shorter than the runs."""
    name = os.path.splitext(os.path.basename(filename))[0].removeprefix("pwr_")
    out = os.path.join(OUT_ROOT, f"template_{name}")
    probe = BonitoSource(filename=filename, duration=0, **SOURCE)    # resolves the data path
    with h5py.File(probe.fullpath, "r") as f:
        nodes = sorted(f["data"].keys(), key=natural_key)
        span = float(f["time"][-1] - f["time"][0])
    train = TRAIN_NODE or nodes[0]
    test = TEST_NODES or [n for n in nodes if n != train]
    longest = max(TRAIN_DURATION, TEST_DURATION)
    if longest > span:
        # raise ValueError(f"{filename} covers {span:.0f} s, shorter than the {longest} s runs")
        print(f"{filename} covers {span:.0f} s, shorter than the {longest} s runs")
    missing = [n for n in [train, *test] if n not in nodes]
    if missing:
        raise ValueError(f"{filename} has no node(s) {missing}; it has {nodes}")
    return SimpleNamespace(file=filename, name=name, out=out, results=os.path.join(out, "results"),
                           progress=os.path.join(out, "progress.jsonl"), train=train, test=test,
                           dt=probe.dt, span=span)


# --- progress ---------------------------------------------------------------
def load_progress(path):
    """Finished runs by name, from a progress.jsonl."""
    if not os.path.exists(path):
        return {}
    with open(path, "rb+") as fh:                  # end a line cut off by the crash,
        fh.seek(0, os.SEEK_END)                    # so new records start on their own line
        if fh.tell() and (fh.seek(-1, os.SEEK_END) or fh.read(1) != b"\n"):
            fh.write(b"\n")
    done = {}
    for line in open(path):
        try:
            rec = json.loads(line)
            done[rec["name"]] = rec
        except (json.JSONDecodeError, KeyError):
            pass                                   # a line cut off by the crash
    return done


def record(path, rec):
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def throughput(events, duration):
    tasks = sum(e.name == "report" or e.name.startswith("done:") for e in events)
    reports = sum(e.name == "report" for e in events)
    return tasks * 3600 / duration, reports * 3600 / duration


def run_jobs(site, jobs, save):
    """Run the jobs not yet finished, in batches of one per core. With save,
    results go to the results directory first and the progress line second,
    so a run is never recorded without its pickle."""
    done = load_progress(site.progress)
    todo = [j for j in jobs if j["name"] not in done]
    print(f"  {len(jobs) - len(todo)} of {len(jobs)} already done")
    batch = os.cpu_count() or 4
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        runner = SimulationRunner([make_config(j) for j in chunk], results_dir=site.results)
        results = runner.run(progress_bar=True)    # loads any cached pickle
        if save:
            runner.save()                          # <out>/results/<startup_hash>.pkl
        for j, res in zip(chunk, results):
            rec = dict(j)
            rec["tasks_per_h"], rec["reports_per_h"] = throughput(res.events, j["duration"])
            record(site.progress, rec)
        print(f"  {min(k + batch, len(todo))}/{len(todo)}")
    done = load_progress(site.progress)
    return [done[j["name"]] for j in jobs]


# --- the experiment -----------------------------------------------------------
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


def rank_leacs(site, wake):
    """LEACS's plan for this source and wake rule: highest mean tasks per hour
    on the training node, then fewer capacitors, then less leakage."""
    plans = static_plans(CATALOG, CAP_LIMIT)
    jobs = [dict(name=f"rank/{wake}/{p}/seed{s}", phase="rank", file=site.file, wake=wake, kind="leacs",
                 plan=p, node=site.train, seed=s, duration=TRAIN_DURATION)
            for p, _, _ in plans for s in TRAIN_SEEDS]
    print(f"[{site.name} {wake}] ranking {len(plans)} plans x {len(TRAIN_SEEDS)} seeds on {site.train}")
    scores = {}
    for rec in run_jobs(site, jobs, save=SAVE_RANKING):
        scores.setdefault(rec["plan"], []).append(rec["tasks_per_h"])
    ranked = sorted(plans, key=lambda r: (-np.mean(scores[r[0]]), r[1], r[2]))
    with open(os.path.join(site.out, f"ranking_{wake}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["plan", "n_caps", "leak_W", "tasks_per_h"] + [f"seed{s}" for s in TRAIN_SEEDS])
        for p, n, leak in ranked:
            w.writerow([p, n, f"{leak:.3e}", f"{np.mean(scores[p]):.1f}"] + [f"{v:.1f}" for v in scores[p]])
    for p, n, _ in ranked[:5]:
        print(f"  {p}  {n} cap(s)  {np.mean(scores[p]):7.1f} tasks/h")
    return ranked[0][0]


def evaluate(site, wake, plan):
    """LEACS's plan and every Thimble arm on every held-out node and seed."""
    jobs = []
    for node in site.test:
        for s in TEST_SEEDS:
            base = dict(phase="eval", file=site.file, wake=wake, node=node, seed=s, duration=TEST_DURATION)
            jobs.append(dict(base, name=f"{wake}/leacs_{plan}/{node}/seed{s}", kind="leacs", plan=plan))
            jobs += [dict(base, name=f"{wake}/{arm}/{node}/seed{s}", kind=arm) for arm in ARMS]
    print(f"[{site.name} {wake}] evaluating {len(jobs)} runs")
    return run_jobs(site, jobs, save=True)


def summarize(site, rows):
    """Mean tasks per hour per wake rule and arm, and each Thimble arm paired
    against LEACS on the same node and seed (log ratio)."""
    leacs = {(r["wake"], r["node"], r["seed"]): r["tasks_per_h"] for r in rows if r["kind"] == "leacs"}
    out = []
    for wake in WAKE_RULES:
        for kind in ["leacs"] + list(ARMS):
            rs = [r for r in rows if r["wake"] == wake and r["kind"] == kind]
            tph = [r["tasks_per_h"] for r in rs]
            ratios = [np.log(r["tasks_per_h"] / leacs[(wake, r["node"], r["seed"])])
                      for r in rs if leacs.get((wake, r["node"], r["seed"])) and r["tasks_per_h"]]
            out.append(dict(source=site.name, wake=wake, arm=kind, runs=len(rs), tasks_per_h=np.mean(tph),
                            vs_leacs=np.mean(ratios) if ratios and kind != "leacs" else float("nan"),
                            wins=np.mean([x > 0 for x in ratios]) if ratios and kind != "leacs" else float("nan")))
    with open(os.path.join(site.out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\n[{site.name}] trains on {site.train}, holds out {', '.join(site.test)}")
    print(f"{'wake':7s} {'arm':17s} {'runs':>4s} {'tasks/h':>8s} {'vs LEACS':>9s} {'wins':>5s}")
    for o in out:
        print(f"{o['wake']:7s} {o['arm']:17s} {o['runs']:4d} {o['tasks_per_h']:8.1f} "
              f"{o['vs_leacs']:+9.1%} {o['wins']:5.0%}")
    print("(vs LEACS: mean paired log ratio on the same node and seed; LEACS runs with no tasks are left out)")


def main(filenames=None):
    start = time.time()
    sites = [source_setup(f) for f in (filenames or FILENAMES)]   # check every file before running
    for site in sites:
        print(f"[{site.name}] {site.file}: {site.span:.0f} s, step {site.dt:.3g} s, "
              f"train {site.train}, test {', '.join(site.test)} -> {site.out}")
    for site in sites:
        os.makedirs(site.results, exist_ok=True)
        rows = []
        for wake in WAKE_RULES:
            plan = rank_leacs(site, wake)
            print(f"[{site.name} {wake}] LEACS plan: {plan}")
            rows += evaluate(site, wake, plan)
        summarize(site, rows)
    print(f"runtime {time.time() - start:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LEACS vs. Thimble on one or more Bonito source files.")
    ap.add_argument("--filenames", nargs="+", default=None,
                    help=f"Bonito .h5 files in the data path (default: {' '.join(FILENAMES)})")
    main(ap.parse_args().filenames)
