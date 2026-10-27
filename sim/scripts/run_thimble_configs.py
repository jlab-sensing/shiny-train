#!/usr/bin/env python

"""Run LEACS simulation."""

import time

from sim.callback import LeacSimConfig
from sim.models import (
    SineSource,
    BonitoSource,
    CVCapacitor,
    SimulationRunner,
    SMSink,
    capacitor_list,
    simulation_config_matrix,
)
from sim.state_machines import MATRSinkSM

if __name__ == "__main__":
    # E6 catalog, millifarad decade: 1.0, 1.5, 2.2, 3.3, 4.7, 6.8 mF
    caps = capacitor_list(1e-3, 6.8e-3, decade="E6")

    configs = simulation_config_matrix(
        [CVCapacitor],
        caps,
        [BonitoSource, BonitoSource, BonitoSource, BonitoSource],
        [
            {"duration": 3600, "downsample": 10000, "name": "node0"},
            {"duration": 3600, "downsample": 10000, "name": "node1"},
            {"duration": 3600, "downsample": 10000, "name": "node2"},
            {"duration": 3600, "downsample": 10000, "name": "node3"},
        ],
        [SMSink],
        {"sink": MATRSinkSM()},
        [LeacSimConfig],
        {
            # --- circuit (CapacitorStorageSimConfig); src, caps, sink and name come from this matrix
            "p_lines": 2,                   # switched power lines (charge line + load line)
            "sw": None,                     # switch model, e.g. Switches.IdealSwitch(); None: the models.py default

            # --- design-time board and objective
            "cap_limit": 2,                 # M: most capacitors on the board
            "qp_weighting": "weighted",     # "unweighted" (tasks per cap) | "weighted" (energy per cap)

            # --- when to replan
            "replan_trigger": "depth",      # "depth" (every replan_depth transitions) | "forecast" (overdraw risk)
            "replan_depth": 2,              # depth trigger interval; -1: the depth trigger never fires
                                            #   (with replan_trigger="depth", -1 means no replanning: a static plan)
            "forecast_horizon": 1,          # h: transitions the forecast looks ahead
            "forecast_risk": 0.0,           # epsilon: replan when P(overdraw within h) > epsilon
            "forecast_gate": True,          # skip a replan no other board capacitor could help with

            # --- what a replan solves
            "visit_rule": "one",            # visit counts n_i: "one" | "ceil" (expected, rounded up) | "max" (worst case)
            "visit_depth": 3,               # D: transitions the visit counts cover (weighted objective only)
            "certificate": "mixed",         # "mixed" (near tasks vs stored energy, later vs full charge)
                                            #   | "current" (all vs stored) | "capacity" (all vs full charge)
            "freeze_board": True,           # replans keep the design-time board
            "cover_board": True,            # every board capacitor keeps at least one task
            "replan_fallback": "presolve",  # infeasible replan: "presolve" (revert to design-time map) | "keep"

            # --- runtime
            "wake_rule": "vmax",            # "vmax" (wake at V_max) | "energy" (wake_margin x one run, capped at full)
            "wake_margin": 1.1,             # energy wake target, in runs of the resumed task
            "harvest_routing": "fallback",  # "children" (next tasks' caps, else idle)
                                            #   | "fallback" (then the running cap, then another board cap)
            "charge_band": 0.05,            # V: while awake, a cap at V_max counts as full until this far below
        },
        # names=[f"unweighted/src{i}" for i in range(4)],
        names=[f"weighted_replan_depth_vmax/src{i}" for i in range(4)],
    )

    runner = SimulationRunner(configs)
    start = time.time()
    results = runner.run(progress_bar=True)
    print(f"runtime: {time.time() - start}")
    runner.save('/home/mwmaster/catch')
    runner.plot()
    runner.show()
