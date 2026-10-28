#!/usr/bin/env python3

"""Plots svaed results.

Path to results can either be a single saved pickle or a directory to plot
multiple results.

To test this script you can run `./example_sim.py` which generates `pickle.pkl`
that can be used with this script.

Usage:
    ./plot_results.py results.pkl
    ./plot_results.py results/
"""

import argparse
import os

from sim.models import (
    SimulationPlotter,
    SimulationResults,
    SimulationRunner,
)


def plot_single(filepath: str):
    """Plot single result"""

    res = SimulationResults.load(filepath)

    plotter = SimulationPlotter()
    plotter.plot([res])
    plotter.show()


def plot_multiple(directory: str):
    res = SimulationRunner.load(directory)

    plotter = SimulationPlotter()
    plotter.plot(res)
    plotter.show()


def entry():
    """Entrypoint to the script.

    Sets the arguments and calls the appropriate plotting function.
    """

    parser = argparse.ArgumentParser(description="Process simulation results.")
    parser.add_argument("path", type=str, help="Path to the results directory or file")
    args = parser.parse_args()

    if os.path.isfile(args.path):
        plot_single(args.path)
    elif os.path.isdir(args.path):
        plot_multiple(args.path)
    else:
        raise RuntimeError("Path is not a file or directory")


if __name__ == "__main__":
    entry()
