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
import importlib.util
import os

from sim.models import (
    SimulationPlotter,
    SimulationResults,
    SimulationRunner,
)


def load_config_into_main(config_file: str) -> None:
    """Imports an external script and maps its classes into __main__ so pickle can resolve them."""
    abs_path = os.path.abspath(config_file)
    module_name = os.path.splitext(os.path.basename(abs_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {abs_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Expose the definitions into the current __main__ namespace
    import __main__

    for attr in dir(mod):
        if not attr.startswith("__"):
            setattr(__main__, attr, getattr(mod, attr))


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
    parser.add_argument(
        "--config-file",
        "-c",
        type=str,
        default=None,
        help="Path to Python script containing custom Config classes",
    )

    args = parser.parse_args()

    if args.config_file:
        load_config_into_main(args.config_file)


    if os.path.isfile(args.path):
        plot_single(args.path)
    elif os.path.isdir(args.path):
        plot_multiple(args.path)
    else:
        raise RuntimeError("Path is not a file or directory")


if __name__ == "__main__":
    entry()
