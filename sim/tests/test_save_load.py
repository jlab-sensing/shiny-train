import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr

from sim.models import (
    Capacitor,
    CapacitorStorageSim,
    CapacitorStorageSimConfig,
    ConstantSink,
    ConstantSource,
    CVCapacitor,
    IdealCapacitor,
    SimulationPlotter,
    SimulationResults,
    SimulationRunner,
)


class MyConfig(CapacitorStorageSimConfig):
    def callback(self, time: float):
        # connect caps
        for cap in self.caps:
            cap.connect(0)

        if time < 0.5:
            self.src.connect(0)
            self.sink.disconnect(0)
        elif time >= 0.5:
            self.src.disconnect(0)
            self.sink.connect(0)


def create_config(caps: list[Capacitor]):
    src = ConstantSource(0.1, dt=0.1, duration=60.0, voltage=3.3)
    sink = ConstantSink(0.00001)
    config = MyConfig(src, caps, sink, len(caps))
    return config


def create_runner_configs() -> list[CapacitorStorageSimConfig]:
    cap_values = [10e-6, 47e-6, 100e-6]

    ideal_caps = [IdealCapacitor(c) for c in cap_values]
    ideal_config = create_config(ideal_caps)
    ideal_config.name = "Ideal"

    cv_caps = [CVCapacitor(c) for c in cap_values]
    cv_config = create_config(cv_caps)
    cv_config.name = "Leaky"

    #
    # Runner
    #

    configs = [
        ideal_config,
        cv_config,
    ]

    return configs



class TestTemplate(unittest.TestCase):
    def test_save_load(self):
        cap_values = [10e-6, 100e-6]

        src = ConstantSource(0.1, voltage=1.0, duration=2, dt=1)
        caps = [Capacitor(c) for c in cap_values]
        sink = ConstantSink(0.1)

        config = MyConfig(src, caps, sink, len(caps))

        sim = CapacitorStorageSim(config)
        save_res = sim.run()


        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "results.pkl")

            save_res.save(filepath)

            load_res = SimulationResults.load(filepath)

            plotter = SimulationPlotter()
            plotter.plot_single_ts(load_res)

        self.assertTrue(True)

    def test_runner_save_load(self):
        configs = create_runner_configs()
        runner = SimulationRunner(configs)

        with redirect_stderr(io.StringIO()):
            runner.run()

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner.save(tmp_dir)

            results = SimulationRunner.load(tmp_dir)

            plotter = SimulationPlotter()
            plotter.plot_multiple_ts(results)

        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
