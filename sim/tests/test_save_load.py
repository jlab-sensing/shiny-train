import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr

from sim.callback import (
    LeacSimConfig,
    MATRSinkSM,
)
from sim.models import (
    BonitoSource,
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
    SMSink,
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


class TestSaveLoad(unittest.TestCase):
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SimulationRunner(configs, results_dir=tmp_dir)

            with redirect_stderr(io.StringIO()):
                runner.run()

            runner.save()

            results = SimulationRunner.load(tmp_dir)

            plotter = SimulationPlotter()
            plotter.plot_multiple_ts(results)

        self.assertTrue(True)

    def test_runner_cache(self):
        """Test two consecutive runner."""

        configs = create_runner_configs()
        configs2 = create_runner_configs()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # run and save results
            runner = SimulationRunner(configs, results_dir=tmp_dir)
            with redirect_stderr(io.StringIO()):
                runner.run()
            runner.save()

            # run annother runner to check execution path
            runner2 = SimulationRunner(configs2, results_dir=tmp_dir)
            with redirect_stderr(io.StringIO()):
                runner2.run()

        self.assertTrue(True)

    def test_runner_results_exist(self):
        """Tests that the runner can load if results already exist."""

        configs = create_runner_configs()
        configs2 = create_runner_configs()

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SimulationRunner(configs, results_dir=tmp_dir)

            with redirect_stderr(io.StringIO()):
                runner.run()

            runner.save()

            runner2 = SimulationRunner(configs2, results_dir=tmp_dir)

            for cfg in configs2:
                res = runner2._check_results_exist(cfg)
                self.assertIsNotNone(res)

    def test_runner_result_not_exist(self):
        """Tests that the runner can load if results already exist."""

        configs = create_runner_configs()

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SimulationRunner(configs, results_dir=tmp_dir)

            for cfg in configs:
                res = runner._check_results_exist(cfg)
                self.assertIsNone(res)

    def test_results_save_leacs(self):
        """Test saving of a typical leacs config"""

        cap_values = [10e-3, 100e-3]

        src = BonitoSource(duration=10, downsample=10000)
        sink = SMSink(sink=MATRSinkSM())
        caps = [CVCapacitor(c) for c in cap_values]

        config = LeacSimConfig(
            # from original config class
            src=src,
            caps=caps,
            sink=sink,
            p_lines=2,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SimulationRunner([config], results_dir=tmp_dir)
            runner.run(progress_bar=False)

            runner.save()


if __name__ == "__main__":
    unittest.main()
