import io
import unittest
from contextlib import redirect_stderr

from sim.models import (
    Capacitor,
    CapacitorStorageSimConfig,
    ConstantSink,
    ConstantSource,
    CVCapacitor,
    IdealCapacitor,
    SimulationRunner,
)


class MyConfig(CapacitorStorageSimConfig):
    def callback(self, time: float):

        # connect caps
        for cap in self.caps:
            cap.connect(0)

        if time < 30:
            self.src.connect(0)
            self.sink.disconnect(0)
        else:
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


class TestRunner(unittest.TestCase):
    def test_runner(self):
        configs = create_runner_configs()
        runner = SimulationRunner(configs)

        with redirect_stderr(io.StringIO()):
            runner.run()

        runner.plot()

        self.assertTrue(True)

    def test_runner_no_progress_bar(self):
        configs = create_runner_configs()
        runner = SimulationRunner(configs)
        runner.run(progress_bar=False)
        runner.plot()

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
