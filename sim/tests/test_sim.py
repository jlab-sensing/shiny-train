"""Checks that the sim is functional."""

import unittest

from sim.models import (
    Capacitor,
    CapacitorStorageSim,
    CapacitorStorageSimConfig,
    Sink,
    ConstantSource,
    SineSource,
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


class TestSim(unittest.TestCase):
    def test_constant_source(self):
        cap_values = [10e-6, 100e-6]

        src = ConstantSource(1, duration=2, dt=1)
        caps = [Capacitor(c) for c in cap_values]
        sink = Sink()

        config = MyConfig(src, caps, sink, len(caps))

        sim = CapacitorStorageSim(config)
        sim.run()

        self.assertTrue(True)

    def test_sine_source(self):
        cap_values = [10e-6, 100e-6]

        src = SineSource(1, 0.5, 10, 0, duration=2, dt=1)
        caps = [Capacitor(c) for c in cap_values]
        sink = Sink()

        config = MyConfig(src, caps, sink, len(caps))

        sim = CapacitorStorageSim(config)
        sim.run()

        self.assertTrue(True)

    def test_many_caps(self):
        # Initially tried with 100 but took 60s to complete
        cap_values = [100e-6 for _ in range(10)]

        src = ConstantSource(1, duration=2, dt=1)
        caps = [Capacitor(c) for c in cap_values]
        sink = Sink()

        config = MyConfig(src, caps, sink, 2)

        sim = CapacitorStorageSim(config)
        sim.run()

        self.assertTrue(True)

    def test_many_lines(self):
        cap_values = [10e-6, 100e-6]

        src = ConstantSource(1, duration=2, dt=1)
        caps = [Capacitor(c) for c in cap_values]
        sink = Sink()

        config = MyConfig(src, caps, sink, 10)

        sim = CapacitorStorageSim(config)
        sim.run()

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
