import unittest

from sim.models import (
    Capacitor,
    CapacitorStorageSim,
    CapacitorStorageSimConfig,
    ConstantSink,
    ConstantSource,
    IdealCapacitor,
    LeacsCapacitor,
    PanasonicCapacitors,
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


class TestCaps(unittest.TestCase):
    def test_generic(self):
        cap_values = [10e-6, 100e-6]

        src = ConstantSource(0.1, voltage=1.0, duration=2, dt=1)
        caps = [Capacitor(c) for c in cap_values]
        sink = ConstantSink(0.1)

        config = MyConfig(src, caps, sink, len(caps))

        sim = CapacitorStorageSim(config)
        sim.run()

        self.assertTrue(True)

    def test_ideal(self):
        cap_values = [10e-6, 100e-6]

        src = ConstantSource(0.1, voltage=1.0, duration=2, dt=1)
        caps = [IdealCapacitor(c) for c in cap_values]
        sink = ConstantSink(0.1)

        config = MyConfig(src, caps, sink, len(caps))

        sim = CapacitorStorageSim(config)
        sim.run()

        self.assertTrue(True)

    def test_missing_library(self):
        cap_values = [10e-6, 100e-6]

        src = ConstantSource(0.1, voltage=1.0, duration=2, dt=1)
        caps = [Capacitor(c, library="cap1.lib") for c in cap_values]
        sink = ConstantSink(0.1)

        config = MyConfig(src, caps, sink, len(caps))

        with self.assertRaises(FileNotFoundError):
            sim = CapacitorStorageSim(config)
            sim.run()

    def test_leacs(self):
        cap_values = [10e-6, 100e-6]

        src = ConstantSource(0.1, voltage=1.0, duration=2, dt=1)
        caps = [LeacsCapacitor(c) for c in cap_values]
        sink = ConstantSink(0.1)

        config = MyConfig(src, caps, sink, len(caps))

        sim = CapacitorStorageSim(config)
        sim.run()

        self.assertTrue(True)

    def test_panasonic(self):
        src = ConstantSource(0.1, voltage=1.0, duration=2, dt=1)
        caps = [PanasonicCapacitors.SPCAP.C100uF()]
        sink = ConstantSink(0.1)

        config = MyConfig(src, caps, sink, len(caps))

        sim = CapacitorStorageSim(config)
        sim.run()

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
