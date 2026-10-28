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
    capacitor_list,
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


class TestCapacitorList(unittest.TestCase):
    def test_e6_single_decade(self):
        result = capacitor_list(1e-6, 6.8e-6)
        expected = [1e-6, 1.5e-6, 2.2e-6, 3.3e-6, 4.7e-6, 6.8e-6]
        self.assertEqual(result, expected)

    def test_default_decade_is_e6(self):
        self.assertEqual(
            capacitor_list(1e-6, 10e-6),
            capacitor_list(1e-6, 10e-6, decade="E6"),
        )

    def test_e3_single_decade(self):
        result = capacitor_list(1e-6, 4.7e-6, decade="E3")
        self.assertEqual(result, [1e-6, 2.2e-6, 4.7e-6])

    def test_e12_single_decade(self):
        result = capacitor_list(1e-6, 8.2e-6, decade="E12")
        expected = [
            1.0e-6,
            1.2e-6,
            1.5e-6,
            1.8e-6,
            2.2e-6,
            2.7e-6,
            3.3e-6,
            3.9e-6,
            4.7e-6,
            5.6e-6,
            6.8e-6,
            8.2e-6,
        ]
        self.assertEqual(result, expected)

    def test_bounds_are_inclusive(self):
        result = capacitor_list(1.5e-6, 4.7e-6)
        self.assertEqual(result, [1.5e-6, 2.2e-6, 3.3e-6, 4.7e-6])

    def test_bounds_between_values(self):
        # Neither bound is a series value
        result = capacitor_list(1.6e-6, 4.0e-6)
        self.assertEqual(result, [2.2e-6, 3.3e-6])

    def test_min_equals_max_on_a_value(self):
        result = capacitor_list(4.7e-6, 4.7e-6)
        self.assertEqual(result, [4.7e-6])

    def test_min_equals_max_not_on_a_value(self):
        self.assertEqual(capacitor_list(5e-6, 5e-6), [])

    def test_range_with_no_values(self):
        # E6 has 6.8e-6 and 10e-6, nothing strictly between 7e-6 and 9e-6
        self.assertEqual(capacitor_list(7e-6, 9e-6), [])

    def test_all_values_within_bounds(self):
        lo, hi = 2e-9, 3e-6
        result = capacitor_list(lo, hi)
        self.assertTrue(result)
        self.assertTrue(all(lo <= c <= hi for c in result))

    # ---- Multiple decades ----

    def test_multiple_decades(self):
        # 1e-6 .. 100e-6: two full E6 decades plus the 100e-6 endpoint
        result = capacitor_list(1e-6, 100e-6)
        expected = [
            1e-6,
            1.5e-6,
            2.2e-6,
            3.3e-6,
            4.7e-6,
            6.8e-6,
            10e-6,
            15e-6,
            22e-6,
            33e-6,
            47e-6,
            68e-6,
            100e-6,
        ]
        self.assertEqual(result, expected)

    def test_count_across_many_decades(self):
        # 1e-12 .. 1e-9 is three decades of 6 values plus the endpoint
        self.assertEqual(len(capacitor_list(1e-12, 1e-9, decade="E6")), 19)
        self.assertEqual(len(capacitor_list(1e-12, 1e-9, decade="E3")), 10)
        self.assertEqual(len(capacitor_list(1e-12, 1e-9, decade="E12")), 37)

    def test_large_values(self):
        result = capacitor_list(1.0, 10.0)
        self.assertEqual(result, [1.0, 1.5, 2.2, 3.3, 4.7, 6.8, 10.0])

    def test_invalid_decade_raises(self):
        with self.assertRaises(NotImplementedError):
            capacitor_list(1e-6, 10e-6, decade="E24")

    def test_zero_min_raises(self):
        with self.assertRaises(ValueError):
            capacitor_list(0, 10e-6)

    def test_negative_min_raises(self):
        with self.assertRaises(ValueError):
            capacitor_list(-1e-6, 10e-6)

    def test_zero_or_negative_max_raises(self):
        with self.assertRaises(ValueError):
            capacitor_list(1e-6, 0)
        with self.assertRaises(ValueError):
            capacitor_list(1e-6, -10e-6)

    def test_min_greater_than_max_raises(self):
        with self.assertRaises(ValueError):
            capacitor_list(10e-6, 1e-6)

    def test_invalid_decade_checked_before_range(self):
        with self.assertRaises(NotImplementedError):
            capacitor_list(-1, -2, decade="bogus")


if __name__ == "__main__":
    unittest.main()
