import os
import unittest

from sim.models import BonitoSource, ConstantSource, SineSource


class TestSources(unittest.TestCase):
    def test_constant(self):
        """Check constant source."""

        src = ConstantSource(0.1, dt=0.1, duration=5.0, voltage=3.3)
        self.assertAlmostEqual(src.get_voltage(), 3.3)
        self.assertAlmostEqual(src.get_power(0.5), 0.1)

    def test_sine(self):
        """Check sine wave source."""

        # 60 Hz sine osciallating between 0.1 and 0.3
        src = SineSource(0.1, 0.2, 60.0, 0.0, dt=0.1, duration=3.0, voltage=2.5)

        self.assertAlmostEqual(src.get_voltage(), 2.5)

        # check power at two points
        self.assertAlmostEqual(src.get_power(0.0), 0.2)
        self.assertGreater(src.get_power(0.02214), 0.2)

    def test_bonito_basic(self):
        """Check reading of data for a few timestamps."""

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "..", "data", "bonito", "pwr_cars.h5")

        src = BonitoSource(data_path, voltage=1.65, downsample=1)

        self.assertAlmostEqual(src.get_voltage(), 1.65)
        self.assertAlmostEqual(src.get_power(0.0), 5.69625264568467e-06)
        src.next()
        self.assertAlmostEqual(src.get_power(1.0e-05), 7.096335328652324e-06)
        src.next()
        self.assertAlmostEqual(src.get_power(2.0e-05), 7.09716456e-06)
        src.next()
        self.assertAlmostEqual(src.get_power(3.0e-05), 6.89668724e-06)

    def test_bonito_name(self):
        """Check getting different nodes power data."""

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "..", "data", "bonito", "pwr_cars.h5")

        src = BonitoSource(data_path, name="node3", voltage=2.7, downsample=1)

        self.assertAlmostEqual(src.get_voltage(), 2.7)
        self.assertAlmostEqual(src.get_power(0.0), 3.652359120783698e-06)

    def test_bonito_downsample(self):
        """Check if downsample correctly gets the next datapoint."""

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "..", "data", "bonito", "pwr_cars.h5")

        # dataset is known to be 1e-5 dt, downsample to 0.1s
        src = BonitoSource(data_path, name="node3", voltage=2.7, downsample=10000)

        self.assertAlmostEqual(src.get_power(0.0), 3.652359120783698e-06)
        src.next()
        self.assertAlmostEqual(src.get_power(0.1), 1.4404980150767657e-05)

    def test_bonito_offset(self):
        """Check we can successfully set the offset."""

        # TODO (jmadden173): Check offset is working.
        self.assertTrue(True)

    def test_bonito_duration(self):
        """Check we can't set duration past the end of data."""

        # TODO (jmadden173): Check offset is working.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
