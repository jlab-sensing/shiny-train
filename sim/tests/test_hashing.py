import unittest

from sim.callback import (
    LeacSimConfig,
    MATRSinkSM,
)
from sim.models import (
    BonitoSource,
    CapacitorStorageSimConfig,
    ConstantSink,
    ConstantSource,
    CVCapacitor,
    SMSink,
)


class TestCapacitorStorageSimConfigHash(unittest.TestCase):
    def create_config(self, name="TestConfig", duration=10):
        """Helper method to instantiate the config easily."""

        cap_values = [1e-3, 10e-3]

        cfg = CapacitorStorageSimConfig(
            src=ConstantSource(0.1, dt=1, duration=duration),
            caps=[CVCapacitor(v) for v in cap_values],
            sink=ConstantSink(0.1),
            p_lines=2,
            name=name,
        )

        return cfg

    def create_config_complex(self, name="TestComplex"):
        """Helper function to create leacs/matr/bonito config."""

        cap_values = [1e-3, 10e-3, 100e-3]

        cfg = LeacSimConfig(
            src=BonitoSource(duration=60, downsample=10000),
            caps=[CVCapacitor(v) for v in cap_values],
            sink=SMSink(sink=MATRSinkSM()),
            p_lines=2,
            name=name,
        )

        return cfg

    def test_hash_returns_string(self):
        """Test that _hash returns a valid hex digest string (SHA-256)."""

        config = self.create_config()

        result_hash = config._hash()

        self.assertIsInstance(result_hash, str)
        # SHA-256 hex digest is always 64 characters long
        self.assertEqual(len(result_hash), 64)

    def test_hash_consistency(self):
        """Test that identical configurations produce the exact same hash."""
        config1 = self.create_config()
        config2 = self.create_config()

        self.assertEqual(config1._hash(), config2._hash())

    def test_hash_exclusion(self):
        """Test that modifying an excluded attribute does NOT change the hash."""

        config1 = self.create_config(duration=10)
        config2 = self.create_config(duration=20)

        hash1 = config1._hash(excluded_attr=["src"])
        hash2 = config2._hash(excluded_attr=["src"])
        self.assertEqual(hash1, hash2)

    def test_hash_inclusion_change(self):
        """Test that modifying a non-excluded attribute DOES change the hash."""

        config1 = self.create_config(duration=10)
        config2 = self.create_config(duration=20)

        self.assertNotEqual(config1._hash(), config2._hash())

    def test_hash_consistency_complex(self):
        """Test that identical configurations produce the exact same hash."""
        config1 = self.create_config_complex()
        config2 = self.create_config_complex()

        self.assertEqual(config1._hash(), config2._hash())

    def test_hash_exclusion_complex(self):
        """Test that modifying an excluded attribute does NOT change the hash."""

        config1 = self.create_config_complex()
        config2 = self.create_config_complex()
        config2.replan_depth = 10

        hash1 = config1._hash(excluded_attr=["replan_depth"])
        hash2 = config2._hash(excluded_attr=["replan_depth"])
        self.assertEqual(hash1, hash2)

    def test_hash_inclusion_change_complex(self):
        """Test that modifying a non-excluded attribute DOES change the hash."""

        config1 = self.create_config(name="config")
        config2 = self.create_config(name="config")
        config2.replan_depth = 10

        self.assertNotEqual(config1._hash(), config2._hash())


if __name__ == "__main__":
    unittest.main()
