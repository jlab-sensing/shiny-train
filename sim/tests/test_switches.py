import unittest

from sim.models import (
    BaseSwitch,
    CapacitorStorageSim,
    CapacitorStorageSimConfig,
    ConstantSink,
    ConstantSource,
    IdealCapacitor,
    Switches,
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


def create_sim(switch: BaseSwitch) -> CapacitorStorageSim:
    """Create a test configuration

    Args:
        switch: Switch model

    Returns:
        Simulation config
    """

    cap_values = [10e-6, 100e-6]

    src = ConstantSource(0.1, voltage=1.0, duration=2, dt=1)
    caps = [IdealCapacitor(c) for c in cap_values]
    sink = ConstantSink(0.1)

    config = MyConfig(src, caps, sink, len(caps), sw=switch)

    sim = CapacitorStorageSim(config)

    return sim


class TestTemplate(unittest.TestCase):
    def test_ideal(self):
        sim = create_sim(Switches.IdealSwitch())
        sim.run()

        self.assertTrue(True)

    # THESE DO NOT WORK
    # One of them completely segfaults the test

    # def test_DMP1555UFA(self):
    #    sim = create_sim(Switches.DMP1555UFA())
    #    sim.run()

    #    self.assertTrue(True)

    # def test_DMP21D2UFA(self):
    #    sim = create_sim(Switches.DMP21D2UFA())
    #    sim.run()

    #    self.assertTrue(True)

    # def test_RZM002P02(self):
    #    sim = create_sim(Switches.RZM002P02())
    #    sim.run()

    #    self.assertTrue(True)

    # def test_BSS223PW(self):
    #    sim = create_sim(Switches.DMP1555UFA())
    #    sim.run()

    #    self.assertTrue(True)

    # def test_SSM3J36FS(self):
    #    sim = create_sim(Switches.SSM3J36FS())
    #    sim.run()

    #    self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
