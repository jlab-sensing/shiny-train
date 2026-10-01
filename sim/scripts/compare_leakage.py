#!/usr/bin/env python3

"""Compare the total leakage energy of various models of capacitors


"""

from sim.models import (
    CapacitorStorageSimConfig,
    ConstantSource,
    IdealCapacitor,
    LeacsCapacitor,
    VeryLeakyCapacitor,
    PanasonicCapacitors,
    ConstantSink,
    CapacitorStorageSim
)


class TestCapacitor(CapacitorStorageSimConfig):

    # flag for sufficient capacitor charge
    charged = False

    def callback(self, time: float):
        if not self.charged:
            self.src.connect(0)

            # check that all capacitors are fully charged
            self.charged = True
            for cap in self.caps:
                cap.connect(0)

                if cap.voltage < 3.3:
                    self.charged = False
        else:
            # disconnect everything after charged
            self.src.reset()
            self.sink.reset()
            for cap in self.caps:
                cap.reset()


if __name__ == "__main__":
    src = ConstantSource(1, duration=3600, dt=1)

    # list of capacitors to evaluate, first one is baseline
    caps = [
        VeryLeakyCapacitor(100e-6),
        IdealCapacitor(100e-6),
        LeacsCapacitor(100e-6),
        PanasonicCapacitors.SPCAP.C100uF(),
        PanasonicCapacitors.SPCAP.C470uF(),
        PanasonicCapacitors.POSCAP.C680uF(),
    ]

    sink = ConstantSink(0.1)

    config = TestCapacitor(src, caps, sink, 2)

    sim = CapacitorStorageSim(config)

    sim.run()
    print(sim.circuit)
    sim.plot()
