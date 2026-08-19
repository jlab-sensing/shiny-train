#!/usr/bin/env python3

"""Runs a test to check simulation validity.

The test chooses reasonable values for capacitors and charges them
individually. Then they are discharged individually across the load. After
a capacitor is charged on line 1 and discharged on line 2.

This test requires visual inspection of the generated plots.
"""

from sim.models import (
    Capacitor,
    CapacitorStorageSim,
    CapacitorStorageSimConfig,
    Sink,
    ConstantSource,
)

class MyConfig(CapacitorStorageSimConfig):
    def callback(self, time: float):

        # Test each of the switches
        if time < 1.0:
            self.reset()

        elif time < 1.5:
            # Connect power to cap 0
            self.src.connect(0)
            self.caps[0].connect(0)

        elif time < 2.0:
            # connect power to cap 1
            self.reset()
            self.src.connect(1)
            self.caps[1].connect(1)

        elif time < 2.5:
            # connect power to cap 2
            self.reset()
            self.src.connect(2)
            self.caps[2].connect(2)

        elif time < 3.0:
            # connect cap 0 to sink
            self.reset()
            self.sink.connect(0)
            self.caps[0].connect(0)

        elif time < 3.5:
            # connect cap 1 to sink
            self.reset()
            self.sink.connect(1)
            self.caps[1].connect(1)

        elif time < 4.0:
            # connect cap 2 to sink
            self.reset()
            self.sink.connect(2)
            self.caps[2].connect(2)

        # Check that we can correctly control lines
        elif time < 4.5:
            # connect src to cap 0 on line 1
            self.reset()
            self.src.connect(1)
            self.caps[0].connect(1)

        elif time < 5.0:
            # connect cap 0 to sink on line 2, while leaving src on line 1
            self.sink.connect(2)
            self.caps[0].disconnect(1)
            self.caps[0].connect(2)


# 10uF, 100uF, 1mF
cap_values = [10e-6, 100e-6, 1e-3]

src = ConstantSource(1, duration=10, dt=1)
caps = [Capacitor(c) for c in cap_values]
sink = Sink()

config = MyConfig(src, caps, sink, len(caps))

sim = CapacitorStorageSim(config)
sim.run()
sim.plot()
