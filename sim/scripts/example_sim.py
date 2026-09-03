#!/usr/bin/env python

"""Runs an example simulation"""

from sim.models import (
    Capacitor,
    CapacitorStorageSim,
    CapacitorStorageSimConfig,
    ConstantSource,
    Sink,
)


class MyConfig(CapacitorStorageSimConfig):
    def callback(self, time: float):

        # print(f"Simulation time: {time}")

        # connect caps
        for cap in self.caps:
            cap.connect(0)

        if time < 0.5:
            self.src.connect(0)
            self.sink.disconnect(0)
        elif time >= 0.5:
            self.src.disconnect(0)
            self.sink.connect(0)

        # toggle load based on cap voltage
        # if self.caps[0].voltage > 0.5:
        #    print("discharge")
        #    self.src.disconnect(0)
        #    self.sink.connect(0)
        # elif self.caps[0].voltage < 0.2:
        #    print("charge")
        #    self.src.connect(0)
        #    self.sink.disconnect(0)


cap_values = [10e-6, 100e-6]

src = ConstantSource(1, 0.1, duration=2, dt=1)
caps = [Capacitor(c) for c in cap_values]
sink = Sink()

config = MyConfig(src, caps, sink, len(caps))

sim = CapacitorStorageSim(config)
sim.run()
print(sim.circuit)
sim.plot()
