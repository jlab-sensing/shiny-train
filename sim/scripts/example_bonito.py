#!/usr/bin/env python

"""Runs an example simulation"""

import os

from sim.models import (
    BonitoSource,
    Capacitor,
    CapacitorStorageSim,
    CapacitorStorageSimConfig,
    Sink,
)


class MyConfig(CapacitorStorageSimConfig):
    def callback(self, time: float):

        print(f"Simulation time: {time}")

        # connect caps
        for cap in self.caps:
            cap.connect(0)

        if time < 20:
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


script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "bonito", "pwr_cars.h5")
src = BonitoSource(
    data_path, name="node0", voltage=3.3, downsample=10000, duration=30.0
)

caps = [Capacitor(c) for c in cap_values]
sink = Sink()

config = MyConfig(src, caps, sink, len(caps))

sim = CapacitorStorageSim(config)
sim.run()
print(sim.circuit)
sim.plot()
