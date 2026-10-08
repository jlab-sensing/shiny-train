#!/usr/bin/env python

"""Example showing multiple simulation runs in parallel.

Deep copy with the `copy` module did not with the class inheritence
configuration. We recommend creating a function to return a specific
configuration based on parameters as shown here.
"""

from sim.models import (
    Capacitor,
    CapacitorStorageSimConfig,
    ConstantSink,
    ConstantSource,
    CVCapacitor,
    SimulationRunner,
)


class MyConfig(CapacitorStorageSimConfig):
    def callback(self, time: float):

        # connect caps
        for cap in self.caps:
            cap.connect(0)

        if time < 30:
            self.src.connect(0)
            self.sink.disconnect(0)
        else:
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


def create_config(caps: list[Capacitor]):
    src = ConstantSource(0.1, dt=0.1, duration=3600.0, voltage=3.3)
    sink = ConstantSink(0.00001)
    config = MyConfig(src, caps, sink, len(caps))
    return config


if __name__ == "__main__":
    cap_values = [10e-6, 47e-6, 100e-6]

    ideal_caps = [CVCapacitor(c) for c in cap_values]
    ideal_config = create_config(ideal_caps)

    cv_caps = [CVCapacitor(c) for c in cap_values]
    cv_config = create_config(cv_caps)

    #
    # Runner
    #

    configs = [
        ideal_config,
        cv_config,
    ]

    names = [
        "Ideal",
        "CV Capacitor",
    ]

    runner = SimulationRunner(configs, names)
    results = runner.run()
    runner.plot()
    runner.show()
