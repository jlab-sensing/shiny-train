#!/usr/bin/env python

"""Example showing multiple simulation runs in parallel.

Deep copy with the `copy` module did not with the class inheritance
configuration. We recommend creating a function to return a specific
configuration based on parameters as shown here.
"""

from sim.models import (
    BonitoSource,
    Capacitor,
    CapacitorStorageSimConfig,
    ConstantSink,
    CVCapacitor,
    Event,
    IdealCapacitor,
    SimulationRunner,
)


class MyConfig(CapacitorStorageSimConfig):
    next_50 = 500.0
    next_70 = 700.0
    next_90 = 900.0

    def callback(self, time: float):

        event = None

        # connect caps
        for cap in self.caps:
            cap.connect(0)

        if time < 30:
            self.src.connect(0)
            self.sink.disconnect(0)
        else:
            self.src.disconnect(0)
            self.sink.connect(0)

        if time >= self.next_50:
            event = Event("50s", time)
            self.next_50 += 500.0

        if time >= self.next_70:
            event = Event("70s", time)
            self.next_70 += 700.0

        if time >= self.next_90:
            event = Event("90s", time, ignore=True)
            self.next_90 += 900.0

        return event


def create_config(caps: list[Capacitor]):
    src = BonitoSource(duration=3600.0, downsample=10000, voltage=3.3)
    sink = ConstantSink(0.00001)
    config = MyConfig(src, caps, sink, len(caps))
    return config


if __name__ == "__main__":
    cap_values = [10e-6, 47e-6, 100e-6]

    ideal_caps = [IdealCapacitor(c) for c in cap_values]
    ideal_config = create_config(ideal_caps)
    ideal_config.name = "Ideal"

    cv_caps = [CVCapacitor(c) for c in cap_values]
    cv_config = create_config(cv_caps)
    cv_config.name = "Leaky"

    #
    # Runner
    #

    configs = [
        ideal_config,
        cv_config,
    ]

    runner = SimulationRunner(configs)
    results = runner.run()
    runner.plot()
    runner.show()
