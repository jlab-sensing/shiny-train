#!/usr/bin/env python3

"""Compare the total leakage energy of various models of capacitors"""

from sim.models import (
    Capacitor,
    CapacitorStorageSimConfig,
    ConstantSink,
    ConstantSource,
    CVCapacitor,
    IdealCapacitor,
    SimulationRunner,
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


def create_config(caps: list[Capacitor]):
    src = ConstantSource(1, duration=3600, dt=1)
    sink = ConstantSink(0.1)
    config = TestCapacitor(src, caps, sink, 2)
    return config


if __name__ == "__main__":
    values = [
        100e-6,
        150e-6,
        220e-6,
        330e-6,
        470e-6,
        680e-6,
        1e-3,
    ]

    ideal_caps = [IdealCapacitor(c) for c in values]
    ideal_config = create_config(ideal_caps)

    cv_caps = [CVCapacitor(c) for c in values]
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
