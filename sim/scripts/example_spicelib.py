#!/usr/bin/env python

import logging
import spicelib
from spicelib import RawRead, SpiceEditor
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from matplotlib import pyplot as plt
from spicelib import SimRunner



# set the logger to print to console and at info level
loglevel = logging.INFO
logger = logging.getLogger(__name__)
logging.basicConfig(level=loglevel)
spicelib.set_log_level(loglevel)



netlist_file = "example.net"
#raw_file = "ngsteps.raw"


# select spice model
runner = SimRunner(simulator=NGspiceSimulator)
netlist = SpiceEditor(netlist_file)

runner.run(netlist)


logger.info("Simulation done")


for raw_file, log_file in runner:
    logger.info(f"Raw file: {raw_file}, Log file: {log_file}")

    raw = RawRead(raw_file)

    logger.info(f"Trace names: {raw.get_trace_names()}")
    logger.info(f"Raw property: {raw.get_raw_property()}")

    time = raw.get_trace("time")
    v_in = raw.get_trace("v(source)")


    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(time.get_wave(), v_in.get_wave(), label="source")
    ax.grid()
    ax.legend()

    plt.tight_layout()
    plt.show()



## set default arguments
#netlist['R1'].value_str = '4k'
#netlist['V1'].model = "SINE(0 1 3k 0 0 0)"  # Modifying the behavior of the voltage source
#netlist.remove_instruction('.op')
#netlist.add_instruction(".tran 1n 3m")
#netlist.add_instruction(".plot V(out)")
#netlist.add_instruction(".save all")
#
## .step dec param cap 1p 10u 1
#for cap in sweep_log(1e-12, 10e-6, 10):
#    netlist['C1'].value = cap
#    runner.run(netlist, callback=processing_data)


logger.info("Done")
