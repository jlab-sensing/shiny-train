example_XXX.py are each demonstrations of the use of different subsystems of our simulation harness

run_thimble_configs.py and run_leacs_baseline.py run the two arms of the primary experiment, in isolation. [DEPRECATED: Use run_source.py]
run_source.py takes a data source file (expected to be in `shiny-train/sim/data`) and runs the entire primary experiment with resumable checkpointing

figures.py is AI slop, but it makes pretty pictures from data
plot.py is johns artisanal equivalent to figures.py
