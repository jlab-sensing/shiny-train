# Simulator

## Install

```
pip3 install .
```

## Linting and formatting

[Ruff](https://docs.astral.sh/ruff/) is used for formatting and linting. Install development dependencies and run `ruff`.

```
pip3 install -e .[dev]
ruff format
ruff check
```

## Unittests

The builtin module [unittest](https://docs.python.org/3/library/unittest.html) is used for writing tests. See [this](https://docs.python.org/3/library/unittest.html#assert-methods) page for list of assert methods. Use the following to run the tests.

```
python -m unittest
```

### Running 

## FAQ

### `spicelib` vs `PySpice`?

`spicelib` is better maintained but hooks into a SPICE executable. `PySpice` hooks into underlying C libraries to give SPICE functionality in python. In order to change state while the simulation is running we need `PySpice`.

### Cannot find libraries in netlist

You may see an error something like this.

```
INFO:spicelib.SimRunner:SimRunner initialized
ERROR:spicelib.SpiceEditor:Could not find library '"/home/jtmadden/repos/jlab/shiny-train/cap.lib"'
INFO:__main__:Done
```

**Manually remove the quotes from the .net file.** `spicelib` "autoescapes" the path.

### Regex error on sheet labels

The following error happens when KiCAD *sheet* labels are used instead of *global* labels. Sheet labels are named like `/source` while global label would just be `source`. The `spicelib` regex is not setup to handle sheet labels.

```
spicelib.editor.editor_errors.UnrecognizedSyntaxError: Line: "S1 /source /C1 Net-_S1-C+_ GND __S1
" doesn't match regular expression "^(?P<designator>S§?\w+)(?P<nodes>(?:\s+[\w+-\.¥«»]+){4})\s+(?P<value>.*)(?:\s+;.*)?\\?\s*$"
```

### What non-ideal capacitor model is being used?

[see this](https://www.ecircuitcenter.com/Calc/Cap_Model/Create_Cap_SPICE_Model.htm)
