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

### Simulation error

If you are getting something like the following error, then it could be due to the ngspice version. Versions **35** and **47** are known to be working. Version **45** with `Ubuuntu 24.04` did not work.

```
simulation command "run" failed
```

**To fix:**

> There is a more efficient way to do this but, this is just what we did to get it to work.

Version 47 can be compiled from source and downloaded from [sourceforge](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/47/).

We ran the following for the `ngspice` binary.

```
./compile_linux.sh
```

Then ran the fallowing to build the shared library.

```
mkdir ngspice-47-build
pushd ngspice-47-build

../configure \
  --prefix=/usr/local \
  --enable-xspice \
  --disable-debug \
  --enable-cider \
  --with-readline=yes \
  --enable-openmp \
  --with-ngshared

make -j$(nproc)
```

These were the paths to the compiled binaries.

```
./ngspice-47/release/src/ngspice"
./ngspice-47/ngspice-47-build/src/.libs/libngspice.so.0.0.16"
```

Then you can set environment variables to specify these versions. Adding them to `.bashrc` will make them persistant.

```
export SPICE_COMMAND="/home/mwmast/Downloads/ngspice-47/release/src/ngspice"
export LIBRARY_PATH="/home/mwmast/Downloads/ngspice-47/ngspice-47-build/src/.libs/libngspice.so.0.0.16"
```

Resources:
- https://pyspice.fabrice-salvaire.fr/releases/v1.4/installation.html#ngspice-compilation
- https://pyspice.fabrice-salvaire.fr/releases/v1.4/faq.html#how-to-set-the-ngspice-library-path
