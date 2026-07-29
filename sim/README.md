# Simulator


## FAQ

### Why `spicelib`?

We are using `spicelib` instead of `PySpice` since it is being actively maintained. `PySpice` has many deprecated dependencies where examples would not work out of the box.

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
