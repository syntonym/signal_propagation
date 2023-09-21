Calcium Signal Propagation speed analysis pipeline

Used in the paper "Multi-chamber cardioids unravel human heart development and cardiac defects" to analyse the speed of calcium signal propagation in cardioids. Most of the algorithms are present in the `calsipro` package, with some in `calsiprovis.library`. The rest of `calsiprovis` is the interactive web application, based on `rapids` and `runnel`. `runnel` starts a webserver and provides all necesarry tools to start python in a browser, whereas `rapids` has some helper functions to have applications run with `runnel`. Finally `hoernchen` is an sqlite-based ORM.

For analysis use the `calsipro` functions or use the `calsiprovis` CLI. To run the interactive web application build all the python packages with `poetry`, create a python environment with all packages installed and run:

`runnel PATH_TO_RAPIDS_WHEEL PATH_TO_CALSIPROVIS_WHEEL--foreground=calsiprovis.app:run --pyodide-dir=./src/runnel/pyodide/ --foreground-arg=PATH_TO_EXPERIMENT_REFS --background=rapids.socket.backend:run`

where `PATH_TO_EXPERIMENT_REFS` points to file containing all experiments in a json format `[{'name': 'plate1', 'path': 'PATH_TO_EXPERIMENT_FILE'}, ...]`. The individual experiment files can be created and manipulated with commands provided by the calsiprovis CLI. Three commands add CZI files to the experiment: `add_gcamp`, `add_fluro` and `add_brightfield`, for details see the individual commands.

