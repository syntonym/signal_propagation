# Calcium Signal Propagation speed analysis pipeline

Used in the paper "Multi-chamber cardioids unravel human heart development and cardiac defects" to analyse the speed of calcium signal propagation in cardioids.

## Installation

Have the [uv](https://docs.astral.sh/uv/) python package manager installed.
Execute the command `uv sync` in the `calsiprovis` subdirectory. All local dependencies (present in the directories `calsipro`, `runnel`, `rapids` and `hoernchen`) are build automatically.

## Running

### CLI usage

- Choose a writeable path that will hold a JSON file identifying all images that belong to this experiment, referred to as `$EXPERIMENT` in the next steps.
- Add GCamp files via `calsiprovis add-gcamp $EXPERIMENT $PATH_GCAMP_FILE`, with `$PATH_GCAMP_FILE` being a path to a gcamp (czi) file
- Use `calsiprovis add-brightfield $EXPERIMENT $PATH_BF_FILE` to add brightfield (czi) files
- Use `calsiprovis add-fluro $EXPERIMENT $FLURO_NAME $PATH_FLURO_FILE` to add flurophore files. `$FLURO_NAME` is the name of the flurophor used, e.g. bfp
- Use `calsiprovis panalyse --cores $CORES $EXPERIMENT` to analyse all wells of the experiment on `$CORES` many cores in parallel

### GUI Usage

Run the following command in the `calsiprovis` subdirectory:

```
uv run runnel ../rapids/dist/rapids-0.5.0-py3-none-any.whl dist/calsiprovis-0.18.0-py3-none-any.whl ../calsipro/dist/calsipro-0.16.0-py3-none-any.whl --foreground=calsiprovis.frontend:run --pyodide-dir=../runnel/src/runnel/pyodide/ --foreground-arg=$REFS --background=rapids.socket.backend:run
```

where `$REFS` is a writeable path that will hold a JSON containing references to multiple experiment files.

In the panel to add an experiment fill the top line with an experimental name, the bottom line with a path to the experiment file (corresponding to `$EXPERIMENT` in the CLI usage). Click the button `Add Experiment` to add the experiment to the `$REFS` file. Below the form to add an experiment is a list of buttons with all experiments. Click the button of an experiment to load it. This adds three new forms and a list of experiment read outs. The first form adds gcamp files to the experiment, fill in the path to the gcamp file and click the `Add Gcamp` button. The second form adds fluro files to the experiment, fill in the path to the czi file, the flurophore type (e.g. `bfp`) and click the `Add Fluro` button. The third form adds a brightfield file to the experiment, fill in the path to the czi file and click on `Add Brightfield`.

Below the forms are 6 buttons plus one button per added fluropore type and loads intermediate results for quality control. Before loading intermediate results, run the analysis with the `analyse` or `panalyse` CLI commands. Clicking the `max` button shows the maximal intensity projection of the gcamp files. Clicking the `mask` button displays the mask (identified organoid). Clicking the `video` button shows the gcamp timecourse as a video. Clicking the `peaks` button shows the peaks identified with the default peak finding algorithm. Clicking the `peaks simple` button shows the peaks identified with the `simple` algorithm. Clicking the `dist` button shows the maximal intensity histograms used to find the organoid mask. Clicking the `beat` button shows the beat image of the first identified beat. Clicking one of the flurophore buttons shows the image of that flurophore.

All analysis results are saved next to the `$EXPERIMENT` file.

## Packages overview

Most of the algorithms are present in the `calsipro` package, with some in `calsiprovis.library`. The rest of `calsiprovis` is the interactive web application, based on `rapids` and `runnel`. `runnel` starts a webserver and provides all necesarry tools to start python in a browser, whereas `rapids` has some helper functions to have applications run with `runnel`. Finally `hoernchen` is an sqlite-based ORM.

## Changelog

### Version 2

Version 2 incorporates some bugfixes and an improved README. No algorithms are changed in version 2.

- The packages `calsipro` and `calsiprovis` had incorrectly defined dependencies, which made them impossible to run without modifying the code. This version corrects the dependencies so the packages can be build and run. Thanks to Abhijeet Krishna for notifying me of this problem.
- In version 1 building and packaging was done by `poetry`. In version 2 building and packaging is done by `uv`, which also handles python versions.


### Version 1

Initial publication

