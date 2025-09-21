import click
from importlib.metadata import version  
import calsipro.io
import calsipro.visualisations
from pathlib import Path
import logging
import logging.config
from multiprocessing import Pool
import glob

from calsiprovis.experiment import Experiment


def config_logging(debug, incremental=False):
    if debug:
        level = 'DEBUG'
    else:
        level = 'INFO'

    handlers = {'console': {'class': 'logging.StreamHandler',
                            'stream': 'ext://sys.stderr',
                            'level': level,
                            'formatter': 'detailed'}}

    formatters = {'detailed': {'format': '%(asctime)s %(levelname)-5s %(name)-16s %(message)s', 'datefmt': '%Y-%m-%d %H:%M:%S'}}

    logging.config.dictConfig({'version': 1,
                            'incremental': incremental,
                            'handlers': handlers,
                            'formatters': formatters,
                            'root': {'level': level, 'handlers': ['console']},
                        })
    if debug:
        logging.getLogger('calsipro.analysis').setLevel('DEBUG')


config_logging(False)

logger = logging.getLogger('cli')


def visualise(scene, d):
    logger.info(f'Visualising {scene}')
    calsipro.visualisations.show_mapped_picture(d, f'results/{scene}_max.tif')
    logger.info('Visualisation done')
    calsipro.visualisations.show_mapped_distribution(d, axis=0, out=f'results/{scene}_distribution.tif')

def analyse_scene(experiment, scene, video=True, html=False):

    from calsiprovis.library import load_max_projection, load_intensity_distribution, load_mask, load_fluro, _load_peaks, _load_peaks_simple, load_peaks, load_peaks_simple, load_video, speed_per_peak, speed_per_peak_simple, speed_analysis, speed_analysis_simple, _load_mask, load_fluro_distribution, load_fluro_mask, load_brightfield, load_brightfield_distribution

    logger.info('Checking max projection')
    load_max_projection(experiment, scene)
    logger.info('Checking intensity distribution')
    load_intensity_distribution(experiment, scene, html=html)
    logger.info('Checking mask')
    mask = _load_mask(experiment, scene)
    load_mask(experiment, scene)
    logger.info('Checking peaks')
    _, _, peaks, _ = _load_peaks(experiment, scene)
    _, _, peaks_simple, _ = _load_peaks_simple(experiment, scene)
    load_peaks(experiment, scene, html=html)
    load_peaks_simple(experiment, scene, html=html)
    if scene in experiment.brightfield.keys():
        logger.info('Checking brightfield')
        load_brightfield(experiment, scene)
        logger.info('Checking brightfield distribution')
        load_brightfield_distribution(experiment, scene, html=html)
    else:
        logger.warning(f'Scene {scene} does not have any brightfield information')
    if video:
        logger.info('Checking vidoe')
        load_video(experiment, scene)
    if peaks and mask.any():
        logger.info('Checking speed analysis')
        speed_analysis(experiment, scene)
    if peaks_simple and mask.any():
        logger.info('Checking speed analysis simple')
        speed_analysis_simple(experiment, scene)
    for fluro in experiment.fluros.keys():
        logger.info(f'Checking fluro {fluro} distribution')
        load_fluro_distribution(experiment, fluro, scene, html=html)
        logger.info(f'Checking fluro {fluro} mask')
        load_fluro_mask(experiment, fluro, scene)
        logger.info(f'Checking fluro {fluro} image')
        load_fluro(experiment, fluro, scene)
    if mask.any():
        logger.info('Checking beats')
        for idx in range(min(10, len(peaks))):
            speed_per_peak(experiment, scene, idx)
        for idx in range(min(10, len(peaks_simple))):
            speed_per_peak_simple(experiment, scene, idx)


@click.group()
def cli():
    '''CALcium SIgnal PROpagation VISualisation tool'''
    pass


@cli.command()
@click.argument("path")
def scenes(path):
    '''Load experiment PATH and display the name of all scenes'''
    experiment = Experiment.load(path)
    scenes = set(list(experiment.gcamp.keys()) + [scene for fluro in experiment.fluros for scene in experiment.fluros[fluro].keys()])
    scenes = sorted(scenes)
    for scene in scenes:
        print(scene)

@cli.command()
@click.argument("path")
@click.option("--cores", default=4, help='Number of cores to use in parallel')
def panalyse(path, cores):
    '''Analyse all scenes of the experiment at PATH in parallel.'''
    experiment = Experiment.load(path)
    scenes = set(list(experiment.gcamp.keys()) + [scene for fluro in experiment.fluros for scene in experiment.fluros[fluro].keys()])
    scenes = sorted(scenes)
    with Pool(cores) as p:
        results = []
        for scene in scenes:
            print(scene)
            res = p.apply_async(analyse_scene, (experiment, scene))
            results.append(res)
        for scene ,res in zip(scenes, results):
            print("waiting", scene)
            res.wait()


@cli.command()
@click.argument("path")
@click.argument("scene")
@click.option("--video/--no-video", default=True)
@click.option("--html/--normal", default=False)
@click.option('--debug/--no-debug', default=False)
def analyse(path, scene, video, html, debug):
    '''Analyse experiment at PATH the single scene SCENE.'''

    config_logging(debug, incremental=True)
    logger.info('Info logging activated')
    logger.debug('Debug logging activated')

    logger.info(f'Analysing project {path} scene {scene} video {video} output {html}')
    calsipro_version = version("calsipro")
    calsiprovis_version = version("calsiprovis")
    logger.info(f"Executing with version calsipro {calsipro_version} and calsiprovis {calsiprovis_version}.")
    experiment = Experiment.load(path)
    logger.info("Experiment loaded, starting analysis")
    analyse_scene(experiment, scene, video=video, html=html)
    logger.info("Calsipro done")


@cli.command()
@click.argument("path")
@click.argument("gcamp", nargs=-1)
@click.option("--enable-glob/--disable-glob", default=False, help='Expand the given gcamp file paths following shell globbing rules, i.e. expand "*"')
def add_gcamp(path, gcamp, enable_glob):
    '''Add one or more gcamp file(s) to the experiment at PATH'''
    _add_gcamp(path, gcamp, enable_glob)


def _add_gcamp(path, gcamp, enable_glob):
    path = Path(path).absolute()
    if path.exists():
        experiment = Experiment.load(str(path))
    else:
        experiment = Experiment.new(str(path))
    for gcamp in gcamp:
        if enable_glob:
            ps = glob.glob(gcamp)
        else:
            ps = [gcamp]
        for p in ps:
            gcamp = str(Path(p).absolute())
            experiment.add_gcamp(gcamp)


@cli.command()
@click.argument("path")
@click.argument("name")
@click.argument("fluro", nargs=-1)
@click.option("--enable-glob/--disable-glob", default=False, help='Expand the given gcamp file paths following shell globbing rules, i.e. expand "*"')
def add_fluro(path, fluro, name, enable_glob):
    '''Add one or more flurophor file(s) to the experiment at PATH.'''
    _add_fluro(path, fluro, name, enable_glob)


def _add_fluro(path, fluro, name, enable_glob):
    path = Path(path).absolute()
    if path.exists():
        experiment = Experiment.load(str(path))
    else:
        experiment = Experiment.new(str(path))
    for fluro in fluro:
        if enable_glob:
            ps = glob.glob(fluro)
        else:
            ps = [fluro]
        for p in ps:
            fluro = Path(p).absolute()
            experiment.add_fluro(str(fluro), name)


@cli.command()
@click.argument("path")
@click.argument("brightfield", nargs=-1)
@click.option("--enable-glob/--disable-glob", default=False, help='Expand the given gcamp file paths following shell globbing rules, i.e. expand "*"')
def add_brightfield(path, brightfield, enable_glob):
    '''Add one or more brightfield file(s) to the experiment at PATH.'''
    _add_brightfield(path, brightfield, enable_glob)


def _add_brightfield(path, brightfield, enable_glob):
    experiment_path = Path(path).absolute()
    if experiment_path.exists():
        experiment = Experiment.load(str(experiment_path))
    else:
        experiment = Experiment.new(str(experiment_path))
    for bf_path in brightfield:
        if enable_glob:
            paths = glob.glob(bf_path)
        else:
            paths = [bf_path]
        for single_path in paths:
            bf_file = str(Path(single_path).absolute())
            experiment.add_brightfield(bf_file)


@cli.command()
@click.argument("path")
@click.argument("scene")
@click.option('--debug/--no-debug', default=False)
def trace(path, scene, debug):
    '''Calculate only the trace for scene SCENE in experiment at PATH.'''

    config_logging(debug, incremental=True)
    logger.info('Info logging activated')
    logger.debug('Debug logging activated')

    logger.info(f'Creating trace for project {path} scene {scene}')
    calsipro_version = version("calsipro")
    calsiprovis_version = version("calsiprovis")
    logger.info(f"Executing with version calsipro {calsipro_version} and calsiprovis {calsiprovis_version}.")
    experiment = Experiment.load(path)

    logger.info("Experiment loaded, starting analysis")

    from bokeh.plotting import figure
    from calsiprovis.library import read_data, _load_mask
    from calsiprovis.bokeh_helpers import save
    import numpy as np

    path = experiment.gcamp[scene]
    data = read_data(path, scene=scene)[0]
    data = data[1:, :, :]
    mask = _load_mask(experiment, scene)
    m, mm = np.min(data), np.max(data)

    data = ((data - m) / (mm-m))

    signal = np.mean(data[:, mask], axis=(1))
    time = np.arange(len(signal))

    p1 = figure(width=100, height=100, x_range=(0, np.max(time)), y_range=(np.min(signal), np.max(signal)), toolbar_location=None, tools="wheel_zoom", active_scroll="wheel_zoom")
    p1.line(time, signal)

    cache_path = experiment.cache_path(scene, f'trace_{scene}.svg')
    save(p1, str(cache_path))
    logger.info("Calsipro done")


if __name__ == "__main__":
    cli()
