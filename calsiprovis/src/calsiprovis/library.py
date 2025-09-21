from pathlib import Path
from rapids.backend import backend, is_backend
from dataclasses import dataclass, field
from typing import Optional, Any
from decimal import Decimal
from calsiprovis.experiment import Experiment
import logging

if is_backend:
    import base64
    import imageio.v3 as iio
    import json
    from calsipro.io import show_scenes, read_data
    import pickle
    import calsipro.visualisations
    import calsipro.peak_calling
    from bokeh.embed import json_item
    from bokeh.plotting import figure
    from bokeh.io import save as bokeh_save
    import numpy as np
    import zlib
    import scipy.ndimage
    from aicsimageio import AICSImage

    from bokeh.resources import INLINE
    import json
    import zstandard
    from matplotlib import cm

logger = logging.getLogger(__name__)


MAX_PROJECTION_VERSION = (0, 9, 2)
INTENSITY_DISTRIBUTION_VERSION = (0, 9, 2)
MASK_VERSION = (0, 15, 8)
FLURO_VERSION = (0, 16, 6)
BRIGHTFIELD_VERSION = (0, 9, 2)
CALCULATE_THRESHOLD_VERSION = (0, 11, 0)
PEAK_CALLING_VERSION = (0, 15, 11)
SPEED_VERSION = (0, 16, 12)
PEAK_CALLING_SIMPLE_VERSION = (0, 16, 10)
SPEED_SIMPLE_VERSION = (0, 16, 12)

FAIL_ON_CACHE_MISS = False

MEMORY_CACHE = {}

@dataclass
class DataOrigin:
    scene: str
    path: str
    f1: Optional[str] = None
    f2: Optional[str] = None
    f3: Optional[str] = None

@dataclass
class ExperimentRef:
    name: str
    path: str

    def serialize(self):
        return {'name': self.name, 'path': self.path}

    @classmethod
    def deserialize(cls, d):
        return ExperimentRef(d['name'], d['path'])

def version_compatible(saved, current):
    if saved is None:
        return False
    for saved_v, current_v in zip(saved, current):
        if saved_v < current_v:
            return False
    return True

def max_version(*versions):
    running = list(range(len(versions)))
    v_idx = 0
    max_version = max(len(v) for v in versions)
    while v_idx < max_version:
        m = max(versions[i][v_idx] for i in running)
        new_running = [i for i in running if versions[i][v_idx] == m]
        if len(new_running) == 0:
            return versions[running[0]]
        elif len(new_running) == 1:
            return versions[new_running[0]]
        v_idx += 1
        running = new_running
    return versions[running[0]]


def zstandard_compress(data):
    ctx = zstandard.ZstdCompressor()
    return ctx.compress(data)


def zstandard_decompress(data):
    ctx = zstandard.ZstdDecompressor()
    return ctx.decompress(data)


def bokeh_json_cache(cache_path, f, version):
    if cache_path in MEMORY_CACHE:
        return MEMORY_CACHE[cache_path]

    recalculate = True
    version_path = cache_path.parent / (".version_" + cache_path.name)

    if version_path.exists():
        with version_path.open(mode="rb") as file:
            current_version = pickle.load(file)
    else:
        current_version = None

    if version_compatible(current_version, version) and cache_path.exists():
        try:
            with cache_path.open(mode='rb') as file:
                compressed_figure = file.read()
            serialized_figure = zstandard_decompress(compressed_figure).decode('utf8')
            recalculate = False
        except Exception as e:
            logger.exception('Exception during cache loading')
            recalculate= True
    else:
        recalculate = True

    if recalculate:
        value = f()
        jp = json_item(value)
        serialized_figure = json.dumps(jp)
        compressed_figure = zstandard_compress(serialized_figure.encode('utf8'))
        with cache_path.open(mode='wb') as f:
            f.write(compressed_figure)
        with version_path.open(mode="wb") as file:
            pickle.dump(version, file)
    MEMORY_CACHE[cache_path] = serialized_figure
    return serialized_figure


def bokeh_html_cache(cache_path, f, version):
    if cache_path in MEMORY_CACHE:
        return MEMORY_CACHE[cache_path]
    recalculate = True
    version_path = cache_path.parent / (".version_" + cache_path.name)

    if version_path.exists():
        with version_path.open(mode="rb") as file:
            current_version = pickle.load(file)
    else:
        current_version = None

    if version_compatible(current_version, version) and cache_path.exists():
        try:
            with cache_path.open(mode='r') as file:
                figure = file.read()
            recalculate = False
        except Exception as e:
            logger.exception('Exception during cache loading')
            recalculate= True
    if recalculate:
        value = f()
        bokeh_save(value, cache_path)
        with cache_path.open(mode='r') as f:
            figure = f.read()
        with version_path.open(mode="wb") as file:
            pickle.dump(version, file)
    MEMORY_CACHE[cache_path] = figure
    return figure


def pickle_cache(cache_path, f, version):
    if cache_path in MEMORY_CACHE:
        return MEMORY_CACHE[cache_path]
    recalculate = True
    version_path = cache_path.parent / (".version_" + cache_path.name)

    if version_path.exists():
        with version_path.open(mode="rb") as file:
            current_version = pickle.load(file)
    else:
        current_version = None

    if not version_compatible(current_version, version):
        logger.info(f'{cache_path.name}: Version {current_version} is incompatible with requested version {version}, invalidating cache')
    elif not cache_path.exists():
        logger.info(f'{cache_path.name}: Cache {cache_path} does not exist')
    else:
        logger.info(f'{cache_path.name}: Using cache version {current_version} for requested version {version}')
        try:
            with cache_path.open(mode='rb') as file:
                compressed_value = file.read()
            serialized_value = zstandard_decompress(compressed_value)
            value = pickle.loads(serialized_value)
            recalculate = False
        except Exception as e:
            if FAIL_ON_CACHE_MISS:
                raise e
            recalculate= True
    if recalculate:
        if FAIL_ON_CACHE_MISS:
            raise ValueError(f"Cache miss for file {cache_path} with version {version}, only {current_version} is present")
        value = f()
        serialized_value = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_value = zstandard_compress(serialized_value)
        with cache_path.open(mode='wb') as file:
            file.write(compressed_value)
        with version_path.open(mode="wb") as file:
            pickle.dump(version, file)

    MEMORY_CACHE[cache_path] = value
    return value


def image_cache(cache_path, f, version):
    if cache_path in MEMORY_CACHE:
        return MEMORY_CACHE[cache_path]
    recalculate = True
    version_path = cache_path.parent / (".version_" + cache_path.name)

    if version_path.exists():
        with version_path.open(mode="rb") as file:
            current_version = pickle.load(file)
    else:
        current_version = None

    if version_compatible(current_version, version) and cache_path.exists():
        try:
            value = iio.imread(cache_path)
            recalculate = False
        except:
            recalculate= True
    if recalculate:
        value = f()
        iio.imwrite(cache_path, value)
        with version_path.open(mode="wb") as file:
            pickle.dump(version, file)
    MEMORY_CACHE[cache_path] = value
    return value



@backend
def get_bokeh_script():
    return INLINE.js_raw


@backend
def get_experiment(path):
    if Path(path).exists():
        print("Loading experiment", path)
        experiment = Experiment.load(path)
    else:
        print("Creating new experiment", path)
        try:
            experiment = Experiment.new(path)
        except Exception as e:
            logger.exception('Exception during experiment loading')
            experiment = None
    return experiment


@backend
def add_gcamp_to(experiment, gcamp_path):
    experiment.add_gcamp(gcamp_path)
    experiment.save()
    return experiment


@backend
def add_brightfield_to(experiment, brightfield_path):
    experiment.add_brightfield(brightfield_path)
    experiment.save()
    return experiment


@backend
def add_fluro_to(experiment, fluro, fluro_path):
    experiment.add_fluro(fluro_path, fluro)
    experiment.save()
    return experiment


def load_data_origins(experiment):
    scenes = set(list(experiment.gcamp.keys()) + [scene for fluro in experiment.fluros for scene in experiment.fluros[fluro].keys()])

    fluros = [experiment.fluros[fluro_id] for fluro_id in experiment.fluro_order]

    origins = []
    for scene in scenes:
        do = DataOrigin(scene, experiment.gcamp[scene], [fluro.get(scene, None) for fluro in fluros])
        origins.append(do)
    return origins


@backend
def load_max_projection(experiment, scene, width=100, height=100):

    path = experiment.gcamp[scene]

    def generate():
        logger.info('Generating max projection')
        data = read_data(path, scene=scene)[0]
        data = data[1:, :, :]
        d = np.max(data, axis=0)
        d = calsipro.analysis.normalize(d)
        d = (d*255).astype(np.uint8)
        return d

    image = image_cache(experiment.cache_path(scene, 'max_projection.png'), generate, MAX_PROJECTION_VERSION)

    b = iio.imwrite('<bytes>', image, extension='.png')
    encoded = base64.b64encode(b).decode("utf8")
    return f'<img width={width} height={height} src="data:image/png;base64,{encoded}" />'


@backend
def load_intensity_distribution(experiment, scene, html=False):

    path = experiment.gcamp[scene]

    def generate():
        logger.info('Generating intensity distribution')
        data = read_data(path, scene=scene)[0]
        data = data[1:, :, :]
        d = np.max(data, axis=0)
        #d = calsipro.analysis.normalize(d)

        if np.min(d) == 0:
            offset = 1
        else:
            offset = 0
        freq, bins = np.histogram(np.log(d+offset), bins=80)
        p = figure(height_policy='max', width_policy='max', frame_width=100, frame_height=100, toolbar_location=None, y_axis_type='log')
        p.quad(top=freq, bottom=1, left=bins[:-1], right=bins[1:])
        p.xaxis.visible = False
        p.yaxis.visible = False

        threshold = np.log(calsipro.analysis.calculate_threshold(d))
        p.line(x=[threshold, threshold], y=[1, np.max(freq)], color='red')
        return p

    version = INTENSITY_DISTRIBUTION_VERSION
    if html:
        cache_path = experiment.cache_path(scene, 'distribution.html')
        serialized_figure = bokeh_html_cache(cache_path, generate, version)
    else:
        cache_path = experiment.cache_path(scene, 'distribution.json.zstd')
        serialized_figure = bokeh_json_cache(cache_path, generate, version)
    return serialized_figure


def _calculate_mask_from_gcamp(experiment, scene, min_size=20):
    path = experiment.gcamp[scene]
    data = read_data(path, scene=scene)[0]
    data = data[1:, :, :]
    d = np.var(data, axis=0)

    threshold = calsipro.analysis.calculate_threshold(d)
    mask = calsipro.analysis.calculate_mask(d, th=threshold, raw=True)
    return mask


def _calculate_mask_from_brightfield(experiment, scene, min_size=20):
    path = experiment.brightfield[scene]
    data = read_data(path, scene=scene)[0]
    data = data[0, :, :]
    assert len(data.shape) == 2
    mask = calsipro.analysis.calculate_bf_mask(data)
    mask = calsipro.analysis.shrink_mask(mask, 5)
    return mask


def _load_mask(experiment, scene, min_size=20):
    def generate():
        if scene in experiment.brightfield:
            logger.info('Generating mask from brightfield')
            mask = _calculate_mask_from_brightfield(experiment, scene, min_size=min_size)
        else:
            logger.info('Generating mask from gcamp')
            mask = _calculate_mask_from_gcamp(experiment, scene, min_size=min_size)
        return mask

    cache_path = experiment.cache_path(scene, 'mask.pickle.zstd')
    result = pickle_cache(cache_path, generate, MASK_VERSION)
    return result


def _load_fluro(experiment, fluro, scene):
    def generate():
        logger.info(f'Generating fluro {fluro}')
        path = experiment.fluros[fluro][scene]
        data = read_data(path, scene=scene)[0]
        data = data[0, :, :]
        m = np.min(data)
        return data
    cache_path = experiment.cache_path(scene, f'{fluro}.pickle.zstd')
    result = pickle_cache(cache_path, generate, FLURO_VERSION)
    return result


@backend
def load_mask(experiment, scene, width=100, height=100):

    def generate():
        mask = _load_mask(experiment, scene)
        return mask

    cache_path = experiment.cache_path(scene, 'mask.png')
    image = image_cache(cache_path, generate, MASK_VERSION)

    b = iio.imwrite('<bytes>', image, extension='.png')
    encoded = base64.b64encode(b).decode("utf8")
    return f'<img width={width} height={height} src="data:image/png;base64,{encoded}" />'


def _load_brightfield(experiment, scene):
    path = experiment.brightfield[scene]
    def generate():
        data = read_data(path, scene=scene)[0][0,:,:]
        return data
    cache_path = experiment.cache_path(scene, f'brightfield.pickle.zstd')
    result = pickle_cache(cache_path, generate, BRIGHTFIELD_VERSION)
    return result


@backend
def load_brightfield(experiment, scene, width=100, height=100):

    def generate():
        d = _load_brightfield(experiment, scene)
        d = calsipro.analysis.normalize(d)
        d = (d*255).astype(np.uint8)
        return d

    cache_path = experiment.cache_path(scene, f'brightfield.png')
    image = image_cache(cache_path, generate, BRIGHTFIELD_VERSION)

    b = iio.imwrite('<bytes>', image, extension='.png')
    encoded = base64.b64encode(b).decode("utf8")
    return f'<img width={width} height={height} src="data:image/png;base64,{encoded}" />'


@backend
def load_brightfield_distribution(experiment, scene, html=False):
    def generate():
        logger.info('Generating Brightfield')
        d = _load_brightfield(experiment, scene)
        if np.min(d) == 0:
            offset = 1
        else:
            offset = 0
        counts, bins = np.histogram(np.log(d+offset), bins=80)
        freq = np.log(1+counts)
        p = figure(height_policy='max', width_policy='max', frame_width=100, frame_height=100, toolbar_location=None, y_axis_type='log')
        p.quad(top=freq, bottom=1, left=bins[:-1], right=bins[1:])
        p.xaxis.visible = False
        p.yaxis.visible = False

        threshold = np.log(calsipro.analysis.calculate_threshold(d))
        p.line(x=[threshold, threshold], y=[1, np.max(freq)], color='red')
        return p

    version = BRIGHTFIELD_VERSION
    if html:
        cache_path = experiment.cache_path(scene, f'brightfield_distribution.html')
        serialized_figure = bokeh_html_cache(cache_path, generate, version)
    else:
        cache_path = experiment.cache_path(scene, f'brightfield_distribution.json.zstd')
        serialized_figure = bokeh_json_cache(cache_path, generate, version)
    return serialized_figure


@backend
def load_fluro(experiment, fluro, scene, width=100, height=100):

    def generate():
        d = _load_fluro(experiment, fluro, scene)
        d = calsipro.analysis.normalize(d)
        d = (d*255).astype(np.uint8)
        return d

    cache_path = experiment.cache_path(scene, f'{fluro}.png')
    image = image_cache(cache_path, generate, FLURO_VERSION)

    b = iio.imwrite('<bytes>', image, extension='.png')
    encoded = base64.b64encode(b).decode("utf8")
    return f'<img width={width} height={height} src="data:image/png;base64,{encoded}" />'


def _calculate_fluro_threshold_and_mask(data, organoid_mask):
    if not organoid_mask.any():
        return organoid_mask
    mask = calsipro.analysis._calculate_fluro_threshold_and_mask(data, organoid_mask)
    return mask


@backend
def load_fluro_mask(experiment, fluro, scene, width=100, height=100):

    def generate():
        logger.info(f'Generating fluro {fluro} mask')
        f = _load_fluro(experiment, fluro, scene)
        organoid_mask = _load_mask(experiment, scene)
        mask = _calculate_fluro_threshold_and_mask(f, organoid_mask)
        return mask

    version = max_version(FLURO_VERSION, CALCULATE_THRESHOLD_VERSION, MASK_VERSION)
    cache_path = experiment.cache_path(scene, f'mask_{fluro}.png')
    image = image_cache(cache_path, generate, version)

    b = iio.imwrite('<bytes>', image, extension='.png')
    encoded = base64.b64encode(b).decode("utf8")
    return f'<img width={width} height={height} src="data:image/png;base64,{encoded}" />'


@backend
def load_fluro_distribution(experiment, fluro, scene, html=False):
    def generate():
        logger.info(f'Generating fluro {fluro} distribution')
        d = _load_fluro(experiment, fluro, scene)
        freq, counts, bins = calsipro.analysis.calculate_histogram(d)

        p = figure(height_policy='max', width_policy='max', frame_width=100, frame_height=100, toolbar_location=None, y_axis_type='log')
        p.quad(top=freq, bottom=1, left=bins[:-1], right=bins[1:])
        p.xaxis.visible = False
        p.yaxis.visible = False

        threshold = np.log(calsipro.analysis.calculate_threshold(d))
        p.line(x=[threshold, threshold], y=[1, np.max(freq)], color='red')
        return p

    version = FLURO_VERSION
    if html:
        cache_path = experiment.cache_path(scene, f'{fluro}_distribution.html')
        serialized_figure = bokeh_html_cache(cache_path, generate, version)
    else:
        cache_path = experiment.cache_path(scene, f'{fluro}_distribution.json.zstd')
        serialized_figure = bokeh_json_cache(cache_path, generate, version)
    return serialized_figure


def _load_peaks(experiment, scene):
    path = experiment.gcamp[scene]

    def generate():
        logger.info('Generating peak calling')
        data = read_data(path, scene=scene)[0]
        data = data[1:, :, :]

        mask = _load_mask(experiment, scene)

        m, mm = np.min(data), np.max(data)

        data = ((data - m) / (mm-m))

        signal = np.mean(data[:, mask], axis=(1))
        time = np.arange(len(signal))

        peaks, baseline = calsipro.peak_calling.call_peaks(time, signal)

        return time, signal, peaks, baseline

    version = max_version(PEAK_CALLING_VERSION, BRIGHTFIELD_VERSION, MASK_VERSION)
    cache_path = experiment.cache_path(scene, 'peaks.pickle.zstd')
    time, signal, peaks, baseline = pickle_cache(cache_path, generate, version)
    return time, signal, peaks, baseline


def _load_peaks_simple(experiment, scene):
    path = experiment.gcamp[scene]

    def generate():
        logger.info('Generating peak calling')
        data = read_data(path, scene=scene)[0]
        data = data[1:, :, :]

        mask = _load_mask(experiment, scene)

        m, mm = np.min(data), np.max(data)

        data = ((data - m) / (mm-m))

        signal = np.mean(data[:, mask], axis=(1))
        time = np.arange(len(signal))

        peaks, baseline = calsipro.peak_calling.call_peaks_simple(time, signal)

        return time, signal, peaks, baseline

    version = max_version(PEAK_CALLING_SIMPLE_VERSION, BRIGHTFIELD_VERSION, MASK_VERSION)
    cache_path = experiment.cache_path(scene, 'peaks_simple.pickle.zstd')
    time, signal, peaks, baseline = pickle_cache(cache_path, generate, version)
    return time, signal, peaks, baseline


@backend
def load_number_of_peaks(experiment, scene):
    time, signal, peaks, baseline = _load_peaks(experiment, scene)
    return len(peaks)


@backend
def load_number_of_peaks_simple(experiment, scene):
    time, signal, peaks, baseline = _load_peaks_simple(experiment, scene)
    return len(peaks)


@backend
def load_peaks(experiment, scene, html=False):
    path = experiment.gcamp[scene]

    def generate():
        logger.info('Generating peaks figure')
        time, signal, peaks, baseline = _load_peaks(experiment, scene)
        if peaks:
            starts, ends = zip(*peaks)
            ends = np.array(ends)-1
        else:
            starts, ends = [], []

        p1 = figure(width=100, height=100, x_range=(0, np.max(time)), y_range=(np.min(signal), np.max(signal)), toolbar_location=None, tools="wheel_zoom", active_scroll="wheel_zoom")
        p1.line(time, signal)
        p1.circle(np.take(time, starts), np.take(signal, starts), color="green", fill_alpha=0.1, size=10)
        p1.circle(np.take(time, ends), np.take(signal, ends), color="red", fill_alpha=0.1, size=10)

        p1.xaxis.visible = False
        p1.yaxis.visible = False

        return p1

    version = max_version(PEAK_CALLING_VERSION, BRIGHTFIELD_VERSION, MASK_VERSION)
    if html:
        cache_path = experiment.cache_path(scene, 'peaks.html')
        serialized_figure = bokeh_html_cache(cache_path, generate, version)
    else:
        cache_path = experiment.cache_path(scene, 'peaks.json.zstd')
        serialized_figure = bokeh_json_cache(cache_path, generate, version)

    return serialized_figure


@backend
def load_peaks_simple(experiment, scene, html=False):
    path = experiment.gcamp[scene]

    def generate():
        logger.info('Generating peaks figure')
        time, signal, peaks, baseline = _load_peaks_simple(experiment, scene)
        if peaks:
            starts, ends = zip(*peaks)
            ends = np.array(ends)-1
        else:
            starts, ends = [], []

        p1 = figure(width=100, height=100, x_range=(0, np.max(time)), y_range=(np.min(signal), np.max(signal)), toolbar_location=None, tools="wheel_zoom", active_scroll="wheel_zoom")
        p1.line(time, signal)
        p1.circle(np.take(time, starts), np.take(signal, starts), color="green", fill_alpha=0.1, size=10)
        p1.circle(np.take(time, ends), np.take(signal, ends), color="red", fill_alpha=0.1, size=10)

        p1.xaxis.visible = False
        p1.yaxis.visible = False

        return p1

    version = max_version(PEAK_CALLING_SIMPLE_VERSION, BRIGHTFIELD_VERSION, MASK_VERSION)
    if html:
        cache_path = experiment.cache_path(scene, 'peaks_simple.html')
        serialized_figure = bokeh_html_cache(cache_path, generate, version)
    else:
        cache_path = experiment.cache_path(scene, 'peaks_simple.json.zstd')
        serialized_figure = bokeh_json_cache(cache_path, generate, version)

    return serialized_figure


@backend
def load_video(experiment, scene, width=100, height=100, fps=50):
    path = experiment.gcamp[scene]
    cache_path = experiment.cache_path(scene, f'video_{width}_{height}.mp4')
    if cache_path.exists():
        with cache_path.open(mode='rb') as f:
            serialized_video = f.read()
    else:
        logger.info('Generating video')
        data = calsipro.io.read_data(path, scene=scene)[0]
        m, mm = np.min(data), np.max(data)
        data = np.iinfo(np.uint8).max * ((data - m) / (mm-m))
        data = data.astype(np.uint8)
        serialized_video = calsipro.io.write_video(data, width=width, height=height, fps=fps, encoder="h264")

        with cache_path.open(mode='wb') as f:
            f.write(serialized_video)

    b = base64.b64encode(serialized_video).decode("utf8")
    return f'<video controls width={width} height={height}><source type="video/mp4" src="data:video/webm;base64,{b}"></video>'


@backend
def load_bpm(experiment, scene, fps=1):
    mask = _load_mask(experiment, scene)
    if mask.any():
        time, signal, peaks, baseline = _load_peaks(experiment, scene)
        return len(peaks) / (len(signal) * fps)
    else:
        return 0

@backend
def load_bpm_simple(experiment, scene, fps=1):
    mask = _load_mask(experiment, scene)
    if mask.any():
        time, signal, peaks, baseline = _load_peaks_simple(experiment, scene)
        return len(peaks) / (len(signal) * fps)
    else:
        return 0


@dataclass
class ResultSpeed:
    peak: int
    speed: Decimal
    fluro_speeds: list[Decimal]
    origin: Optional[str]
    total_frames: int
    fluro_names: Optional[str]


def _calculate_threshold_and_mask(data, min_size=30, larger=True):
    calculation_needed = True
    pick = 1
    while calculation_needed and pick < 80:
        threshold = calsipro.analysis.calculate_threshold(data, pick=pick)
        mask = calsipro.analysis.calculate_mask(data, th=threshold, raw=True, larger=larger)
        mask_size = mask.sum()
        if mask_size == 0:
            calculation_needed = False
        elif mask_size < min_size:
            pick += 1
        else:
            calculation_needed = False

    return mask


@backend
def speed_per_peak(experiment, scene, peak_idx):
    def generate():
        logger.info('Generating beat image')

        path = experiment.gcamp[scene]
        data = read_data(path, scene=scene)[0]
        data = data[1:, :, :]
        m, mm = np.min(data), np.max(data)
        signal = ((data - m) / (mm-m))

        _, _, peaks, _= _load_peaks(experiment, scene)
        mask = _load_mask(experiment, scene)
        peak = peaks[peak_idx]

        start, end = peak
        s = signal[start:end]
        time = calsipro.analysis.time_analysis(s)

        var = np.var(s, axis=0)
        var_mask = _calculate_threshold_and_mask(var)
        mask = mask & var_mask

        if mask.any():
            time = calsipro.analysis.push_low_pixels(time, mask)

        time = calsipro.analysis.normalize(time)
        time = 1-time
        time = calsipro.analysis.normalize(time)
        time = cm.inferno(time)
        time[:,:,3][~mask] = 0

        time = calsipro.analysis.normalize(time)
        time = (time*255).astype(np.uint8)
        return time

    version = max_version(PEAK_CALLING_VERSION, BRIGHTFIELD_VERSION, MASK_VERSION, SPEED_VERSION)
    image = image_cache(experiment.cache_path(scene, f'beat_{peak_idx}.png'), generate, version)

    b = iio.imwrite('<bytes>', image, extension='.png')
    encoded = base64.b64encode(b).decode("utf8")
    return f'<img width=100 height=100 src="data:image/png;base64,{encoded}" />'


@backend
def speed_per_peak_simple(experiment, scene, peak_idx):
    def generate():
        logger.info('Generating beat image')

        path = experiment.gcamp[scene]
        data = read_data(path, scene=scene)[0]
        data = data[1:, :, :]
        m, mm = np.min(data), np.max(data)
        signal = ((data - m) / (mm-m))

        _, _, peaks, _= _load_peaks_simple(experiment, scene)
        mask = _load_mask(experiment, scene)
        peak = peaks[peak_idx]

        start, end = peak
        s = signal[start:end]
        time = calsipro.analysis.time_analysis(s)

        var = np.var(s, axis=0)
        var_mask = _calculate_threshold_and_mask(var)
        mask = mask & var_mask

        if mask.any():
            time = calsipro.analysis.push_low_pixels(time, mask)

        time = calsipro.analysis.normalize(time)
        time = 1-time
        time = calsipro.analysis.normalize(time)
        time = cm.inferno(time)
        time[:,:,3][~mask] = 0

        time = calsipro.analysis.normalize(time)
        time = (time*255).astype(np.uint8)
        return time

    version = max_version(PEAK_CALLING_SIMPLE_VERSION, BRIGHTFIELD_VERSION, MASK_VERSION, SPEED_SIMPLE_VERSION)
    image = image_cache(experiment.cache_path(scene, f'beat_simple_{peak_idx}.png'), generate, version)

    b = iio.imwrite('<bytes>', image, extension='.png')
    encoded = base64.b64encode(b).decode("utf8")
    return f'<img width=100 height=100 src="data:image/png;base64,{encoded}" />'


@backend
def speed_analysis(experiment, scene):

    def generate():
        logger.info('Generating speed analysis')
        results = []

        fluros_meta = []
        fluros = []
        fluros_mask = []
        for fluro in experiment.fluros:
            fluros_meta.append(fluro)
            f = _load_fluro(experiment, fluro, scene)
            f_threshold = calsipro.analysis.calculate_threshold(f)
            f_mask = calsipro.analysis.calculate_mask(f, th=f_threshold, raw=True)
            fluros_mask.append(f_mask)

            m, mm = np.min(f), np.max(f)
            f_norm = ((f-m) / (mm-m))
            f_norm[~f_mask] = 0.0
            fluros.append(f_norm)

        global_mask = _load_mask(experiment, scene)

        fluros_meta.append('bf')
        bf_mask = global_mask
        for fmask in fluros_mask:
            bf_mask = bf_mask & (~fmask)
        fluros_mask.append(bf_mask)

        _, _, peaks, _= _load_peaks(experiment, scene)

        path = experiment.gcamp[scene]

        data = read_data(path, scene=scene)[0]
        data = data[1:, :, :]

        m, mm = np.min(data), np.max(data)

        signal = ((data - m) / (mm-m))


        resolution = get_resolution(experiment)
        assert resolution[0] == resolution[1]
        resolution = resolution[0]

        TIME_LENGTH = 30
        fps_to_mps = resolution * len(signal)/TIME_LENGTH

        if global_mask.any():

            for peak_index, peak in enumerate(peaks):
                start, end = peak
                s = signal[start:end]

                var = np.var(s, axis=0)
                var_mask = _calculate_threshold_and_mask(var)
                mask = global_mask & var_mask

                fluros_speed = []
                ori = None
                time = calsipro.analysis.time_analysis(s)
                if mask.any():
                    time = calsipro.analysis.push_low_pixels(time, mask)
                    df, speed = calsipro.analysis.calculate_speed(time, mask)
                    speed = fps_to_mps * speed
                    if len(fluros_meta) > 0:
                        ori_mask = calsipro.analysis.find_ori_cluster(time, mask, as_index=False)
                        f = [np.sum(fmask[ori_mask]) for fmask in fluros_mask]
                        m = np.argmax(f)
                        ori = fluros_meta[np.argmax(f)]

                        for fluro_mask in fluros_mask:
                            if fluro_mask.any():
                                df, fluro_speed = calsipro.analysis.calculate_speed(time, mask & (fluro_mask | ori_mask))
                                fluro_speed = fps_to_mps * fluro_speed
                                fluros_speed.append(Decimal(fluro_speed).quantize(Decimal("0.01")))
                            else:
                                fluros_speed.append(None)
                else:
                    speed = -2
                r = ResultSpeed(peak_index, Decimal(speed).quantize(Decimal("0.01")), fluros_speed, origin=ori, total_frames=len(signal), fluro_names=fluros_meta)
                results.append(r)
        return results

    version = max_version(PEAK_CALLING_VERSION, BRIGHTFIELD_VERSION, MASK_VERSION, SPEED_VERSION)
    cache_path = experiment.cache_path(scene, 'speed.pickle.zstd')
    results = pickle_cache(cache_path, generate, version)
    return results

@backend
def speed_analysis_simple(experiment, scene):

    def generate():
        logger.info('Generating simple speed analysis')
        results = []

        fluros_meta = []
        fluros = []
        fluros_mask = []
        for fluro in experiment.fluros:
            fluros_meta.append(fluro)
            f = _load_fluro(experiment, fluro, scene)
            f_threshold = calsipro.analysis.calculate_threshold(f)
            f_mask = calsipro.analysis.calculate_mask(f, th=f_threshold, raw=True)
            fluros_mask.append(f_mask)

            m, mm = np.min(f), np.max(f)
            f_norm = ((f-m) / (mm-m))
            f_norm[~f_mask] = 0.0
            fluros.append(f_norm)

        global_mask = _load_mask(experiment, scene)

        fluros_meta.append('bf')
        bf_mask = global_mask
        for fmask in fluros_mask:
            bf_mask = bf_mask & (~fmask)
        fluros_mask.append(bf_mask)

        _, _, peaks, _= _load_peaks_simple(experiment, scene)

        path = experiment.gcamp[scene]

        data = read_data(path, scene=scene)[0]
        data = data[1:, :, :]

        m, mm = np.min(data), np.max(data)

        signal = ((data - m) / (mm-m))

        global_mask = _load_mask(experiment, scene)

        resolution = get_resolution(experiment)
        assert resolution[0] == resolution[1]
        resolution = resolution[0]

        TIME_LENGTH = 30
        fps_to_mps = resolution * len(signal)/TIME_LENGTH

        if global_mask.any():

            for peak_index, peak in enumerate(peaks):
                start, end = peak
                s = signal[start:end]

                var = np.var(s, axis=0)
                var_mask = _calculate_threshold_and_mask(var)
                mask = global_mask & var_mask

                fluros_speed = []
                ori = None
                time = calsipro.analysis.time_analysis(s)
                if mask.any():
                    time = calsipro.analysis.push_low_pixels(time, mask)
                    df, speed = calsipro.analysis.calculate_speed(time, mask)
                    speed = fps_to_mps * speed
                    if len(fluros) > 0:
                        ori_mask = calsipro.analysis.find_ori_cluster(time, mask, as_index=False)
                        f = [np.mean(fluro[ori_mask]) for fluro in fluros]
                        m = np.argmax(f)
                        ori = "bf"
                        if f[m] > 0.4:
                            ori = fluros_meta[np.argmax(f)]

                        for fluro_mask in fluros_mask:
                            if fluro_mask.any():
                                df, fluro_speed = calsipro.analysis.calculate_speed(time, fluro_mask)
                                fluro_speed = fps_to_mps * fluro_speed
                                fluros_speed.append(Decimal(fluro_speed).quantize(Decimal("0.01")))
                            else:
                                fluros_speed.append(None)
                else:
                    speed = -2
                r = ResultSpeed(peak_index, Decimal(speed).quantize(Decimal("0.01")), fluros_speed, origin=ori, total_frames=len(signal), fluro_names=fluros_meta)
                results.append(r)
        return results

    version = max_version(PEAK_CALLING_SIMPLE_VERSION, BRIGHTFIELD_VERSION, MASK_VERSION, SPEED_SIMPLE_VERSION)
    cache_path = experiment.cache_path(scene, 'speed_simple.pickle.zstd')
    results = pickle_cache(cache_path, generate, version)
    return results

@backend
def load_experiment_refs(path):
    if Path(path).exists():
        with open(path) as f:
            serialized_experiments = json.load(f)
        experiments = [ExperimentRef.deserialize(s) for s in serialized_experiments]
    else:
        experiments = []
    return experiments


@backend
def add_experiment_ref(path, experiment):
    if Path(path).exists():
        with open(path) as f:
            serialized_experiments = json.load(f)
    else:
        serialized_experiments = []
    serialized_experiments.append(experiment.serialize())
    with open(path, mode='w') as f:
        json.dump(serialized_experiments, f)


@backend
def get_resolution(experiment):
    key = list(experiment.gcamp.keys())[0]
    path = experiment.gcamp[key]
    img = AICSImage(path)
    sizes = img.physical_pixel_sizes
    return sizes.X, sizes.Y
