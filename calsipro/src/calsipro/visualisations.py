import numpy as np
from calsipro.analysis import normalize, calculate_mask, time_analysis, push_low_pixels, find_ori_cluster, find_times
from calsipro.io import save

import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image

import polars as pl
from bokeh.plotting import figure
from bokeh.io import show

from matplotlib import cm

from syn_bokeh_helpers import syn_save


def show_mapped_picture(t, out, f=np.max):
    assert isinstance(t, np.ndarray)
    vars = f(t, axis=(0))
    vars = normalize(vars)
    save(vars, out)


def show_mapped_distribution(t, f=np.max, axis=2, out=None, th=None, **kwargs):
    assert isinstance(t, np.ndarray)
    assert out

    vars = f(t, axis=(axis))
    vars = normalize(vars)

    p = figure(frame_width=300, frame_height=300, **kwargs)
    freq, bins = np.histogram(vars, 100)
    bins = [(x+y)/2 for x, y in zip(bins, bins[1:])]
    p.circle(bins, freq)
    if th is not None:
        p.line(x=[th, th], y=[1, np.max(freq)], color='red')
    syn_save(p, out)


def show_mapped_mask_picture(t, out, th=0.5, f=np.max):
    assert isinstance(t, np.ndarray)
    vars = f(t, axis=2)
    vars = normalize(vars)
    mask = calculate_mask(vars, th, raw=True)
    save(mask, out)


def show_traces(t, th=0.25, intensity_cutoff=0.5, time_shift=False, normalize_individual=True, out='f1.png'):
    assert isinstance(t, np.ndarray)
    mask = calculate_mask(t, th)
    t_masked = t[mask]

    if time_shift:
        shift = time_analysis(t, intensity_cutoff=intensity_cutoff)
        shift_masked = shift[mask]

    nan = np.array([np.nan])
    # we need to have NANs between each pixel
    pixels_as_list = [t_masked[i,:] for i in range(t_masked.shape[0])]
    if normalize_individual:
        pixels_as_list = [(p-np.min(p))/(np.max(p)-np.min(p)) for p in pixels_as_list]
    intensity = np.concatenate([x for pixel in pixels_as_list for x in [pixel, nan]])

    l = []
    for k in range(t_masked.shape[0]):
        pixel_time = np.arange(t_masked.shape[1])
        if time_shift:
            pixel_time = pixel_time - shift_masked[k]
        for a in [pixel_time, nan]:
             l.append(a)
    time = np.concatenate(l)

    cvs = ds.Canvas(plot_width=400, plot_height=400)
    agg = cvs.line(pl.DataFrame({'time': time, 'intensity':intensity}).to_pandas(), 'time', 'intensity', agg=ds.count())
    export_image(tf.shade(agg, how='linear'), out, background="white")


def show_speed_picture(t, out, th=0.25, ct=0.5, show_ori=False, color=True, filter=True):
    assert isinstance(t, np.ndarray)
    mask = calculate_mask(t, th)
    time = time_analysis(t, ct)
    time = push_low_pixels(time, mask)

    if show_ori:
        x, y = find_ori_cluster(time, mask, as_index=True)
        furthest_mask = find_times(time, mask, [np.max(time[mask])], as_index=False)[0]
        x, y = int(x), int(y)
        m = np.max(time)
        time[furthest_mask] = m * 1.5
        time[x,y] = m*2
        time = normalize(time)

    if color:
        time = normalize(time)
        time = 1-time
        time = normalize(time)
        time = cm.inferno(time)
        time[:,:,3][~mask] = 0

    save(time, out)


def visualize_debug(out, signal, peaks, debug, debug2, peaks_pairs):
    t = np.arange(len(signal))
    p = figure(frame_width=300, frame_height=300, x_axis_label="time", y_axis_label="Signal")
    p.line(t, signal)
    p.line(t, debug, color="green")
    p.line(t, debug2, color="green")
    p.circle(t[peaks], signal[peaks], color="red")

    if len(peaks_pairs) > 0:
        starts, ends = zip(*peaks_pairs)
        p.circle(starts, signal[np.array(starts)], color="green", size=10, fill_alpha=0.2)
        p.circle(ends, signal[np.array(ends)], color="red", size=10, fill_alpha=0.2)
    syn_save(p, out)


