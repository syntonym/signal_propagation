import numpy as np
import polars as pl
import scipy.ndimage
import numba
import heapq
import logging

def moving_average(a, n=3):
    if n == 0:
        return a
    ret = np.cumsum(a, axis=2, dtype=float)
    ret[:, :, n:] = ret[:, :, n:] - ret[:, :, :-n]
    return ret[:, :, n - 1:] / n


def normalize(data):
    m, mm = np.min(data), np.max(data)
    data = (data - m) / (mm - m)
    return data


def _calculate_bf_threshold_and_mask(data, min_size=30):

    calculation_needed = True
    pick = 1
    while calculation_needed and pick < 80:
        threshold = calculate_threshold(data, pick=pick)
        mask = calculate_mask(data, th=threshold, raw=True, larger=False)
        mask_size = mask.sum()
        if mask_size == 0:
            calculation_needed = False
        elif mask_size < min_size:
            pick += 1
        else:
            calculation_needed = False

    return mask


def _calculate_fluro_threshold_and_mask(data, organoid_mask):
    organoid = data[organoid_mask]
    bg = data[~organoid_mask]

    organoid_size = np.sum(organoid_mask)

    bg_mean = np.mean(bg)
    bg_std = np.std(bg)

    significance_threshold = bg_mean+3*bg_std
    significant_signal = np.sum(organoid > significance_threshold)

    # not enough significant signal
    if significant_signal  < 0.05 * organoid_size:
        return np.zeros(organoid_mask.shape, dtype=bool)

    calculation_needed = True
    pick = 1
    while calculation_needed and pick < 80:
        threshold = calsipro.analysis.calculate_threshold(data[organoid_mask], pick=pick)
        mask = calsipro.analysis.calculate_mask(data, th=threshold, raw=True)
        mask_size = mask.sum()
        if mask_size == 0:
            calculation_needed = False
        elif (mask_size < 0.05*organoid_size) or (threshold < significance_threshold):
            pick += 1
        else:
            calculation_needed = False
    mask[~organoid_mask] = False
    return mask



def calculate_mask(t, th=0.25, raw=False, labelling=True, larger=True):
    if not raw:
        t = np.max(t, axis=0)
        t = normalize(t)
    if larger:
        mask = t >= th
    else:
        mask = t <= th
    if not labelling:
        return mask
    image = np.ones(mask.shape)
    image[~mask] = 0
    image[mask] = 1
    label, count = scipy.ndimage.label(image)
    if count == 1:
        return mask
    else:
        sizes = []
        for k in range(1, count+1):
            size = np.sum(label[mask] == k)
            sizes.append(size)
        if len(sizes) > 0:
            biggest = np.argmax(sizes)+1
        else:
            biggest = 1
        return label == biggest


def calculate_histogram(data):
    N = 80
    zero_bins = 80
    while zero_bins > 4:
        freq, counts, bins = _calculate_histogram(data, N)
        zero_bins = np.sum(counts == 0)
        N = int(N/2)
    return freq, counts, bins


def _calculate_histogram(data, N):
    if data.dtype == np.uint8 or data.dtype == np.uint16:
        data = data.astype(np.int32)
    if np.min(data) == 0:
        offset = 1
    else:
        offset = 0
    try:
        counts, bins = np.histogram(np.log(data+offset), N)
    except Exception as e:
        d1 = data+offset
        d2 = np.log(d1)
        logging.error({'data+offset': d1, 'log(data+offset)': d2, 'offset': offset, 'data min': np.min(data), 'data max': np.max(data), 'data+offset min': np.min(d1), 'data+offset max': np.max(d1), 'log(data+offset min)': np.min(d2), 'log(data+offset max)': np.max(d2)})
        raise e

    for i in range(len(counts)):
        if counts[i] <= 10:
            counts[i] = 1
        else:
            break

    for i in range(1, len(counts)):
        if counts[-i] <= 10:
            counts[-i] = 1
        else:
            break
    freq = np.log(1+counts)
    return freq, counts, bins

def calculate_threshold(data, pick=1):

    if np.min(data) == 0:
        offset = 1
        logging.debug('Calculating threshold, min of data is 0, setting offset to 1.')
    else:
        offset = 0

    freq, counts, bins = calculate_histogram(data)
    left_flood = freq.copy()
    right_flood = freq.copy()
    flood = freq.copy()

    for i in range(1, len(freq)):
        left_flood[i] = max(left_flood[i], left_flood[i-1])

    for i in list(range(0, len(freq)-1))[::-1]:
        right_flood[i] = max(right_flood[i], right_flood[i+1])

    for i in range(0, len(freq)):
        flood[i] = min(left_flood[i], right_flood[i])

    f = flood - freq
    if pick != 1:
        idxs = np.argsort(flood-freq)
        idx = idxs[-pick]
    else:
        idx = np.argmax(flood-freq)
    logging.debug(f'Calculating threshold, index is {idx}')

    rest = freq[idx:]
    low = freq[idx]
    high = np.max(rest)

    idx_offset = max(0, np.argmax(rest >= (low + (high-low)*0.10))-1)
    idx = idx + idx_offset

    logging.debug(f'Calculating threshold, index was pushed to {idx}')


    pixels = np.sum(counts[idx:]) / np.sum(counts)
    if  pixels < 0.001:
        logging.debug('Calculating threshold, threshold is lower than 0.001 percentile, invalidating threshold')
        return np.max(data)+1
    if  0.999 < pixels:
        logging.debug('Calculating threshold, threshold is higher than 0.999 percentile, invalidating threshold')
        return np.max(data)+1
    if idx == 0:
        logging.debug('Calculating threshold, threshold is 0, invalidating threshold')
        return np.max(data)+1
    threshold = np.exp(bins[idx+1])-offset
    logging.debug('Calculating threshold, final threshold is %s', threshold)
    return threshold


def _find_biggest(mask):
    label, count = scipy.ndimage.label(mask)
    if count == 1:
        return mask
    else:
        sizes = list(np.bincount(label[mask]))
        assert sizes[0] == 0
        sizes = sizes[1:]
        assert len(sizes) == count
        if len(sizes) > 0:
            biggest = np.argmax(sizes)+1
        else:
            biggest = 1
        return label == biggest


def shrink_mask(mask, n):
    for i in range(n):
        d1 = np.diff(mask, axis=0)
        d2 = np.diff(mask, axis=1)
        mask[:-1, :][d1 != 0] = 0
        mask[1:, :][d1 != 0] = 0
        mask[:, 1:][d2 != 0] = 0
        mask[:, :-1][d2 != 0] = 0
    return mask


def time_analysis(t, intensity_cutoff=0.5):
    t = t - np.min(t, axis=0).reshape((1, t.shape[1], t.shape[2]))
    t = t / np.max(t, axis=0).reshape((1, t.shape[1], t.shape[2]))

    time = np.argmax(t >= intensity_cutoff, axis=0)
    return time


def push_low_pixels(time, mask):

    m = np.min(time[mask])
    mm = np.max(time[mask])

    time[~mask] = mm+1

    for i in range(m, mm+1):
        if np.sum(time == i) < 30:
            time[time == i] = i+1
            m = m+1
        else:
            break

    for i in range(m, mm+1)[::-1]:
        if np.sum(time == i) < 30:
            time[time == i] = i-1
            mm = mm-1
        else:
            break
    time[~mask] = np.max(time)
    return time


def find_times(data, mask, times, as_index=True):
    data = data.copy()
    data[~mask] = np.min(data)-1
    if as_index:
        return [(data == time).nonzero() for time in times]
    else:
        return [(data == time) for time in times]


def find_ori_cluster(data, mask, as_index=False):
    cluster_mask = np.zeros(data.shape, dtype=np.bool_)
    ori_time = np.min(data[mask])
    cluster_mask[data == ori_time] = 1
    cluster_mask[~mask] = 0
    label, count = scipy.ndimage.label(cluster_mask, scipy.ndimage.generate_binary_structure(2, 2))

    if count > 1:
        sizes = []
        for k in range(1, count+1):
            size = np.sum(label[mask] == k)
            sizes.append(size)
        biggest = np.argmax(sizes)+1
        cluster_mask = (label == biggest)

    if as_index:
        xs, ys = cluster_mask.nonzero()
        return np.array((np.average(xs), np.average(ys))).reshape((2, 1))
    else:
        return cluster_mask


def calculate_speed(time, mask):
    m, mm = np.min(time[mask]), np.max(time[mask])

    ori_pos = np.array(find_ori_cluster(time, mask, as_index=True)).reshape((2, 1))

    timepoints = list(range(m+1, mm+1))
    locations = [np.stack([x, y]) for x, y in find_times(time, mask, timepoints, as_index=True)]

    dts = [0]
    speeds = [-1]
    ns = [np.sum(time == m)]
    total_speed = 0
    total_ns = 0
    for t, l in zip(timepoints, locations):
        dists = np.sqrt(np.sum((l - ori_pos)**2, axis=0))
        dt = t-m
        dl = np.sum(dists)
        n = dists.shape[0]
        if n == 0:
            continue
        total_speed += dl/dt
        total_ns += n
        speeds.append(dl/(n*dt))
        ns.append(n)
        dts.append(dt)
    if total_ns == 0:
        total_ns = 1

    r = (pl.DataFrame({'time': np.array(dts, dtype=np.int64),
                       'speed': np.array(speeds, dtype=np.float64),
                       'n': np.array(ns, dtype=np.int64)}),
         total_speed/total_ns)
    return r


def calculate_speed_better(time, mask):
    ori_pos = find_ori_cluster(time, mask, as_index=True)

    x_dist = np.repeat((np.arange(time.shape[0]) - ori_pos[0]).reshape((time.shape[0], 1)), time.shape[1], axis=1)
    y_dist = np.repeat((np.arange(time.shape[1]) - ori_pos[1]).reshape((1, time.shape[1])), time.shape[0], axis=0)

    dists = np.sqrt(x_dist**2 - y_dist**2)

    time = time - np.min(time)
    speed = dists / time
    speed[time == 0] = 0

    return speed


def tabularize_speed(time, speed, mask):
    m, mm = np.min(time), np.max(time)
    timepoints = list(range(m+1, mm+1))

    time[mask] = mm+1

    ns = [np.sum(time == timepoint) for timepoint in timepoints]
    speed = [np.average(speed[time == timepoint]) for timepoint in timepoints]

    r = (pl.DataFrame({'time': np.array(timepoints, dtype=np.int64),
                       'speed': np.array(speed, dtype=np.float64),
                       'n': np.array(ns, dtype=np.int64)}),
         np.average(speed[mask]))
    return r


@numba.njit(cache=True)
def _reach_init_point(x, y, next, reach, visited, data, level):
    heapq.heappush(next, (0.0, x, y))
    reach[x, y] = 0.0
    level[x, y] = data[x, y]

@numba.njit(cache=True)
def _reach_init(xdim, ydim, next, reach, visited, data, level):

    for y in range(10):
        _reach_init_point(0, y, next, reach, visited, data, level)
        _reach_init_point(0, ydim-y-1, next, reach, visited, data, level)
        _reach_init_point(xdim-1, y, next, reach, visited, data, level)
        _reach_init_point(xdim-1, ydim-y-1, next, reach, visited, data, level)

    for x in range(10):
        _reach_init_point(x, 0, next, reach, visited, data, level)
        _reach_init_point(xdim-x-1, 0, next, reach, visited, data, level)
        _reach_init_point(x, ydim-1, next, reach, visited, data, level)
        _reach_init_point(xdim-x-1, ydim-1, next, reach, visited, data, level)


@numba.njit(cache=True)
def _reach_check(ox, oy, x, y, xdim, ydim, next, reach, data, visited, level):
    W1 = 100
    W2 = 1
    WT = W1+W2
    if 0 <= x < xdim and 0 <= y < ydim:
        r = abs(data[x,y] - level[ox, oy])
        r = max(r, reach[ox, oy])
        reach[x, y] = min(reach[x, y], r)
        level[x, y] = (W1*level[ox, oy] + W2*data[x, y]) / WT
        heapq.heappush(next, (r, x, y))

@numba.njit(cache=True)
def _reach_visit(ox, oy, xdim, ydim, next, reach, data, visited, level):
    if visited[ox, oy]:
        return
    visited[ox, oy] = True

    _reach_check(ox, oy, ox+1, oy, xdim, ydim, next, reach, data, visited, level)
    _reach_check(ox, oy, ox-1, oy, xdim, ydim, next, reach, data, visited, level)
    _reach_check(ox, oy, ox, oy+1, xdim, ydim, next, reach, data, visited, level)
    _reach_check(ox, oy, ox, oy-1, xdim, ydim, next, reach, data, visited, level)


@numba.njit(cache=True)
def _reachability(data, reach, visited, level):
    next = [(np.float64(0.0), 1, 1) for x in range(0)]
    xdim, ydim = data.shape
    _reach_init(xdim, ydim, next, reach, visited, data, level)

    while (len(next) != 0):
        r, x, y = heapq.heappop(next)
        _reach_visit(x, y, xdim, ydim, next, reach, data, visited, level)


def reachability(data):
    data = normalize(data)
    reach = np.ones(shape=data.shape, dtype=data.dtype)
    visited = np.zeros(shape=data.shape, dtype=np.bool_)
    level = np.zeros(shape=data.shape, dtype=data.dtype)
    _reachability(data, reach, visited, level)
    return reach


def calculate_bf_mask(data):
    data = data.copy()
    reach = reachability(data)
    th = calculate_threshold(reach)
    mask = calculate_mask(reach, th=th, raw=True)
    return mask
