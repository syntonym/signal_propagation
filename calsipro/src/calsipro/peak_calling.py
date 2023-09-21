import numpy as np
import pywt


def _bootstrap_baseline(time, y, level_offset=2):
    # de-biasing with wavelet
    wav = 'bior2.2'

    max_level = pywt.dwt_max_level(len(time), wav)
    idx = max(max_level-level_offset, 1)
    coeffs = pywt.wavedec(y, wav, level=idx+1)

    r = coeffs[:1] + [np.zeros(len(c)) for c in coeffs[1:]]
    s = pywt.waverec(r, wav)
    if len(s) > len(y):
        s = s[:len(y)]
    if not len(s) == len(y):
        assert len(s) == len(y)
    return s


def _find_peak_edges(peak_detected):
    i = 0
    peaks = []
    while i < len(peak_detected):
        idx_start = np.argmax(peak_detected[i:])
        if (idx_start == 0) and not peak_detected[i]:
            break
        idx_end = np.argmin(peak_detected[i+idx_start:])
        if idx_end == 0:
            idx_end = len(peak_detected[i+idx_start:])
        peaks.append((i+idx_start, min(i+idx_start+idx_end, len(peak_detected))))
        i = i + idx_start+idx_end+1
    return peaks


def _join_close_peaks(time, signal, peak_detected, peaks, N=1):
    peak_detected = peak_detected.copy()
    for p1, p2 in zip(peaks, peaks[1:]):
        p1_start, p1_end = p1
        p2_start, p2_end = p2

        if p2_start - p1_end <= N:
            peak_detected[p1_end:p2_start] = True
    return peak_detected

def _push_peak_ends(time, signal, peak_detected):
    pd = peak_detected[0]
    for i in range(1, len(time)):
        if pd and not peak_detected[i]:
            if signal[i-1] > signal[i]:
                peak_detected[i] = True
        pd = peak_detected[i]
    return peak_detected


def _call_peaks(time, signal, baseline=None, peak_detected=None, sigma=3):

    # bootstrap

    if baseline is None:
        baseline = _bootstrap_baseline(time, signal)

    if peak_detected is None:
        peak_detected = bootstrap_peaks(time, signal, baseline)

    #peak_detected = _push_peak_ends(time, signal, peak_detected)
    peaks = _find_peak_edges(peak_detected)
    peak_detected = _join_close_peaks(time, signal, peak_detected, peaks)
    peaks = _find_peak_edges(peak_detected)

    is_baseline = ~peak_detected

    # refining
    s = signal
    b = np.zeros(len(time))
    bp = np.zeros(len(time))

    d = np.zeros(len(time))
    dp = np.zeros(len(time))

    v = np.var((signal-baseline)[is_baseline])

    sbp = 1/v
    sdp = 1/np.var(np.diff(signal-baseline)[is_baseline[:-1]])

    N = 10
    p = (N-1)/N

    peak_N = 100
    peak_p = (peak_N-1)/peak_N

    b[0] = baseline[0]
    d[0] = (baseline[-1] - baseline[0]) / len(baseline)
    if d[0] > 0:
        d[0] = 0

    bp[0] = 5*sbp
    dp[0] = 5*sdp

    in_peak = False
    peak_start_b = 0

    for i in range(1, len(time)-1):
        drift_estimate = signal[i+1] - signal[i]
        b_estimate = b[i-1]+d[i-1]

        signal_bigger_than_peak_start = in_peak and signal[i] >= b[i-1]+d[i-1]
        baseline_deviates_significant = (np.abs(signal[i]-b[i-1]+d[i-1]) > 3*np.sqrt(1/bp[i-1]))
        drift_deviates_significant = ((signal[i] > b[i-1]+d[i-1]) and ((np.abs(drift_estimate-d[i-1]) > 3*np.sqrt(1/dp[i-1]))))

        # looks like a peak
        if signal_bigger_than_peak_start or baseline_deviates_significant or drift_deviates_significant:
            if not in_peak:
                in_peak = True
                peak_start_b = signal[i]
            peak_detected[i] = True
            dp[i] = dp[i-1] * peak_p
            d[i] = d[i-1]
            bp[i] = bp[i-1] * peak_p
            b[i] = b[i-1]+d[i-1]
        else:
            in_peak = False
            peak_detected[i] = False
            dp[i] = dp[i-1]*p + sdp
            d[i] = min(0, (drift_estimate*sdp + d[i-1]*dp[i-1]*p) / dp[i])

            bp[i] = bp[i-1]*p + sbp
            b[i] = ((b[i-1] + d[i-1])*bp[i-1]*p + s[i] * sbp) / bp[i]

    # no drift estimage for last value possible
    i = -1
    b_estimate = b[i-1]+d[i-1]

    # looks like a peak
    if (np.abs(signal[i]-b[i-1]+d[i-1]) > 3*np.sqrt(1/bp[i-1])) or (drift_estimate-d[i-1] > 3*np.sqrt(1/dp[i-1])):
        peak_detected[i] = True
        dp[i] = dp[i] * p
        d[i] = d[i-1]
        bp[i] = bp[i-1]
        b[i] = b[i-1]+d[i-1]
    else:
        peak_detected[i] = False
        dp[i] = dp[i-1]*p + sdp
        wd = min(sdp, 1 / (drift_estimate - d[i-1]+sdp/100)**2)
        d[i] = min(0, (drift_estimate*wd + d[i-1]*dp[i-1]*p) / (dp[i-1]*p + wd))

        bp[i] = bp[i-1]*p + sbp
        wb = 1 / (s[i] - b[i-1])**2
        b[i] = (s[i]*wb + (b[i-1] + d[i-1]) * bp[i-1] ) / (wb+bp[i-1])

    #peak_detected = _push_peak_ends(time, signal-b, peak_detected)
    peaks = _find_peak_edges(peak_detected)
    peak_detected = _join_close_peaks(time, signal, peak_detected, peaks)

    return peak_detected, b, bp, d, dp


def call_peaks(time, signal, threshold=3):
    peak_detected, b, bpp, d, dp = _call_peaks(time, signal)
    peaks = _find_peak_edges(peak_detected)

    if len(peaks) == 0:
        return peaks, b

    average_time = int(np.mean([peak[1] - peak[0] for peak in peaks]))
    change = np.lib.stride_tricks.sliding_window_view(b, average_time)
    average_change = np.mean(np.abs(change[:, -1] - change[:, 0]))

    normalised_signal = signal - b
    std = np.std(normalised_signal[~peak_detected])
    peaks = [peak for peak in peaks if np.max(normalised_signal[peak[0]:peak[1]]) > average_change*0.25]
    peaks = [peak for peak in peaks if peak[1]-peak[0] < len(signal)*0.5]
    peaks = [peak for peak in peaks if peak[1] > average_time and peak[0] < len(signal)-average_time]
    peaks = [peak for peak in peaks if peak[1] <= len(signal)-4 and peak[0] >= 4]

    return peaks, b


def bootstrap_peaks(time, measurement, baseline, sigma=1.5):
    s = measurement - baseline
    std = np.std(s)
    return s / std >= sigma

def call_peaks_simple(time, signal, threshold=1.5):
    baseline = _bootstrap_baseline(time, signal, 3)
    s = np.std(baseline)
    peak_detected = (signal-baseline) > threshold*s
    peaks = _find_peak_edges(peak_detected)
    return peaks, baseline
