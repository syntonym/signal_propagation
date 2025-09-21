import numpy as np
from bokeh.plotting import figure
from bokeh.palettes import inferno
import polars as pl
import click

import time
import os.path

from calsipro.organoid_database import Database
from calsipro.io import read_data, read_peaks, read_name_sheet, calculate_image_frames, write_video
from calsipro.analysis import calculate_mask, time_analysis, push_low_pixels, calculate_speed, normalize, call_peaks
from calsipro.visualisations import show_mapped_picture, show_mapped_distribution, show_mapped_mask_picture, show_speed_picture, show_traces, visualize_debug


def analyse_speed(inp, th=0.25, ct=0.5, filter=True, average=0):
    t = inp
    mask = calculate_mask(t, th)
    time = time_analysis(t, ct)
    time = push_low_pixels(time, mask)
    df, speed = calculate_speed(time, mask)
    return df, speed


def make_colorbar(m, mm, out, fps=1):
    from calsipro.bokeh_helpers import save
    p = figure(frame_height=(mm-m)*40, frame_width=40)
    c = inferno(mm-m)[::-1]
    for i in range(mm-m):
        p.rect(x=0, y=i/fps, width=1/fps, height=1/fps, color=c[i])
    save(p, out)


def analyse_organoid(peaks, d, th=0.25, ct=0.5):
    print(th, ct)
    speed_full = []
    speeds = []
    for i, frames_of_interest in enumerate(d):
        frames_of_interest = normalize(frames_of_interest)
        speed_df, speed_estimate = analyse_speed(frames_of_interest, th=th, ct=ct)
        speed_df['beat'] = np.array([i]*len(speed_df), dtype=np.int64)
        speed_df = speed_df.filter(pl.col('speed') >= 0.0)
        speed_full.append(speed_df)
        speeds.append(speed_estimate)
        print(i, speed_df, speed_estimate)

    sf = pl.concat(speed_full)
    print(sf)
    if len(sf) == 0:
        return (-1, -1)
    e = (sf['speed'] * sf['n']).sum() / sf['n'].sum()
    e2 = np.average(speeds)
    print(e, e2)
    return e, e2


def ana_organoid(f_idx, name, peaks_path, root, th, ct=0.5, average=0):
    start_time = time.time()
    print(f'Analysing file {name}')
    peaks = read_peaks(peaks_path, f'Sheet{f_idx+1}')
    if len(peaks) == 0:
        print('No peaks detected, skipping')
        print()
        return -1, -1
    print('Reading data')

    image_frames = calculate_image_frames(peaks)

    start_time_data = time.time()
    data = read_data(os.path.join(root, name), image_frames, average=average)
    end_time_data = time.time()
    time_data = round(end_time_data - start_time_data, 2)
    print(f'Read data in {time_data} seconds')
    print('Starting to analyse video')
    e_weighted, e_unweighted = analyse_organoid(peaks, data, th=th, ct=ct)
    end_time = time.time()
    time_all = round(end_time - start_time, 2)
    print(f'Analysis of file {name} done in {time_all}')
    print()
    return e_weighted, e_unweighted


def plot_all_dists(f_idx, name, peaks_path, root):
    print(name)
    peaks = read_peaks(peaks_path, f'Sheet{f_idx+1}')
    if len(peaks) == 0:
        return -1
    data = read_data(os.path.join(root, name), False)
    raise Exception('Not implemented - data format changed')

    r = root.replace('/', '_')
    if r.startswith('.'):
        r = r[1:]

    for i in range(len(peaks)):
        start_frame = peaks['Var1_1'][i]
        end_frame = peaks['Var1_2'][i]
        frames_of_interest = data[:, :, start_frame:end_frame]
        frames_of_interest = normalize(frames_of_interest)
        #show_mapped_distribution(frames_of_interest, np.max, out=f'distributions/{r}_{name}_beat_{i}_dist.svg', y_axis_type='log')
        #show_mapped_mask_picture(frames_of_interest, f"images/masks/{r}_{name}_beat_{i}_mask.tif", th=0.25)
        show_mapped_picture(frames_of_interest, f'images/max/{r}_{name}_beat_{i}_max.tif')


def analyse_single_data(frames_of_interest, name, th=0.25, ct=0.5):
    speed_df, speed_estimate = analyse_speed(frames_of_interest, th)
    speed_df.to_csv(f"{name}_speed_data.csv")
    print(speed_df)
    print(speed_estimate)

    show_mapped_picture(frames_of_interest, f'{name}_beat_max.tif')
    show_mapped_distribution(frames_of_interest, np.max, out=f'{name}_beat_dist.svg', y_axis_type='log')
    show_mapped_mask_picture(frames_of_interest, f"{name}_beat_mask.tif", th=th)

    show_traces(frames_of_interest, th=th, time_shift=False, intensity_cutoff=ct, normalize_individual=False, out=f'{name}_beat_traces_notimeshift_nonorm.png')
    show_traces(frames_of_interest, th=th, time_shift=False, intensity_cutoff=ct, normalize_individual=True, out=f'{name}_beat_traces_notimeshift_norm.png')
    show_traces(frames_of_interest, th=th, time_shift=True, intensity_cutoff=ct, normalize_individual=True, out=f'{name}_beat_traces_timeshift_norm.png')
    show_speed_picture(frames_of_interest, f'{name}_beat_speed.tif', th, ct, False)


def analyse_single(root, traces, peaks, name):
    trace_path = os.path.join(root, traces)
    peaks_path = os.path.join(root, peaks)

    files = read_name_sheet(trace_path)[0]

    f_idx, name = [(idx, f) for idx, f in enumerate(files) if f == name][0]

    peaks = read_peaks(peaks_path, f'Sheet{f_idx+1}')
    if len(peaks) == 0:
        return -1
    d = read_data(os.path.join(root, name), False)
    raise Exception('Not implemented - data format changed')

    for i in range(len(peaks)):
        start_frame = peaks['Var1_1'][i]
        end_frame = peaks['Var1_2'][i]
        frames_of_interest = d[:, :, start_frame:end_frame]
        frames_of_interest = normalize(frames_of_interest)
        speed_df, speed_estimate = analyse_speed(frames_of_interest, 0.25)
        print(speed_df)

        show_mapped_picture(frames_of_interest, f'{name}_beat_{i}_max.tif')
        show_mapped_distribution(frames_of_interest, np.max, out=f'{name}_beat_{i}_dist.svg', y_axis_type='log')
        th = 0.25
        ct = 0.5
        show_mapped_mask_picture(frames_of_interest, f"{name}_beat_{i}_mask.tif", th=th)

        show_traces(frames_of_interest, th=th, time_shift=False, intensity_cutoff=ct, normalize_individual=False, out=f'{name}_beat_{i}_traces_notimeshift_nonorm.png')
        show_traces(frames_of_interest, th=th, time_shift=False, intensity_cutoff=ct, normalize_individual=True, out=f'{name}_beat_{i}_traces_notimeshift_norm.png')
        show_traces(frames_of_interest, th=th, time_shift=True, intensity_cutoff=ct, normalize_individual=True, out=f'{name}_beat_{i}_traces_timeshift_norm.png')
        show_speed_picture(frames_of_interest, f'{name}_beat_{i}_speed.tif', th, 0.5, color=True, filter=False)


def analyse_folder(root, out, database, ct=0.5, average=0, skip_already_calculated=True):
    trace_path = os.path.join(root, 'alltraces.xlsx')
    peaks_path = os.path.join(root, 'Peakresults90.xlsx')

    files = read_name_sheet(trace_path)[0]

    tifs = [(idx, f) for idx, f in enumerate(files)]
    print(f'Found {len(tifs)} files for analyzing')

    def get_th(name):
        record = database.get(name)
        if record.new:
            raise Exception(f'No threshold for organoid {name}')
        return record.threshold

    def get_name(name):
        if name.endswith('.czi'):
            name = name[2:] + '_1.tif'
        return name

    for f_idx, name in tifs:
        if name is None:
            continue
        oname = get_name(name)
        th = get_th(oname)
        record = database.get(oname)
        if record.speed_estimate_weighted != -2.0 and skip_already_calculated:
            print(f"Skipping {record.organoid}, speed estimate already calculated as {record.speed_estimate_weighted}")
            continue
        speed_estimate_weighted, speed_estimate_unweighted = ana_organoid(f_idx, os.path.join('tifs', oname), peaks_path, root, th=th, ct=ct, average=average)
        record.speed_estimate_weighted = speed_estimate_weighted
        record.speed_estimate_unweighted = speed_estimate_unweighted
        database.save(record)


def analyse_folder_dists(root, traces, peaks):
    trace_path = os.path.join(root, traces)
    peaks_path = os.path.join(root, peaks)

    files = read_name_sheet(trace_path)[0]

    tifs = [(idx, f) for idx, f in enumerate(files) if f.endswith('.tif')]
    for f_idx, name in tifs:
        plot_all_dists(f_idx, name, peaks_path, root)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('out')
@click.argument('folders', nargs=-1)
@click.option('--ct', default=0.50)
@click.option('--threshold-file', default=None)
@click.option('--average', default=0)
@click.option('--skip-already-calculated/--calculate-all', default=True)
def analyse(out, folders, ct, average, threshold_file, skip_already_calculated):
    database = Database(threshold_file)
    for folder in folders:
        print(folder)
        analyse_folder(folder, out, database, ct=ct, average=average, skip_already_calculated=skip_already_calculated)
    #analyse_folder('./D6/Atria', 'alltraces.xlsx', 'Peakresults90.xlsx', 'D6_Atria_result.csv')

    pass


@cli.command()
@click.argument('folders', nargs=-1)
@click.option("--fix-zeros/--only-check", default=False)
@click.option('--threshold-file', default=None)
def check(folders, threshold_file, fix_zeros):
    database = Database(threshold_file)
    no_ths = []
    for folder in folders:
        root = folder
        trace_path = os.path.join(root, 'alltraces.xlsx')
        peaks_path = os.path.join(root, 'Peakresults90.xlsx')

        files = read_name_sheet(trace_path)[0]

        tifs = [(idx, f) for idx, f in enumerate(files)]

        def get_th(name):
            record = database.get(name)
            if record.new:
                raise Exception(f'No threshold for organoid {name}')
            return record.threshold

        def get_name(name):
            if name.endswith('.czi'):
                name = name[2:] + '_1.tif'
            return name

        for f_idx, name in tifs:
            try:
                peaks = read_peaks(peaks_path, f'Sheet{f_idx+1}', raise_error=False)
            except KeyError:
                peaks = []
                print(f"Could not get sheet for {folder} {name}")
            name = get_name(name)
            record = database.get(name)
            record.peaks = len(peaks)
            if record.new:
                record.organoid = name
                record.category = "/".join(folder.split("/")[-2:])
                no_ths.append(record)
                if len(peaks) == 0:
                    record.reason = "No Peaks"
            if fix_zeros:
                database.save(record)

    if len(no_ths) > 0:
        print(f"No Thresholds for {len(no_ths)} files:")
        for record in no_ths:
            print(record.peaks, record.category, record.organoid)


@cli.command()
@click.option('--database', default='database.parquet')
@click.option('--output', default='database.csv')
def export(database, output):
    df = pl.read_parquet(database)
    df.write_csv(output)


@cli.command()
@click.argument('out')
@click.argument('folders', nargs=-1)
@click.option('--th', default=0.25)
@click.option('--ct', default=0.50)
@click.option('--peak-index', default=0)
@click.option('--skip', default='')
@click.option('--average', default=0)
@click.option('--prefix', default=None)
@click.option('--calculate-prefix', default=True)
@click.option('--threshold-file', default=None)
@click.option('--skip-already-calculated/--calculate-all', default=True)
def single(out, folders, peak_index, skip, th, ct, average, prefix, calculate_prefix, threshold_file, skip_already_calculated):
    database = Database(threshold_file)

    for folder in folders:
        print(f'Processing folder {folder}.')

        if calculate_prefix:
            prefix = os.path.join(*folder.split('/')[-2:])
            print(f'Calculated prefix as {prefix}')

        peak_index = 0

        trace_path = os.path.join(folder, 'alltraces.xlsx')
        peaks_path = os.path.join(folder, 'Peakresults90.xlsx')

        files = read_name_sheet(trace_path)[0]

        tifs = [(idx, f) for idx, f in enumerate(files)]

        for f_idx, name in tifs:
            if name is None:
                continue

            print(f'Considering organoid {name} index {f_idx}')
            if name.endswith('.czi'):
                name = name[2:] + '_1.tif'

            record = database.get(name)

            peaks = read_peaks(peaks_path, f'Sheet{f_idx+1}')
            print(f'Organoid has {len(peaks)} many peaks')

            if len(peaks) == 0:
                record.useable = False
                record.peaks = 0
                record.reason = 'No Peaks'
                database.save(record)
                print('Organoid has no peaks, skipping')
                continue
            else:
                record.peaks = len(peaks)
                database.save(record)
                if peak_index >= len(peaks):
                    print(f'Chosen peak {peak_index} is higher than existing peaks, clamping')
                    peak_index = len(peaks)-1
            print(f'Analyzing organoid {name} index {f_idx}')

            th = record.threshold

            image_frames_global = calculate_image_frames(peaks)
            for peak, peak_index in zip(peaks, range(len(image_frames_global))):

                __cache = []
                frames_of_interest = None
                def get():
                    if len(__cache) == 0:
                        image_frames = [image_frames_global[peak_index]]
                        frames_of_interest = read_data(os.path.join(folder, 'tifs', name), image_frames, average=average)
                        frames_of_interest = frames_of_interest[0]
                        __cache.append(frames_of_interest)
                    return __cache[0]

                if prefix:
                    out_folder = os.path.join(out, prefix, f'{name}_peak_{peak_index}_ct_{ct}_th_{th}_average_{average}_')
                else:
                    out_folder = os.path.join(out, f'{name}_peak_{peak_index}_ct_{ct}_th_{th}_average_{average}_')

                os.makedirs(out_folder, exist_ok=True)

                #def analyse_single_data(frames_of_interest, name, th=0.25, ct=0.5):
                o = os.path.join(out_folder, f"{name}_speed_data.csv")
                if not os.path.exists(o) or not skip_already_calculated:
                    print('STEP speed analysis')
                    frames_of_interest = get()
                    speed_df, speed_estimate = analyse_speed(frames_of_interest, th)
                    speed_df.to_csv(o)
                    print(speed_df)
                    print(speed_estimate)

                o = os.path.join(out_folder, f'{name}_beat_max.tif')
                if not os.path.exists(o) or not skip_already_calculated:
                    print('STEP max projection')
                    frames_of_interest = get()
                    show_mapped_picture(frames_of_interest, o)

                o = os.path.join(out_folder, f'{name}_beat_dist.svg')
                if not os.path.exists(o):
                    print('STEP dists')
                    frames_of_interest = get()
                    show_mapped_distribution(frames_of_interest, np.max, out=o, th=th, y_axis_type='log')

                o = os.path.join(out_folder, f"{name}_beat_mask.tif")
                if not os.path.exists(o) or not skip_already_calculated:
                    print('STEP mask')
                    frames_of_interest = get()
                    show_mapped_mask_picture(frames_of_interest, o, th=th)

                o = os.path.join(out_folder, f'{name}_beat_traces_notimeshift_nonorm.png')
                if not os.path.exists(o+'.png') or not skip_already_calculated:
                    print('STEP traces')
                    frames_of_interest = get()
                    show_traces(frames_of_interest, th=th, time_shift=False, intensity_cutoff=ct, normalize_individual=False, out=o)

                o = os.path.join(out_folder, f'{name}_beat_traces_notimeshift_norm.png')
                if not os.path.exists(o+'.png') or not skip_already_calculated:
                    print('STEP traces 2')
                    frames_of_interest = get()
                    show_traces(frames_of_interest, th=th, time_shift=False, intensity_cutoff=ct, normalize_individual=True, out=o)

                o = os.path.join(out_folder, f'{name}_beat_traces_timeshift_norm.png')
                if not os.path.exists(o+'.png') or not skip_already_calculated:
                    print('STEP traces 3')
                    frames_of_interest = get()
                    show_traces(frames_of_interest, th=th, time_shift=True, intensity_cutoff=ct, normalize_individual=True, out=o)

                o = os.path.join(out_folder, f'{name}_beat_speed.tif')
                if not os.path.exists(o) or not skip_already_calculated:
                    print('STEP speed picture')
                    frames_of_interest = get()
                    show_speed_picture(frames_of_interest, o, th=th, ct=ct, show_ori=False, color=True)

@cli.command()
@click.argument('file')
@click.argument('scene')
def debug(file, scene):
    frames = read_data(file, scene)

    frame = frames[0]
    frame = np.sum(frame, axis=(0,1))
    peak_detected, debug1, debug2, peaks = call_peaks(frame, N=10, padding=0)

    visualize_debug('test.svg', frame, peak_detected, debug1, debug2, peaks)

@cli.command()
@click.argument("file")
@click.argument("scene")
@click.argument("output")
@click.option("--encoder", default="vp9", type=click.Choice(["vp9", "h264"]))
def video(file, scene, output, encoder):
    data = read_data(file, scene=scene)[0]
    m, mm = np.min(data), np.max(data)
    data = np.iinfo(np.uint8).max * ((data - m) / (mm-m))
    data = data.astype(np.uint8)
    video = write_video(data, width=300, height=300, encoder=encoder)

    with open(output, mode="wb") as f:
        f.write(video)

if __name__ == '__main__':
    cli()
