from PIL import Image
import openpyxl
import numpy as np
import polars as pl
from calsipro.analysis import moving_average
from aicsimageio import AICSImage
import ffmpeg
import select


def show_scenes(inp):
    img = AICSImage(inp)
    return img.scenes

def read_data(inp, scene=None, requested_frames=None, average=0):
    img = AICSImage(inp)

    if scene is None:
        scene = img.scenes[0]

    img.set_scene(scene)

    if requested_frames is None:
        frames = img.dims['T'][0]
        requested_frames = [(0, frames-average-1)]

    data = img.data
    images = []
    for frame_request in requested_frames:
        start, stop = frame_request
        if start < average:
            start_cut = average-start
        else:
            start_cut = average
        start, stop = start-start_cut, stop+average

        start = max(0, start)
        stop = max(min(frames-1, stop), start+1)

        image = data[start:stop, 0, 0, :, :]

        #image = np.moveaxis(image, [0], [2])
        #image = moving_average(image, n=average)
        image = image[start_cut:stop-start-average, :, :]
        images.append(image)
    return images


def save(img, out):

    # 3 Dimensional data, so an image with channels
    if len(img.shape) == 3:
        # four channels, must be RGBA
        if img.shape[2] == 4:
            mode = 'RGBA'
            img = ((2**8)-1) * img
            img = img.astype('>u1')
        # three channels must be RGB
        elif img.shape[2] == 3:
            mode = 'RGB'
            img = ((2**8)-1) * img
            img = img.astype('>u1')
        else:
            raise Exception('Cannot deduce mode')
    # 2 Dimensional data, so graytone image
    else:
        mode = 'I;16B'
        img = ((2**16)-1) * img
        img = img.astype('>u2')
    img = Image.fromarray(img, mode=mode)
    img.save(out)


def read_name_sheet(path):
    wb = openpyxl.load_workbook(path)
    s = wb['Sheet2']
    v = list(s.values)
    v = list(zip(*v))
    return v


def read_peaks(path, sheet, raise_error=False):
    wb = openpyxl.load_workbook(path)
    try:
        s = wb[sheet]
    except KeyError:
        print(f'Cannot read sheet. Is something wrong? There are only {len(wb.worksheets)} many sheets, asking for {sheet}')
        if raise_error:
            raise KeyError("Cannot read sheet")
        return pl.DataFrame([[], []], columns=['a', 'b'])
    v = list(s.values)
    v = list(zip(*v))
    names = [a[0] for a in v]
    data = [a[1:] for a in v]
    df = pl.DataFrame(data, columns=names)
    return df

def calculate_image_frames(peaks):
    frames = []
    for i in range(len(peaks)):
        start_frame = peaks['Var1_1'][i]
        end_frame = peaks['Var1_2'][i]
        frames.append((start_frame, end_frame))
    return frames

def write_video(data, width=100, height=100, fps=5, encoder="h264"):


    if encoder == "vp9":
        p = (ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='gray', s='{}x{}'.format(data.shape[-2], data.shape[-1]), r=fps)
        .output("pipe:1", format="webm", s=f"{width}x{height}", pix_fmt="yuv420p", deadline="realtime", crf="30", **{"row-mt": "1", "b:v": "2000k", "tile-columns": 3, "c:v": "vp9"})
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True))
    elif encoder == "h264":
        p = (ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='gray', s='{}x{}'.format(data.shape[-2], data.shape[-1]), r=fps)
        .output("pipe:1", format="mp4", movflags="frag_keyframe+empty_moov", s=f"{width}x{height}", pix_fmt="yuv420p", preset="fast", tune="zerolatency", crf="22", **{"c:v": "libx264"})
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True))
    else:
        raise ValueError(f"Unknown encoder {encoder}, only supported are 'vp9' and 'h264'")

    b = data.tobytes()
    outs, errs = p.communicate(input=b)

    if len(outs) == 0:
        raise Exception("Video was not generated correctly\n\n"+errs.decode("utf8"))

    return outs

