import sys

from rapids.backend import backend

if sys.platform != 'emscripten':
    from pathlib import Path
    from bokeh.plotting import figure
    from bokeh.embed import json_item

    from bokeh.resources import INLINE
    import json

@backend
def find_size(path):
    path = Path(path)
    return path.stat().st_size

@backend
def find_files(path):
    path = Path(path)
    names = [f.name for f in path.iterdir()]
    return names

@backend
def generate_plot():
    p = figure()
    p.line(x=[0, 1], y=[1, 0])
    jp = json_item(p)
    j = json.dumps(jp)
    return j

@backend
def get_bokeh_script():
    return INLINE.js_raw
