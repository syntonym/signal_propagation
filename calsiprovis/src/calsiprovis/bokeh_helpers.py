from bokeh.models import Plot
from bokeh.io import export_svg


def _make_backend_svg(p):
    if isinstance(p, Plot):
        p.output_backend = "svg"
    else:
        if hasattr(p, "children"):
            for child in p.children:
                _make_backend_svg(child)
        else:
            try:
                for child in p:
                    _make_backend_svg(child)
            except:
                pass


def save(p, output=None):
    """Save a bokeh plot as svg"""
    _make_backend_svg(p)
    if not output.endswith(".svg"):
        output += ".svg"
    export_svg(p, filename=output)
