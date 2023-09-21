from rapids.widgets import Widget, Div, Signal, Button
import uuid
import js
from rapids.js import create_node

class BokehPlot(Div):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = str(uuid.uuid4())
        self.serialized_plot = None

    def set_json(self, j):
        self.serialized_plot = j
        if self.dom:
            jj = js.JSON.parse(self.serialized_plot)
            js.Bokeh.embed.embed_item(jj, self.id)

    def _dom_init(self, dom):
        dom.id = self.id
        if self.serialized_plot:
            jj = js.JSON.parse(self.serialized_plot)
            js.Bokeh.embed.embed_item(jj, self.id)
        return dom
