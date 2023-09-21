import json
import js
from rapids.js import create_node, create_proxy, log
from rapids.widgets import Div, Button
import uuid

import testapp.library

class BokehPlot(Div):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = str(uuid.uuid4())

    def set_json(self, j):
        jj = js.JSON.parse(j)
        js.Bokeh.embed.embed_item(jj, self.id)

    def build(self):
        dom = create_node('div')
        dom.id = self.id
        return dom


class Display(Div):

    def __init__(self, f):
        super().__init__()
        self.file_name = f
        self.button = Button(self.file_name)
        self.div = Div()
        self.button.on('click', self.on_click)

        self.add_widget(self.button)

    async def on_click(self, signal):
        log('finding size')
        size = await testapp.library.find_size('/home/syrn/' + self.file_name)
        log('setting string')
        self.div.text = str(size)
        log('removinb widget')
        self.remove_child(self.button)

        log('creating BokehPlot node')
        self.plot = w = BokehPlot()
        log('adding widget')
        self.add_widget(self.plot)

        log('creating figure')

        jp = await testapp.library.generate_plot()
        self.plot.set_json(jp)
        log('returning true')
        return True
