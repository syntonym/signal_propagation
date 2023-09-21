from rapids.widgets import Widget, Div, Signal, Button, Flexbox, TextInput, Button, Range
import js
import uuid

from rapids.js import create_node
import asyncio

from calsiprovis import library
from calsiprovis.widgets.origin_widget import OriginWidget


class SingleView(Widget):

    def __init__(self, data_origin):
        self._data_origin = data_origin
        super().__init__()

    def init(self):

        fb = Flexbox()
        self.set_widget('flexbox', fb)

        b1 = Button(label='max')
        b2 = Button(label='mask')
        b3 = Button(label='video')
        b4 = Button(label='peaks')
        b5 = Button(label='dist')

        b1.on('click', self.load_max)
        b2.on('click', self.load_mask)
        b3.on('click', self.load_video)
        b4.on('click', self.load_peaks)
        b5.on('click', self.load_intensity_distribution)

        fb.add_widget(b1)
        fb.add_widget(b3)
        fb.add_widget(b4)
        fb.add_widget(b5)

        range = Range(100, 1000, step=50)
        range.on('change', self.on_range_change)
        fb.add_widget(range)

        fb_ow = Flexbox()
        self.set_widget('ow', fb_ow)
        fb_ow.set_style(**{'justify-content': 'center', 'align-content': 'center'})
        fb_ow.set_widget('ow', OriginWidget(self._data_origin))

    async def on_range_change(self, signal):
        ow = self.id_children['ow'].id_children['ow']
        ow.width = signal.value

    async def load_max(self, signal):
        ow = self.id_children['ow'].id_children['ow']
        await ow.emit(Signal('load_max_projection', self, None))

    async def load_mask(self, signal):
        ow = self.id_children['ow'].id_children['ow']
        await ow.emit(Signal('load_mask_projection', self, None))

    async def load_video(self, signal):
        ow = self.id_children['ow'].id_children['ow']
        await ow.emit(Signal('load_video', self, None))

    async def load_peaks(self, signal):
        ow = self.id_children['ow'].id_children['ow']
        await ow.emit(Signal('load_peaks', self, None))

    async def load_intensity_distribution(self, signal):
        ow = self.id_children['ow'].id_children['ow']
        await ow.emit(Signal('load_intensity_distribution', self, None))
