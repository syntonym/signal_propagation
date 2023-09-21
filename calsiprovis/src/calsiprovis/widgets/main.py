from rapids.widgets import Widget, Div, Signal, Button, Flexbox, TextInput, Button, Grid
import js
import uuid

from rapids.js import create_node
import asyncio

from calsiprovis import library
from calsiprovis.widgets.single_view_selector import SingleViewSelector
from calsiprovis.widgets.origin_widget import OriginWidget


class MainWidget(Widget):

    def __init__(self):
        super().__init__(id_children={'loading': Div(text='loading')}, data={'data_origins': None})

    def init(self):
        f = Flexbox(direction='column')
        self.set_widget('header', f)
        self.label = Div(text='')
        f.add_widget(self.label)

        add_data_container = Flexbox(direction='column')
        self.add_widget(add_data_container)

        self.input_gcamp = TextInput()
        b_add_gcamp = Button(label='Add Gcamp')

        add_data_container.add_widget(Div(text='Gcamp'))
        add_data_container.set_widget('input', self.input_gcamp)
        add_data_container.add_widget(b_add_gcamp)

        b_add_gcamp.on('click', self.add_gcamp)

        self.input_fluro = TextInput()
        self.input_fluro_type = TextInput()
        b_add_fluro = Button(label='Add Fluro')

        add_data_container.add_widget(Div(text='Fluro'))
        add_data_container.add_widget(self.input_fluro)
        add_data_container.add_widget(self.input_fluro_type)
        add_data_container.add_widget(b_add_fluro)

        b_add_fluro.on('click', self.add_fluro)


        self.fb = fb = Flexbox(direction='row')
        self.set_widget('flexbox', fb)

        b1 = Button(label='max')
        b2 = Button(label='mask')
        b3 = Button(label='video')
        b4 = Button(label='peaks')
        b5 = Button(label='peaks simple')
        b6 = Button(label='dist')
        b7 = Button(label='beat')

        b1.on('click', self.load_max)
        b2.on('click', self.load_mask)
        b3.on('click', self.load_video)
        b4.on('click', self.load_peaks)
        b5.on('click', self.load_peaks_simple)
        b6.on('click', self.load_intensity_distribution)
        b7.on('click', self.load_beats)

        fb.add_widget(b1)
        fb.add_widget(b2)
        fb.add_widget(b3)
        fb.add_widget(b4)
        fb.add_widget(b5)
        fb.add_widget(b6)
        fb.add_widget(b7)

        self.fluro_buttons = []

        self.set_widget('grid', Grid())

    @property
    def experiment(self):
        return self.data.get('experiment', None)

    @experiment.setter
    def experiment(self, value):
        self.label.text = value.name
        self.data['experiment'] = value
        self.remove_widget('loading')
        self.data_origins = library.load_data_origins(value)
        for b in self.fluro_buttons:
            self.fb.remove_widget(b)
        self.fluro_buttons = []

        for fluro in value.fluros:
            b = Button(label=fluro)
            self.fb.add_widget(b)
            self.fluro_buttons.append(b)
            b.on('click', self.load_fluro, fluro=fluro)

        for fluro in value.fluros:
            b = Button(label=f"{fluro} mask")
            self.fb.add_widget(b)
            self.fluro_buttons.append(b)
            b.on('click', self.load_fluro_mask, fluro=fluro)

        for fluro in value.fluros:
            b = Button(label=f"{fluro} dist")
            self.fb.add_widget(b)
            self.fluro_buttons.append(b)
            b.on('click', self.load_fluro_distribution, fluro=fluro)


    @property
    def data_origins(self):
        return self.data['data_origins']

    @data_origins.setter
    def data_origins(self, value):
        self.data['data_origins'] = value
        self._data_origin_widgets = []
        grid = self.id_children['grid']
        for child in grid.children:
            grid.remove_widget(child)
        self._data_origin_widgets = []

        for data_origin in self.data['data_origins']:
            ow = OriginWidget(self.experiment, data_origin)
            grid.add_widget(ow)
            self._data_origin_widgets.append(ow)

    async def add_gcamp(self, signal):
        path = self.input_gcamp.value
        self.experiment = await library.add_gcamp_to(self.experiment, path)
        self.data_origins = library.load_data_origins(self.experiment)

    async def add_fluro(self, signal):
        path = self.input_fluro.value
        fluro_type = self.input_fluro_type.value
        self.experiment = await library.add_fluro_to(self.experiment, fluro_type, path)
        self.data_origins = library.load_data_origins(self.experiment)

    async def load_max(self, signal):
        await asyncio.gather(*[ow.emit(Signal('load_max_projection', self, None)) for ow in self._data_origin_widgets])

    async def load_mask(self, signal):
        await asyncio.gather(*[ow.emit(Signal('load_mask', self, None)) for ow in self._data_origin_widgets])

    async def load_video(self, signal):
        await asyncio.gather(*[ow.emit(Signal('load_video', self, None)) for ow in self._data_origin_widgets])

    async def load_peaks(self, signal):
        await asyncio.gather(*[ow.emit(Signal('load_peaks', self, None)) for ow in self._data_origin_widgets])

    async def load_peaks_simple(self, signal):
        await asyncio.gather(*[ow.emit(Signal('load_peaks_simple', self, None)) for ow in self._data_origin_widgets])

    async def load_intensity_distribution(self, signal):
        await asyncio.gather(*[ow.emit(Signal('load_intensity_distribution', self, None)) for ow in self._data_origin_widgets])

    async def load_beats(self, signal):
        await asyncio.gather(*[ow.emit(Signal('load_beat', self, None)) for ow in self._data_origin_widgets])

    async def load_fluro(self, signal, fluro=None):
        await asyncio.gather(*[ow.emit(Signal('load_fluro', self, fluro)) for ow in self._data_origin_widgets])

    async def load_fluro_mask(self, signal, fluro=None):
        await asyncio.gather(*[ow.emit(Signal('load_fluro_mask', self, fluro)) for ow in self._data_origin_widgets])

    async def load_fluro_distribution(self, signal, fluro=None):
        await asyncio.gather(*[ow.emit(Signal('load_fluro_distribution', self, fluro)) for ow in self._data_origin_widgets])
