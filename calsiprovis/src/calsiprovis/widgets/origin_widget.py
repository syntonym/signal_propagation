from rapids.widgets import Widget, Div, Signal, Button, Raw, StyleAttribute
from calsiprovis import library
from calsiprovis.widgets.bokeh_plot import BokehPlot
from rapids.util import print

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

class OriginWidget(Div):

    row = StyleAttribute('grid-row')
    col = StyleAttribute('grid-column')

    def __init__(self, experiment, data_origin):
        self.data_origin = data_origin
        self.path = data_origin.path
        self.scene = data_origin.scene
        self.experiment = experiment

        self.f1 = data_origin.f1
        self.f2 = data_origin.f2
        self.f3 = data_origin.f3
        self.width = 100
        super().__init__()
        self.set_widget('current', Div(text=self.scene))

        self._plot_max_projection = None
        self._plot_intensity_distribution = None
        self._plot_peaks = None
        self._plot_peaks_simple = None
        self._video = None
        self._mask = None
        self._fluros = {}
        self._fluro_distributions = {}
        self._fluro_masks = {}
        self._beat = None
        self._brightfield = None
        self._brightfield_distribution = None

    def init(self):
        row = self.scene.split('-')[1][0]
        col = self.scene.split('-')[1][1:]
        if row in ALPHABET:
            self.row = str(ALPHABET.index(row)+1)
        self.col = col
        self.on('load_max_projection', self.on_load_max_projection)
        self.on('load_mask', self.on_load_mask)
        self.on('load_video', self.on_load_video)
        self.on('load_peaks', self.on_load_peaks)
        self.on('load_peaks_simple', self.on_load_peaks_simple)
        self.on('load_intensity_distribution', self.on_load_intensity_distribution)
        self.on('load_fluro', self.on_load_fluro)
        self.on('load_fluro_distribution', self.on_load_fluro_distribution)
        self.on('load_fluro_mask', self.on_load_fluro_mask)
        self.on('load_beat', self.on_load_beat)
        self.add_widget(Div(text=row+col))

    async def on_load_intensity_distribution(self, signal):
        if not self._plot_intensity_distribution:
            self.set_widget('current', Div(text='Loading plot'))
            serialized_figure = await library.load_intensity_distribution(self.experiment, self.scene)
            self._plot_intensity_distribution = BokehPlot()
            self._plot_intensity_distribution.set_json(serialized_figure)
        self.set_widget('current', self._plot_intensity_distribution)
        return True

    async def on_load_max_projection(self, signal):
        if not self._plot_max_projection:
            self.set_widget('current', Div(text='Loading max'))
            max_projection = await library.load_max_projection(self.experiment, self.scene)
            self._plot_max_projection = Raw(html=max_projection)
        self.set_widget('current', self._plot_max_projection)
        return True

    async def on_load_brightfield_distribution(self, signal):
        if not self._brightfield_distribution:
            self.set_widget('current', Div(text='Loading plot'))
            serialized_figure = await library.load_brightfield_distribution(self.experiment, self.scene)
            self._brightfield_distribution = BokehPlot()
            self._brightfield_distribution.set_json(serialized_figure)
        self.set_widget('current', self._brightfield_distribution)
        return True

    async def on_load_brightfield(self, signal):
        fluro = signal.value
        if not self._brightfield:
            self.set_widget('current', Div(text='Loading'))
            div_mask = await library.load_brightfield(self.experiment, self.scene, width=int(self.width), height=int(self.width))
            self._fluros[fluro] = Raw(html=div_mask)
        self.set_widget('current', self._brightfield)
        return True

    async def on_load_mask(self, signal):
        if not self._mask:
            self.set_widget('current', Div(text='Loading'))
            div_mask = await library.load_mask(self.experiment, self.scene, width=int(self.width), height=int(self.width))
            self._mask = Raw(html=div_mask)
        self.set_widget('current', self._mask)
        return True

    async def on_load_video(self, signal):
        if not self._video:
            self.set_widget('current', Div(text='Loading Video'))
            div_video = await library.load_video(self.experiment, self.scene, width=int(self.width), height=int(self.width))
            self._video = Raw(html=div_video)
        self.set_widget('current', self._video)
        return True

    async def on_load_peaks(self, signal):
        if (not self._plot_peaks):
            self.set_widget('current', Div(text=f'Loading Peakcalling'))
            serialized_figure = await library.load_peaks(self.experiment, self.scene)
            self._plot_peaks = BokehPlot()
            self._plot_peaks.set_json(serialized_figure)
        self.set_widget('current', self._plot_peaks)
        return True


    async def on_load_peaks_simple(self, signal):
        if (not self._plot_peaks_simple):
            self.set_widget('current', Div(text=f'Loading Peakcalling'))
            serialized_figure = await library.load_peaks_simple(self.experiment, self.scene)
            self._plot_peaks_simple = BokehPlot()
            self._plot_peaks_simple.set_json(serialized_figure)
        self.set_widget('current', self._plot_peaks_simple)
        return True


    async def on_load_fluro_distribution(self, signal):
        fluro = signal.value
        if not self._fluro_distributions.get(fluro, None):
            self.set_widget('current', Div(text='Loading plot'))
            serialized_figure = await library.load_fluro_distribution(self.experiment, fluro, self.scene)
            self._fluro_distributions[fluro] = BokehPlot()
            self._fluro_distributions[fluro].set_json(serialized_figure)
        self.set_widget('current', self._fluro_distributions[fluro])
        return True

    async def on_load_fluro(self, signal):
        fluro = signal.value
        if not self._fluros.get(fluro, None):
            self.set_widget('current', Div(text='Loading'))
            div_mask = await library.load_fluro(self.experiment, fluro, self.scene, width=int(self.width), height=int(self.width))
            self._fluros[fluro] = Raw(html=div_mask)
        self.set_widget('current', self._fluros[fluro])
        return True

    async def on_load_fluro_mask(self, signal):
        fluro = signal.value
        if not self._fluro_masks.get(fluro, None):
            self.set_widget('current', Div(text='Loading'))
            div_mask = await library.load_fluro_mask(self.experiment, fluro, self.scene, width=int(self.width), height=int(self.width))
            self._fluro_masks[fluro] = Raw(html=div_mask)
        self.set_widget('current', self._fluro_masks[fluro])
        return True

    async def on_load_beat(self, signal):
        if self._beat is None:
            self.set_widget('current', Div(text='Loading'))
            n = await library.load_number_of_peaks(self.experiment, self.scene)
            if n == 0:
                self._beat = Div(text='No Peaks')
            elif n == 1:
                beat_image = await library.speed_per_peak(self.experiment, self.scene, 0)
                self._beat = Raw(html=beat_image)
            else:
                beat_image = await library.speed_per_peak(self.experiment, self.scene, 1)
                self._beat = Raw(html=beat_image)
        self.set_widget('current', self._beat)
        return True

