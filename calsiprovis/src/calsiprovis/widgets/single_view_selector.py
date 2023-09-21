from rapids.widgets import Widget

from calsiprovis.widgets.select import Select
from calsiprovis.widgets.single_view import SingleView

class SingleViewSelector(Widget):

    def init(self):
        self.on('select', self.on_select)

    @property
    def data_origins(self):
        return self.data['data_origins']

    @data_origins.setter
    def data_origins(self, value):
        self.data['data_origins'] = value
        self.set_widget('select', Select(value, format=lambda x: x.scene))

    async def on_select(self, signal):
        data_origin = signal.value
        self.set_widget('sv', SingleView(data_origin))
