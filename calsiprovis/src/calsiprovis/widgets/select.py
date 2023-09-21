from rapids.widgets import Widget, Div, Signal, Button
from rapids.js import create_node


class Select(Widget):

    dom_node = 'select'

    def __init__(self, options=None, format=None):
        super().__init__()
        if options is None:
            options = []
        if format is None:
            format = lambda x: x
        self._options = options
        self._format = format
        self.on_js('change', self._on_js_change)

    async def _on_js_change(self, signal):
        await self.emit(Signal('select', self, self._options[int(self.dom.value)]))

    def realise(self, dom):
        dom.textContent = ''

        for i, option in enumerate(self._options):
            dom_o = create_node('option')
            dom_o.setAttribute('value', i)
            dom_o.textContent = self._format(option)
            dom.appendChild(dom_o)
