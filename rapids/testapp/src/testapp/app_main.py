import testapp.library
from js import document
from testapp.widgets import Div, Display
from pyodide.code import run_js

async def run():
    value = await testapp.library.find_files('/home/syrn')

    bokeh_scripts = await testapp.library.get_bokeh_script()

    for script in bokeh_scripts:
        run_js(script)

    root = document.getElementById('root')

    container = Div()

    for l in value:
        d = Display(l)
        container.add_widget(d)

    dom = container.render()
    root.appendChild(dom)

    #document.getElementById('root').innerHTML = str(return_msg)

