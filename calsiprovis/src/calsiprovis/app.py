from calsiprovis import library 
from calsiprovis.widgets.pre_main import PreMain
from pyodide.code import run_js
from js import document, console
from rapids.socket.browser import Socket
import rapids.backend
import asyncio
import time

background_tasks = []

async def load_bokeh():
    start_time = time.time()
    bokeh_scripts = await library.get_bokeh_script()
    end_time = time.time()
    console.log(end_time - start_time)
    for script in bokeh_scripts:
        run_js(script)



async def run(experiment_ref_path):
    socket = Socket('ws://localhost:8888')
    client_server = rapids.backend.ClientServer(socket)
    rapids.backend.client_server = client_server

    console.log(client_server)

    bokeh_loaded = asyncio.create_task(load_bokeh())
    background_tasks.append(bokeh_loaded)

    root = document.getElementById('root')
    main = PreMain()
    main_dom = main._dom_render()
    root.appendChild(main_dom)

    main.experiment_ref_path = experiment_ref_path
    main.experiments = await library.load_experiment_refs(main.experiment_ref_path)
    main.trigger_experiments()

