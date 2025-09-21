import js
from js import document
from pyodide.code import run_js
from rapids.socket.browser import Socket
from rapids.js import create_node, create_proxy
from rapids.widgets import Div, Button
import rapids.backend

import micropip


async def run(*args):

    socket = Socket('ws://localhost:8888')
    client_server = rapids.backend.ClientServer(socket)
    rapids.backend.client_server = client_server

    import calsiprovis.app
    await calsiprovis.app.run(*args)



