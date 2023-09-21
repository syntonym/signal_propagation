from js import WebSocket, console, ArrayBuffer
from pyodide.ffi import create_proxy
from asyncio import Queue, Event
import json

class Socket:

    def __init__(self, uri):
        self._socket = WebSocket.new(uri)
        self._is_open = Event()
        self._socket.binaryType = "arraybuffer"
        self._socket.addEventListener('open', create_proxy(self._on_open))
        self._socket.addEventListener('message', create_proxy(self._on_message))
        self._socket.addEventListener('error', create_proxy(self._on_error))
        self._incoming = Queue()

    async def send(self, msg):
        await self._is_open.wait()
        b = ArrayBuffer.new(len(msg))
        b.assign(msg)
        self._socket.send(b)

    async def recv(self):
        await self._is_open.wait()
        msg = await self._incoming.get()
        mv = msg.to_memoryview()
        return mv

    async def _on_open(self, event):
        self._is_open.set()

    async def _on_message(self, event):
        await self._incoming.put(event.data)

    async def _on_error(self, event):
        console.log(event)
