import pickle
import sys
import asyncio

MODE_BACKEND = 'backend'
MODE_CLIENT = 'client'

if sys.platform == 'emscripten':
    mode = MODE_CLIENT
    is_backend = False
    import js.console
    log = js.console.log
else:
    mode = MODE_BACKEND
    is_backend = True
    log = print

backend_server = None
client_server = None

class ClientServer:

    def __init__(self, socket):
        self.socket = socket
        self.lock = asyncio.Lock()

    async def request(self, f, args, kwargs):
        assert self.socket
        msg = {'type': 'execute', 'module': f.__module__, 'function': f.__name__, 'args': args, 'kwargs': kwargs}
        msg_serialized = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        async with self.lock:
            await self.socket.send(msg_serialized)
            serialized_return = await self.socket.recv()
        return_msg = pickle.loads(serialized_return)
        if return_msg["type"] == "result":
            return False, return_msg["result"]
        else:
            return True, return_msg["id"]

    async def wait(self, id):
        msg = {'type': 'load', "id": id}
        msg_serialized = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)

        while True:
            async with self.lock:
                await self.socket.send(msg_serialized)
                serialized_return = await self.socket.recv()
            return_msg = pickle.loads(serialized_return)
            if return_msg["type"] == "result":
                return return_msg["result"]
            elif return_msg["type"] == "wait":
                await asyncio.sleep(return_msg["duration"])




def backend(f):
    if mode == MODE_BACKEND:
        return f
    elif mode == MODE_CLIENT:
        async def wrapper(*args, **kwargs):
            global client_server
            if client_server is None:
                raise Exception('Client server not initiated')
            needs_wait, id_or_result = await client_server.request(f, args, kwargs)
            if needs_wait:
                id = id_or_result
                result = await client_server.wait(id)
            else:
                result = id_or_result
            return result
        return wrapper

