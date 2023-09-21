import websockets
import asyncio
import json
import importlib
import pickle
import multiprocessing.pool

def execute(f, *args, **kwargs):
    try:
        return (f(*args, **kwargs), False)
    except Exception as e:
        print(e)
        return (None, True)

class Server:

    def __init__(self, port=8888, runners=4):
        self.port = port
        self._id = 0
        self.ids = {}
        self.pool = multiprocessing.pool.Pool(runners)

    def process(self, msg):
        msg = pickle.loads(msg)
        if msg['type'] == 'heartbeat':
            return pickle.dumps({"type": "heartbeat"})
        elif msg['type'] == 'execute':
            module_name = msg['module']
            module = importlib.import_module(module_name)
            f = getattr(module, msg['function'])
            args = msg['args']
            kwargs = msg['kwargs']
            self.ids[self._id] = self.pool.apply_async(f, args, kwargs)
            old_id = self._id
            self._id += 1
            return pickle.dumps({"type": "wait", "duration": 0.1, "id": old_id})
        elif msg["type"] == "load":
            id = msg["id"]
            async_result = self.ids[id]
            if async_result.ready():
                result, error_occured = async_result.get(0), False
                return pickle.dumps({'type': 'result', 'result': result, 'error': error_occured}, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                return pickle.dumps({"type": "wait", "duration": 0.1, "id": id})
        else:
            return pickle.dumps({'type': 'error'}, protocol=pickle.HIGHEST_PROTOCOL)

    def run_sync(self):
        asyncio.run(self.run_async())

    async def run_async(self):
        async with websockets.serve(self.loop, "0.0.0.0", 8888):
            await asyncio.Future()  # run forever

    async def loop(self, websocket):
        while True:
            msg = await websocket.recv()
            return_msg = self.process(msg)
            await websocket.send(return_msg)

def run():
    server = Server()
    server.run_sync()
