# Rapids

RPC framework for WASM python <=> server python communication.
WASM python should start `rapids.socket.browser.Socket`, pass this to `rapids.backend.ClientServer` and set this client server as `rapids.backend.client_server`.
Then all function that are decorated with the `@rapids.backend.backend` decorator will not execute, put send a message through the websocket.
On the server side start the `rapids.socket.backend.Server` e.g. by calling `rapids.socket.backend.run()` to receive messages via websocket.
Functions are automatically imported and executed.

## Security

Currently no security is implemented, which means that any websocket connection to the python server backend can execute arbitrary code.
