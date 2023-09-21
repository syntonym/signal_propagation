# Runnel

Runnel is a microframework that runs a python application in a python WASM implementation.
A webserver serves a minimal `index.html`, which loads pyodide, micropip and all wheels specified on the command line.
Then the WASM python runs an entrypoint.
Optionally Runnel runs a python entrypoint in a background process.
