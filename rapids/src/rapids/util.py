import sys
if sys.platform == 'emscripten':
    import js.console
    print = js.console.log
else:
    print = print
