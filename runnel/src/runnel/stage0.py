from flask import Flask, send_from_directory, send_file, render_template
import importlib.resources
import importlib.metadata
from pathlib import Path
from zlib import adler32
import gunicorn.app.base


app = Flask(__name__)
app.config['DARKMODE'] = True


class WheelStore:

    def __init__(self):
        self.wheels = {}
        self.fg = None
        self.pyodide_dir = None
        self.fg_arg = None

    def put_app(self, fg):
        self.fg = fg

    def get_module(self):
        return self.fg.split(':')[0]

    def get_callable(self):
        return self.fg.replace(':', '.')

    def put_arg(self, arg):
        self.fg_arg = arg

    def get_arg(self):
        if self.fg_arg is None:
            return ''
        else:
            return f"'{self.fg_arg}'"

    def put_pyodide_dir(self, path):
        self.pyodide_dir = path

    def get_pyodide_dir(self):
        return self.pyodide_dir

    def put(self, path):
        path = Path(path)
        self.wheels[path.name] = path.absolute()

    def get(self, name):
        return self.wheels.get(name)

    def get_all_wheels(self):
        return self.wheels.keys()


wheel_store = WheelStore()


@app.route("/")
def index():
    return render_template('index.html', wheels=wheel_store.get_all_wheels(), module=wheel_store.get_module(), callable=wheel_store.get_callable(), arg=wheel_store.get_arg(), dark=app.config['DARKMODE'])


@app.route("/pyodide/<path:path>")
def serve_pyodide(path):
    pyodide_dir = wheel_store.get_pyodide_dir()
    if pyodide_dir:
        return send_from_directory(pyodide_dir, path)
    else:
        dir = importlib.resources.files('runnel') / 'pyodide'
        if path in [f.name for f in dir.iterdir()]:
            f = (dir/path).open(mode='rb')

            # calculate e-tag to enable HTTP caching
            check = adler32(f'runnel/pyodide/{path}'.encode("utf-8")) & 0xFFFFFFFF
            version = importlib.metadata.version('runnel')
            etag = f"{version}-{check}"
            return send_file(f, download_name=path, etag=etag)
        else:
            return 404


@app.route("/wheels/<path:path>")
def serve_wheel(path):
    wheel_path = wheel_store.get(path)
    if wheel_path:
        return send_file(wheel_path)
    else:
        return 404


class StandaloneApplication(gunicorn.app.base.BaseApplication):

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def run(foreground, wheels, pyodide_dir=None, dark=True, arg=None):
    wheel_store.put_app(foreground)
    if arg:
        wheel_store.put_arg(arg)
    wheel_store.put_pyodide_dir(pyodide_dir)
    for wheel in wheels:
        wheel_store.put(wheel)

    options = {
        'bind': '0.0.0.0:8088',
        'workers': 1,
    }

    app.config['DARKMODE'] = dark

    standalone_app = StandaloneApplication(app, options)
    standalone_app.run()
