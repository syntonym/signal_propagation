from multiprocessing import Process
from pathlib import Path
import importlib
import sys

from runnel.stage0 import run as stage0_run
import click

@click.command()
@click.argument('wheels', nargs=-1)
@click.option('--foreground', required=True)
@click.option('--foreground-arg', default=None)
@click.option('--background', default=None)
@click.option('--dark/--light', default=True)
@click.option('--pyodide-dir', default=None)
def run(wheels, background, foreground, foreground_arg, pyodide_dir, dark):
    assert background is None or len(background.split(':')) == 2
    assert len(foreground.split(":")) == 2

    if pyodide_dir:
        pyodide_dir = Path(pyodide_dir).absolute()
        assert pyodide_dir.exists() and pyodide_dir.is_dir()
        for file in ['pyodide.js', 'pyodide.asm.js', 'pyodide.asm.data', 'pyodide.asm.wasm']:
            if not(pyodide_dir / file).exists():
                print(f'incorrect pyodide directory, file "{file}" not present')
                sys.exit(1)

    for wheel in wheels:
        if not Path(wheel).exists():
            print(f'File "{wheel}" does not exist')
            sys.exit(1)

    background_callable = None
    if background:
        background_module, callable_name = background.split(':')
        module = importlib.import_module(background_module)
        background_callable = getattr(module, callable_name)

    stage0_thread = Process(target=stage0_run, args=[foreground, wheels, pyodide_dir, dark, foreground_arg])
    stage0_thread.start()

    if background_callable:
        background_process = Process(target=background_callable)
        background_process.start()
