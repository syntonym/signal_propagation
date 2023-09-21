from dataclasses import dataclass, field
from pathlib import Path
import json

def show_scenes(inp):
    from aicsimageio import AICSImage
    img = AICSImage(inp)
    return img.scenes


@dataclass
class Experiment:
    name: str
    path: str
    cache: str
    brightfield: dict = field(default_factory=dict)
    gcamp: dict = field(default_factory=dict)
    fluros: dict = field(default_factory=dict)
    fluro_order: list = field(default_factory=list)

    def _to_dict(self):
        return {'experiment': self.name, 'path': self.path, 'cache': self.cache, 'gcamp': self.gcamp, 'fluros': self.fluros, 'fluro_order': self.fluro_order, 'brightfield': self.brightfield}

    def serialize(self):
        return json.dumps(self._to_dict())

    @staticmethod
    def deserialize(s):
        metadata = json.loads(s)
        experiment = Experiment(metadata['experiment'], metadata['path'], metadata['cache'])
        experiment.gcamp = metadata['gcamp']
        experiment.fluros = metadata['fluros']
        experiment.fluro_order = metadata['fluro_order']
        experiment.brightfield = metadata.get('brightfield', {})
        return experiment

    def save(self):
        s = self.serialize()
        with Path(self.path).open(mode='w') as f:
            f.write(s)

    @staticmethod
    def load(path):
        with Path(path).open() as f:
            s = f.read()
        return Experiment.deserialize(s)

    @staticmethod
    def new(path):
        path = Path(path).absolute()
        name = path.name.rsplit('.', maxsplit=1)[0]
        root = path.parent
        cache_dir = root/ f'{name}_results'

        cache_dir.mkdir(exist_ok=True)
        experiment = Experiment(name, str(path), str(cache_dir))
        experiment.save()
        return experiment

    def add_gcamp(self, path):
        scenes = show_scenes(path)
        for scene in scenes:
            self.gcamp[scene] = path
        self.save()

    def add_brightfield(self, path):
        scenes = show_scenes(path)
        for scene in scenes:
            self.brightfield[scene] = path
        self.save()

    def add_fluro(self, path, fluro):
        scenes = show_scenes(path)
        for scene in scenes:
            d  = self.fluros.get(fluro, {})
            d[scene] = path
            self.fluros[fluro] = d
        if fluro not in self.fluro_order:
            self.fluro_order.append(fluro)
        self.save()

    def cache_path(self, scene, f):
        scene_path = Path(self.cache) / scene
        scene_path.mkdir(exist_ok=True)
        return  scene_path / f

