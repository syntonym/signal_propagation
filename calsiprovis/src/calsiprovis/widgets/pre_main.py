from rapids.widgets import Widget, Div, Signal, Button, Flexbox, TextInput, Button, StyleAttribute
from rapids.util import print
import js
import uuid

from rapids.js import create_node
import asyncio

from calsiprovis import library
from calsiprovis.widgets.main import MainWidget
from calsiprovis.widgets.single_view_selector import SingleViewSelector

class Title(Div):
    font_size = StyleAttribute('font-size')
    font_style = StyleAttribute('font-style')

class PreMain(Widget):

    def init(self):
        self.experiment_ref_path = None
        self.experiments = []
        self.fb = fb = Flexbox(direction='column')
        self.margin_bottom = '40px'

        self.set_widget('selector', fb)

        title = Title(text='Calsiprovis')
        title.font_size = '2em'
        title.font_style = 'bold'
        fb.set_widget('title', title)

        self.set_widget('current', Div(text='Loading known Experiments'))


    def trigger_experiments(self):
        self.set_widget('current', Div(text=''))
        self.exp_holder = exp = Flexbox(direction='row', wrap='wrap')

        for experiment in self.experiments:
            b = Button(label=experiment.name)
            b.on('click', self.load_data, experiment=experiment)
            self.exp_holder.add_widget(b)

        fb = self.fb
        fb.set_widget('label', Div(text='Add Experiment'))
        fb.set_widget('input_name', TextInput())
        fb.set_widget('input_experiment', TextInput())
        b1 = Button(label='Add Experiment')
        fb.set_widget('b1', b1)
        b1.on('click', self.add_experiment)
        fb.set_widget('exp', exp)


    async def add_experiment(self, signal):
        if self.experiment_ref_path is None:
            return

        experiment_name = self.fb.id_children['input_name'].value
        experiment_path = self.fb.id_children['input_experiment'].value
        experiment = library.ExperimentRef(experiment_name, experiment_path)

        self.experiments.append(experiment)

        b = Button(label=experiment.name)
        b.on('click', self.load_data, experiment=experiment)
        self.exp_holder.add_widget(b)

        await library.add_experiment_ref(self.experiment_ref_path, experiment)


    async def load_data(self, signal, experiment):
        self.set_widget('selector', Div())
        self.set_widget('current', Div(text=f"Loading {experiment.name}"))
        if experiment != None:
            experiment = await library.get_experiment(experiment.path)
            main = MainWidget()
            self.set_widget('selector', self.fb)
            main.experiment = experiment
            self.set_widget('current', main)
        else:
            self.set_widget('current', Div(text=f"Error"))
