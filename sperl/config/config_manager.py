import copy
import itertools
from typing import List

import numpy as np

from sperl.config.experiment import ExperimentConfig
from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig


class ConfigManager:
    def __init__(self, **kwargs):
        self.experiment = ExperimentConfig()
        self.general = GeneralConfig()

        for var_name, value in kwargs.items():
            if hasattr(self.experiment, var_name):
                setattr(self.experiment, var_name, value)
            else:
                setattr(self.general, var_name, value)

    def get_single_run_configs(self) -> List[SingleRunConfig]:
        attrs, experiment_values = self._attributes_n_values()
        run_confs: List[SingleRunConfig] = []
        for run_vals in experiment_values:
            run_conf = SingleRunConfig()
            for attr, val in zip(attrs, run_vals):
                setattr(run_conf, attr, val)
            run_confs.append(run_conf)
        self._add_seeds(run_confs)
        return run_confs

    def _add_seeds(self, single_run_configs: List[SingleRunConfig]):
        n_runs = len(single_run_configs)
        seed_sequence = np.random.SeedSequence(self.experiment.seed)
        seed_sequences = seed_sequence.spawn(n_runs)
        for i, single_run in enumerate(single_run_configs):
            single_run.seed_seq = seed_sequences[i]

    def _attributes_n_values(self):
        config_dict = copy.copy(self.experiment.__dict__)
        del config_dict["seed"]
        attributes = config_dict.keys()
        experiment_values = itertools.product(*config_dict.values())
        return attributes, experiment_values
