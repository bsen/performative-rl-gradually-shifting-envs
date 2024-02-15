import numpy as np

from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig


class Agent2AlgoBase:
    def __init__(self, run_config: SingleRunConfig, general_config: GeneralConfig):
        self._run_config = run_config
        self._general_cofig = general_config

    def step(
        self,
        agent1_policy,
        agent2_policy: np.ndarray,
        agent2_env,
    ) -> np.ndarray:
        raise NotImplementedError()
