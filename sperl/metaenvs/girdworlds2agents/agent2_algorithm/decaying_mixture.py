import numpy as np
from scipy.special import softmax

from sperl.metaenvs.girdworlds2agents.agent2_algorithm.base import Agent2AlgoBase


class DecayingMixture(Agent2AlgoBase):
    NAME = "decaying-mixture"

    def step(
        self,
        agent1_policy,
        old_agent2_policy: np.ndarray,
        agent2_env,
    ) -> np.ndarray:
        weight = self._run_config.agent2_algorithm[1]
        state_action_value_agent2 = agent2_env.state_action_value(agent1_policy)
        softmax_values_agent2 = softmax(
            state_action_value_agent2 * self._run_config.beta, axis=1
        )
        return (1 - weight) * old_agent2_policy + weight * softmax_values_agent2
