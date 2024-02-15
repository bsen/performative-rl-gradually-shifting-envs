import numpy as np

from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig


class MetaEnvironment:
    """
    An abstract class describing what a meta environment should implement.
    """

    def __init__(
        self,
        run_config: SingleRunConfig,
        general_config: GeneralConfig,
        random_generator: np.random.Generator,
    ):
        self._run_config: SingleRunConfig = run_config
        self._general_config: GeneralConfig = general_config
        self._random_generator: np.random.Generator = random_generator

    def change_function(self, policy, policy_index=0):
        """
        An implementation of the change functions from the paper.
        The probability transition and reward functions of the underlying
        environment change.

        Args:
            policy -- a policy which the environment adopts to
            num_trajectories -- the number of trajectories (can be set to -1,
                in which case  the exact probability transition and reward functions
                are returned.

        Returns:
            If the num_trajectories is larger than 0 :
                a list trajectories_over_t of num_trajectories many trajectories
                from the new environment. The list has shape
                num_trajectories x len_trajectory x 3, where
                trajectories_over_t[a,b,c] corresponds to trajectory a of the
                trajectories of this round, the b-th entry in this trajectory and
                if c==0 the state, c==1 the action and c==2 the reward of this entry.
            If the num_trajectories is equal to -1 :
                the current probability transition and reward functions of the
                underlying environment
        """
        raise NotImplementedError()

    def get_trajectories(self, num_trajectories: int):
        raise NotImplementedError()

    def reset(self, random_generator):
        self._random_generator = random_generator

    def is_terminal(self, state) -> bool:
        raise NotImplementedError()

    def num_states(self, policy_index) -> int:
        raise NotImplementedError()

    def num_actions(self, policy_index) -> int:
        raise NotImplementedError()

    def start_distribution(self, policy_index=0) -> np.ndarray:
        raise NotImplementedError()

    def num_policies(self) -> int:
        raise NotImplementedError()

    def get_random_generator(self):
        return self._random_generator

    def get_general_config(self):
        return self._general_config

    def get_run_config(self):
        return self._run_config
