from typing import Tuple

import numpy as np

from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig
from sperl.data.occupancy import OccupancyMeasure
from sperl.metaenvs.abstract_meta_env import MetaEnvironment
from sperl.metapolicies.algorithms.approximate_opt_problem import (
    ApproximateOptProblem,
)


class MetaPolicy:
    """
    An abstract class for meta policies like RepeatedRetraining, DelayedRR and
    WeightedDRR.
    """

    def __init__(
        self,
        run_config: SingleRunConfig,
        general_config: GeneralConfig,
        meta_env: MetaEnvironment,
        random_generator,
        policy_index,
    ):
        """
        This method initializes a meta-policy, by providing a setting
        (the setting defines things like the lamda and gamma parameters.)
        """
        self._run_config = run_config
        self._general_config = general_config

        self.opt_problem: ApproximateOptProblem
        self.n_steps = 0
        self.meta_env: MetaEnvironment = meta_env
        self.gamma = run_config.gamma
        self.exact_opt_problem = general_config.exact_optimization
        self.v = run_config.v
        self.random_generator = random_generator
        self.cur_occupancy = np.zeros(
            (
                meta_env.num_states(policy_index),
                meta_env.num_actions(policy_index),
            )
        )
        self.policy_index = policy_index
        self.num_states = meta_env.num_states(policy_index)
        self.num_actions = meta_env.num_actions(policy_index)
        self._current_occupancy: OccupancyMeasure = OccupancyMeasure(
            np.zeros(
                (
                    meta_env.num_states(policy_index),
                    meta_env.num_actions(policy_index),
                )
            )
        )
        self._reward_value: float

    @property
    def run_config(self):
        return self._run_config

    @property
    def general_config(self):
        return self._general_config

    def empirical_occupancy_n_reward(self, trajectories) -> Tuple[np.ndarray, float]:
        """
        Computes the empirical occupancy measures given a list of trajectories.
        Args:
            trajectories -- a list of size
                num_trajectories x len_trajectory x 3 , where
                trajectories[a,b,c]
                corresponds to trajectory a of the
                trajectories of this round, the b-th entry in this trajectory
                and if c==0 the state, c==1 the action and c==2 the reward
                of this entry.
        """
        occupancy = np.zeros(
            (
                self.meta_env.num_states(self.policy_index),
                self.meta_env.num_actions(self.policy_index),
            )
        )
        total_mass = 0.0
        total_reward = 0.0
        for trajectory in trajectories:
            last_t = len(trajectory) - 1
            for t, (state, action, reward) in enumerate(trajectory):
                if t == last_t and action is None:
                    continue
                gamma_factor = self.gamma**t
                occupancy[state, action] += gamma_factor
                total_mass += gamma_factor
                total_reward += gamma_factor * reward

        occupancy /= total_mass * (1.0 - self.gamma)
        total_reward /= len(trajectories)
        return occupancy, total_reward

    def reset(
        self, meta_env: MetaEnvironment, random_generator: np.random.Generator
    ) -> None:
        """
        This method resets the number of iterations of this policy and
        initializes the starting policy anew.
        It does not reset the optimization class or the training setting.
        """
        self.random_generator = random_generator
        self.meta_env = meta_env
        self.opt_problem = self.new_opt_problem()
        self._current_occupancy = self.uniform_occupancy()

    def random_policy(self):
        policy = self.random_generator.random((self.num_states, self.num_actions))
        policy = np.divide(
            policy, np.outer(np.sum(policy, axis=1), np.ones(self.num_actions))
        )
        return policy

    def uniform_policy(self):
        policy = np.ones((self.num_states, self.num_actions))
        policy = np.divide(policy, self.num_actions)
        return policy

    def uniform_occupancy(self):
        num_states = self.meta_env.num_states(self.policy_index)
        num_actions = self.meta_env.num_actions(self.policy_index)
        return OccupancyMeasure(
            np.ones((num_states, num_actions))
            / ((1 - self.gamma) * num_states * num_actions)
        )

    def meta_step(self, meta_observation, policy_idx):
        """
        Keyword arguments:
            meta_observation -- some representation of the state in which the
            meta-environment is. This could be for example a list of
            trajectories of the environment or the probability transition and
            reward functions of the environment.
        Returns:
            If self.exact_opt_problem is True:
                an tuple (status_upd, occupancy measure) ,
                where status_upd is True, if the policy was updated and False
                if it was not updated in this step.
                occupancy_measure is a occupancy-measure which parameterizes a
                the policy currently played by the learner and it is
                represented as a numpy array of dimension
                num_states x num_actions
            otherwise:
                a tuple (occupancy_measure, num_samples_opt, num_samples_occ)
                where num_samples_opt is the number of samples used in this step
                input in the optimization problem and num_samples_occ is the number
                of samples in this step used to calculate the occupancy measure.
        """
        raise NotImplementedError()

    def get_num_samples_opt(self):
        raise NotImplementedError()

    def get_num_samples_occ(self):
        raise NotImplementedError()

    @property
    def current_occupancy(self) -> OccupancyMeasure:
        return self._current_occupancy

    def new_opt_problem(self):
        return ApproximateOptProblem(self)

    def got_updated(self) -> bool:
        raise NotImplementedError()

    @property
    def current_reward_value(self) -> float:
        return self._reward_value
