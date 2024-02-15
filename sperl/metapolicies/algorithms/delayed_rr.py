import numpy as np

from sperl import utils
from sperl.metapolicies.abstract_meta_policy import MetaPolicy


class DelayedRR(MetaPolicy):
    NAME = "DRR"

    def reset(self, meta_env, random_generator):
        super().reset(meta_env, random_generator)
        self._n_steps = 0
        self._n_iterations = 0
        self._d_bar = None
        self._num_samples: int = -1
        self.empirical_occ_start = None

    def meta_step(self, meta_observation, policy_idx):
        assert policy_idx == self.policy_index
        num_samples = 0

        self._n_steps += 1
        (
            empirical_occ,
            self._reward_value,
        ) = self.empirical_occupancy_n_reward(meta_observation)

        if self._n_steps == self._run_config.k:
            self._n_iterations += 1
            self._n_steps = 0
            # calculate the new optimal occupancy measure
            if self.exact_opt_problem:
                raise NotImplementedError("Currently not supported")

            if self._d_bar is None:
                self._d_bar = empirical_occ
            else:
                self._d_bar = (
                    self._d_bar * self._general_config.previous_occupancy_mix
                    + empirical_occ * (1 - self._general_config.previous_occupancy_mix)
                )

            self._current_occupancy = self.opt_problem.solve(
                [meta_observation],
                [self._d_bar],
                self.policy_index,
                np.zeros((self.num_states, self.num_actions)),
            )
            num_samples = utils.num_samples(meta_observation)

        self._num_samples = num_samples

    def get_num_samples_opt(self):
        return self._num_samples

    def get_num_samples_occ(self):
        return self._num_samples

    def got_updated(self) -> bool:
        return self._n_steps == 0
