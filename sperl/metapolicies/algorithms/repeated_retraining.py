import numpy as np

from sperl import utils
from sperl.metapolicies.abstract_meta_policy import MetaPolicy


class RepeatedRetraining(MetaPolicy):
    NAME = "RR"

    def reset(self, meta_env, random_generator):
        super().reset(meta_env, random_generator)
        self.d_bar = None
        self._num_samples: int

    def meta_step(self, meta_observation, policy_idx):
        assert policy_idx == self.policy_index
        if self.exact_opt_problem:
            raise NotImplementedError("Currently not supported")

        empirical_occ, self._reward_value = self.empirical_occupancy_n_reward(
            meta_observation
        )

        if self.d_bar is None:
            self.d_bar = empirical_occ
        else:
            self.d_bar = (
                self.d_bar * self._general_config.previous_occupancy_mix
                + empirical_occ * (1 - self._general_config.previous_occupancy_mix)
            )

        self._current_occupancy = self.opt_problem.solve(
            [meta_observation],
            [self.d_bar],
            self.policy_index,
            np.zeros((self.num_states, self.num_actions)),
        )

        self._num_samples = utils.num_samples(meta_observation)

    def get_num_samples_opt(self):
        return self._num_samples

    def get_num_samples_occ(self):
        return self._num_samples

    def got_updated(self) -> bool:
        return True
