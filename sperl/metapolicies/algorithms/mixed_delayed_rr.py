import math

import numpy as np

from sperl import utils
from sperl.metapolicies.abstract_meta_policy import MetaPolicy


class MixedDelayedRR(MetaPolicy):
    NAME = "MDRR"

    def reset(self, meta_env, random_generator):
        super().reset(meta_env, random_generator)
        self.empirical_occupancy_list = []
        self.trajectories_list = []
        self.num_samples_occ_list = []
        self._n_steps = 0
        self.n_iterations = 0
        self.d_bar = None
        if self.exact_opt_problem:
            raise ValueError(
                "MixedDelayedRR should not be used with an exact"
                "optimization problem, but with the approximate one."
            )
        self._num_samples_opt: int
        self._num_samples_occ: int

    def meta_step(self, meta_observation, policy_idx):
        assert policy_idx == self.policy_index
        # the number of samples used to calculate the occupancy measure which is
        # returned.
        num_samples_occ = 0
        # the number of samples used for the optimization problem (apart from the
        # ones used to calculate the empirical occupancy measures for every step)
        num_samples_opt = 0
        cur_k = self._run_config.k
        self._n_steps += 1
        self.num_samples_occ_list.append(utils.num_samples(meta_observation))

        emp_occ, self._reward_value = self.empirical_occupancy_n_reward(
            meta_observation
        )
        self.empirical_occupancy_list.append(emp_occ)
        self.trajectories_list.append(meta_observation)

        if self._n_steps == cur_k:
            self._n_steps = 0
            # calculate the samples for each round, by using
            # algorithm 6 from the paper.

            # F from algorithm 6
            sample_list = []

            # the list of occupancy measures used for the optimization problem
            occupancy_list = []

            # M' from algorithm 6
            num_samples_opt = np.inf

            # |F| from algorithm 6
            num_samples_used = 0

            # W from algorithm 6
            cur_total_weight = 0

            for t in range(cur_k, 0, -1):
                trajectories_t = self.trajectories_list[t - 1]
                num_samples_t = self.num_samples_occ_list[t - 1]
                occupancy_t = self.empirical_occupancy_list[t - 1]
                occupancy_list.append(occupancy_t)
                num_samples_occ += num_samples_t

                # v from theorem 5
                v = self.v
                # w_t from theorem 5
                weight_t = ((v - 1) / (v**cur_k - 1)) * v ** (t - 1)

                if num_samples_opt <= num_samples_used + num_samples_t:
                    sample_list.append(
                        self.take_n_samples(
                            trajectories_t,
                            int(num_samples_opt - num_samples_used),
                        )
                    )
                    break

                sample_list.append(trajectories_t)
                num_samples_used += num_samples_t
                cur_total_weight += weight_t

                if num_samples_used - cur_total_weight * num_samples_opt < 0:
                    num_samples_opt = num_samples_used / cur_total_weight

            self._current_occupancy = self.opt_problem.solve(
                sample_list,
                occupancy_list,
                self.policy_index,
                np.zeros((self.num_states, self.num_actions)),
            )
            self.num_samples_occ_list = []
            self.trajectories_list = []
            self.n_iterations += 1
            self.empirical_occupancy_list = []

        self._num_samples_opt = math.ceil(num_samples_opt)
        self._num_samples_occ = num_samples_occ

    def get_num_samples_opt(self):
        return self._num_samples_opt

    def get_num_samples_occ(self):
        return self._num_samples_occ

    def take_n_samples(self, trajectories, num_samples):
        """
        Args:
            trajectories -- a list of size num_trajectories x len_trajectory x 3,
                where trajectories_over_t[a,b,c] corresponds trajectory a of the
                trajectories of this round, the b-th entry in this trajectory
                and if c==0 the state, c==1 the action and c==2 the reward
                of this entry.
        Returns:
            a list of trajectories which contains num_samples samples and is a subset
            of the input list of trajectories. (the last trajectory in the return value
            may be shortened so that in total the number of samples is num_samples.)
        """
        result = []
        result_num_samples = 0
        for trajectory in trajectories:
            cur_num_samples = utils.num_samples([trajectory])
            if num_samples >= result_num_samples + cur_num_samples:
                result.append(trajectory)
                result_num_samples += cur_num_samples
            else:
                num_samples_left = num_samples - result_num_samples
                result.append(trajectory[: num_samples_left + 1])
                return result

            if num_samples == result_num_samples:
                return result

        raise ValueError(
            "The number of samples in the trajectories is ",
            utils.num_samples(trajectories),
            "which is lower than " "num_samples",
            num_samples,
        )

    def got_updated(self) -> bool:
        return self._n_steps == 0
