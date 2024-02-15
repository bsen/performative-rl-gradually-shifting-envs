from typing import List

import numpy as np

from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig
from sperl.generatedata.measures.all_measures import AllMeasures
from sperl.generatedata.measures.stepwise_occupancy import StepwiseOccupancy
from sperl.metaenvs.girdworlds2agents.meta_env import Gridworld2AgentEnv
from sperl.metapolicies.algorithms import get_meta_policy


class MetaTrajectoryGenerator:
    def __init__(
        self,
        run_config: SingleRunConfig,
        general_config: GeneralConfig,
        seed_sequence: np.random.SeedSequence,
    ):
        self.run_config = run_config
        self.general_config = general_config
        child_seeds = seed_sequence.spawn(2)

        self.meta_env = Gridworld2AgentEnv(
            self.run_config, self.general_config, child_seeds[0]
        )
        self.policy_rng = np.random.default_rng(child_seeds[1])
        meta_policy_class = get_meta_policy(self.run_config.meta_policy)
        self.meta_policy = meta_policy_class(
            self.run_config,
            self.general_config,
            self.meta_env,
            self.policy_rng,
            0,
        )
        self._trajectories: List = []
        self.distances = AllMeasures(self.run_config, self.general_config)
        self._occ_updated = False
        self.meta_policy.reset(self.meta_env, self.policy_rng)
        self.step = 0
        self.finished = False

    def run(self, max_steps: int):
        iteratr = range(self.step, self.step + max_steps)
        n_deploys = self.general_config.num_deployments
        will_finish = max_steps > (n_deploys - self.step)
        if will_finish:
            iteratr = range(self.step, n_deploys)
        for self.step in iteratr:
            self._update_environment()
            self.meta_policy.meta_step(self._trajectories, 0)
            self.distances.step(
                self.meta_policy.current_occupancy,
                self.step,
                self.meta_policy.get_num_samples_opt(),
                self.meta_policy.got_updated(),
                self.meta_policy.current_reward_value,
            )
            if self.meta_policy.got_updated():
                self._occ_updated = True
            self.print_step()
        self.step += 1

        if will_finish:
            self.finished = True

    def print_step(self):
        print(self.step, ",", sep="", end="", flush=True)
        if self.step % 100 == 0:
            print()

    def _print_info(self, step: int):
        print("Iteration: {}".format(step))
        if step != 0 and self._occ_updated:
            cur_dist = (
                self.distances.distance_dict[StepwiseOccupancy.name()]  # type: ignore
                .measures_list[-1]
                .distance
            )
            print("  distance: {}".format(cur_dist))

    def _update_environment(self):
        self.meta_env.change_function(
            [self.meta_policy.current_occupancy.policy],
            0,
        )
        self._trajectories = self.meta_env.get_trajectories(
            self.run_config.num_trajectories
        )
