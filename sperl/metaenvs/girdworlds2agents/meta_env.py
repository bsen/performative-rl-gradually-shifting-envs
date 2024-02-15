from typing import List

import numpy as np

from sperl import utils
from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig
from sperl.metaenvs.abstract_meta_env import MetaEnvironment
from sperl.metaenvs.girdworlds2agents.agent2_algorithm.base import Agent2AlgoBase
from sperl.metaenvs.girdworlds2agents.agent2_algorithm.decaying_mixture import (
    DecayingMixture,
)
from sperl.metaenvs.girdworlds2agents.agent2_algorithm.noisy_mixture import NoisyMixture


class Gridworld2AgentEnv(MetaEnvironment):
    AGENT2_ALGORITHMS = [DecayingMixture, NoisyMixture]
    H_REWARD = -0.5
    F_REWARD = -0.02
    GRIDS: List[np.ndarray] = [
        np.array(  # 0
            [
                [-0.01, -0.01, -0.01, -0.01],
                [-0.01, F_REWARD, -0.01, H_REWARD],
                [-0.01, H_REWARD, -0.01, H_REWARD],
                [-0.01, F_REWARD, -0.01, +1],
            ]
        ),
        np.array(  # 1
            [
                [-0.01, -0.01, -0.01],
                [-0.01, F_REWARD, -0.01],
                [-0.01, H_REWARD, H_REWARD],
                [-0.01, F_REWARD, +1],
            ]
        ),
        np.array(  # 2
            [
                [-0.01, -0.01, -0.01],
                [-0.01, F_REWARD, -0.01],
                [-0.01, H_REWARD, +1],
            ]
        ),
        np.array(  # 3
            [
                [-0.01, -0.01, -0.01],
                [-0.01, F_REWARD, +1],
            ]
        ),
        np.array(  # 4
            [
                [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
                [-0.01, -0.01, F_REWARD, -0.01, H_REWARD, -0.01, -0.01, -0.01],
                [-0.01, -0.01, -0.01, H_REWARD, -0.01, -0.01, F_REWARD, -0.01],
                [
                    -0.01,
                    F_REWARD,
                    -0.01,
                    -0.01,
                    -0.01,
                    H_REWARD,
                    -0.01,
                    F_REWARD,
                ],
                [-0.01, -0.01, -0.01, H_REWARD, -0.01, -0.01, F_REWARD, -0.01],
                [
                    -0.01,
                    H_REWARD,
                    H_REWARD,
                    -0.01,
                    F_REWARD,
                    -0.01,
                    H_REWARD,
                    -0.01,
                ],
                [
                    -0.01,
                    H_REWARD,
                    -0.01,
                    -0.01,
                    H_REWARD,
                    -0.01,
                    H_REWARD,
                    -0.01,
                ],
                [-0.01, -0.01, -0.01, H_REWARD, -0.01, F_REWARD, -0.01, +1],
            ]
        ),
        np.array(  # 5
            [
                [-0.01, -0.01, -0.01, -0.01],
                [-0.01, F_REWARD, -0.01, H_REWARD],
                [-0.01, -0.01, H_REWARD, H_REWARD],
                [-0.01, F_REWARD, -0.01, +1],
            ]
        ),
    ]

    def __init__(
        self,
        run_config: SingleRunConfig,
        general_config: GeneralConfig,
        seed_sequence: np.random.SeedSequence,
    ):
        seeds = seed_sequence.spawn(2)
        random_generator = np.random.default_rng(seeds[0])
        super().__init__(run_config, general_config, random_generator)
        self.seed_sequence = seeds[1]
        self.initial_states = [
            state
            for state in range(self.num_states())
            if (coords := self.state_to_coord(state))
            and (coords[0] == 0 or coords[1] == 0)
        ]
        self.reset(random_generator)
        self.terminal_state = self.coord_to_state(
            (self.grid_world.shape[0] - 1, self.grid_world.shape[1] - 1)
        )
        self.probability_transition_fun = np.empty(0)
        self.reward_fun = np.empty(0)
        self.policy = np.empty(0)
        self._agent2_algo: Agent2AlgoBase = self.construct_agent2_algorithm()

    def construct_agent2_algorithm(self) -> Agent2AlgoBase:
        for agent2_algorithm in self.AGENT2_ALGORITHMS:
            if agent2_algorithm.NAME == self._run_config.agent2_algorithm[0]:
                return agent2_algorithm(self._run_config, self._general_config)
        raise ValueError(
            "Algorithm {} for agent 2 is not supported.".format(
                self._run_config.agent2_algorithm[0]
            )
        )

    @property
    def grid_world(self) -> np.ndarray:
        return self.GRIDS[self._general_config.grid_world]

    def change_function(self, policy, policy_index=0):
        """
        Args:
            policy -- A policy, represented by a numpy array with shape
                num_states x num_actions.
                Each entry policy[s, a] represents the probability of choosing
                action a in state s.
            num_trajectories -- the number of trajectories if it is set to a
                value > 0.
                A value of -1 indicates that the return should be not
                trajectories, but the exact probability transition and reward
                functions in form of a tuple (prob_trans, reward_func), where
                prob_trans is a numpy array of shape S x A x S which stores a
                probability transition function. reward_func is a numpy array of
                shape S x A, which stores the reward function.
        """
        assert policy_index == 0
        self.agent2_policy = self._agent2_algo.step(
            policy[0], self.agent2_policy, self.agent2_env
        )

        # compute the probability transition and reward functions
        prob_trans = np.zeros(
            (self.num_states(), self.num_actions(), self.num_states())
        )
        reward_func = np.zeros((self.num_states(), self.num_actions()))
        for state in range(self.num_states()):

            # the reward and transition probability in case agent 2 intervenes
            for action_agent2 in range(4):
                next_state = self.next_state(state, action_agent2)
                reward_func[state] += self.agent2_policy[
                    state, action_agent2
                ] * self._no_intervention_reward(next_state)
                prob_trans[state, :, next_state] += self.agent2_policy[
                    state, action_agent2
                ]

            # reward and transition probabilities in case agent 2 does not intervene
            # (i.e. agent 2 selects action 4)
            for action in range(self.num_actions()):
                no_intervention_next_state = self.next_state(state, action)
                prob_trans[
                    state, action, no_intervention_next_state
                ] += self.agent2_policy[state, 4]
                reward_func[state, action] += self.agent2_policy[
                    state, 4
                ] * self._no_intervention_reward(no_intervention_next_state)

        self.probability_transition_fun = prob_trans
        self.reward_fun = reward_func
        self.policy = policy

    def get_trajectories(self, num_trajectories: int) -> List:
        if num_trajectories <= 0:
            raise ValueError(
                "num_trajectories should be >0, but equals {}".format(num_trajectories)
            )
        seeds = self.seed_sequence.spawn(2)
        self.seed_sequence = seeds[1]
        return utils.generate_trajectories_reward_next_state(
            self.probability_transition_fun,
            self.reward_fun,
            policy=self.policy[0],
            num_trajectories=num_trajectories,
            max_sample_steps=self._general_config.max_trajectory_length,
            meta_env=self,
            seed_sequence=seeds[0],
        )

    def num_states(self, policy_index=0):
        assert policy_index == 0
        return int(np.prod(self.grid_world.shape))

    def num_actions(self, policy_index=0):
        assert policy_index == 0
        return 4

    def reset(self, random_generator):
        # initialize the starting policy of agent2 as the one always choosing
        # action 4
        super().reset(random_generator)
        self.agent2_policy = np.outer(
            np.array((0, 0, 0, 0, 1)), np.ones(self.num_states())
        ).T
        # initialize the randomness
        from sperl.metaenvs.girdworlds2agents.agent2_environment import (  # cyclic import
            Agent2Environment,
        )

        self.agent2_env = Agent2Environment(self)

    def is_terminal(self, state):
        return state == self.terminal_state

    def _no_intervention_reward(self, next_state):
        """Returns the reward when the next state is next_state."""
        next_coord = self.state_to_coord(next_state)
        return self.grid_world[next_coord[0], next_coord[1]]

    def next_coord(self, state, action):
        """
        Returns the next coordinate of the agent in the grid, given a state
        and an action.
        """
        assert action in [0, 1, 2, 3], f"Illegal move {action} was given."

        coord = self.state_to_coord(state)

        if self.is_terminal(state):
            return coord

        action_mapping = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

        def tuple_sum(t1, t2):
            assert len(t1) == 2 and len(t2) == 2
            return t1[0] + t2[0], t1[1] + t2[1]

        coord_action = action_mapping[action]
        next_coord = tuple_sum(coord_action, coord)

        if (
            0 <= next_coord[0] < self.grid_world.shape[0]
            and 0 <= next_coord[1] < self.grid_world.shape[1]
        ):
            return next_coord
        else:
            return coord

    def next_state(self, state, action):
        return self.coord_to_state(self.next_coord(state, action))

    def start_distribution(self, policy_index=0):
        assert policy_index == 0
        rho = np.zeros(self.num_states(), dtype="float64")
        initial_states = self.initial_states
        rho[initial_states] = 1 / len(initial_states)
        return rho

    def state_to_coord(self, state):
        grid_shape = self.grid_world.shape
        coord = (state // grid_shape[1], state % grid_shape[1])

        return coord

    def coord_to_state(self, coordinate):
        grid_shape = self.grid_world.shape
        state_id = coordinate[0] * grid_shape[1] + coordinate[1]

        return state_id

    def num_policies(self) -> int:
        return 1
