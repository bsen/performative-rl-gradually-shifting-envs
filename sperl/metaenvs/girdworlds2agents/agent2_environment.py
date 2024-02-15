import itertools
from typing import List

import numpy as np

from sperl.metaenvs.abstract_env import Environment
from sperl.utils import exact_value_iteration


class Agent2Environment(Environment):
    def __init__(self, meta_env):
        self._meta_env = meta_env
        self._random_generator = meta_env.get_random_generator()
        self._general_config = meta_env.get_general_config()
        self._run_config = meta_env.get_run_config()
        self._agent2_grid = self._perturb_grid(self._random_generator, 0.3)
        self._policy_agent1: np.ndarray
        self._state_action_table: List[List]

    def is_terminal(self, state):
        return self._meta_env.is_terminal(state)

    def set_policy_agent1(self, policy_agent1):
        self._policy_agent1 = policy_agent1
        self._state_action_table = self._state_action_table_agent2()

    @property
    def num_states(self):
        return self._meta_env.num_states()

    def actions(self, state):
        return [0, 1, 2, 3, 4]

    def _state_action_table_agent2(self):
        """
        Returns a list table of size num_states x 5 x M x 3,
        where table[s, a, i, 0] is the probability of reward table[s, a, i, 1]
        and state table[s, a, i, 2] to occur after state s and action a.
        """
        table = [[[] for _ in range(5)] for _ in range(self._meta_env.num_states())]

        for state in range(self._meta_env.num_states()):
            # handle actions 0, 1, 2 and 3
            for action in [0, 1, 2, 3]:
                next_coord = self._meta_env.next_coord(state, action)
                next_state = self._meta_env.coord_to_state(next_coord)
                reward = self._agent2_grid[next_coord[0], next_coord[1]] - 0.05
                table[state][action] = [[1, reward, next_state]]

            # handle action 4
            table[state][4] = [[] for _ in range(4)]
            for action_agent1 in [0, 1, 2, 3]:
                next_coord = self._meta_env.next_coord(state, action_agent1)
                next_state = self._meta_env.coord_to_state(next_coord)
                reward = self._agent2_grid[next_coord[0], next_coord[1]]

                table[state][4][action_agent1] = [
                    self._policy_agent1[state, action_agent1],
                    reward,
                    next_state,
                ]

        return table

    def reward_and_next_state(self, state, action):
        """
        Returns an array reward_state of size M x 3
        where reward_state[i, 0] is the probability of reward
        reward_state[i, 1] and state reward_state[i, 2] occurring.
        """
        return self._state_action_table[state][action]

    def _perturb_grid(self, random_num_gen, threshold):
        meta_env = self._meta_env
        grid_shape = meta_env.grid_world.shape

        grid_values = np.unique(meta_env.grid_world)
        grid_values = np.delete(grid_values, np.where(grid_values == 1))

        new_grid = np.zeros(shape=grid_shape)
        random_uniform = random_num_gen.random(grid_shape)
        for i, j in itertools.product(range(grid_shape[0]), range(grid_shape[1])):
            # initial states stay the same
            if (
                meta_env.coord_to_state([i, j]) in meta_env.initial_states
                or random_uniform[i, j] > threshold
            ):
                new_grid[i, j] = meta_env.grid_world[i, j]
            else:
                new_grid[i, j] = random_num_gen.choice(grid_values)

        # goal value remains the same
        new_grid[-1, -1] = meta_env.grid_world[-1, -1]

        return new_grid

    def state_action_value(self, policy_agent1):
        """
        Returns the optimal state-action value function of agent 2 in the
        current environment, given a policy of agent 1.
        """
        gamma = self._general_config.agent2_gamma
        self.set_policy_agent1(policy_agent1)
        # the state value function:
        state_value = exact_value_iteration(self, self._general_config.agent2_gamma)
        state_action_value = np.zeros((self._meta_env.num_states(), 5))
        # now use the state-value function, to calculate the state-action-value
        # function
        for state in range(self._meta_env.num_states()):
            for action_agent2 in range(5):
                for prob, reward, next_state in self._state_action_table[state][
                    action_agent2
                ]:
                    state_action_value[state, action_agent2] += prob * (
                        reward + gamma * state_value[next_state]
                    )

        return state_action_value
