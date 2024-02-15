class Environment:
    """A class describing what an environment should implement."""

    def __init__(self, env_setting, random_generator):
        raise NotImplementedError

    def is_terminal(self, state):
        raise NotImplementedError

    @property
    def num_states(self):
        raise NotImplementedError

    def actions(self, state):
        raise NotImplementedError

    def reward_and_next_state(self, state, action):
        """
        Returns an array reward_state of size M x 3
        where reward_state[i, 0] is the probability of reward
        reward_state[i, 1] and next state reward_state[i, 2] occurring after
        state and action.
        """
        raise NotImplementedError
