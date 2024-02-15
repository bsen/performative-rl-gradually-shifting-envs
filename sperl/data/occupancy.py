import numpy as np


class OccupancyMeasure:
    def __init__(self, occupancy_measure: np.ndarray):
        self._occupancy: np.ndarray = occupancy_measure
        self._policy_computed: bool = False
        self._policy: np.ndarray

    @property
    def occ(self) -> np.ndarray:
        return self._occupancy

    @property
    def policy(self) -> np.ndarray:
        if not self._policy_computed:
            self._policy = self._compute_policy()
            self._policy_computed = True
        return self._policy

    def _compute_policy(self) -> np.ndarray:
        num_states, num_actions = self.occ.shape
        state_occupancy = np.sum(self.occ, axis=1)

        # the following lines ensure that for the states where the occupancy
        # measure is zero get a value of 1/A in the final policy
        zero_occupancy = np.zeros(num_states)
        zero_occupancy[state_occupancy == 0] = 1
        state_occupancy[state_occupancy == 0] = num_actions
        policy = np.outer(zero_occupancy, np.ones(num_actions))
        policy += self.occ

        denominator = np.outer(state_occupancy, np.ones(num_actions))
        return policy / denominator
