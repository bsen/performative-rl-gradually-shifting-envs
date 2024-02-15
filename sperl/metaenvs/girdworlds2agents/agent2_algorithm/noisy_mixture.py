import numpy as np

from sperl.metaenvs.girdworlds2agents.agent2_algorithm.decaying_mixture import (
    DecayingMixture,
)


class NoisyMixture(DecayingMixture):
    NAME = "noisy-mixture"
    NOISE_STD = 0.05

    def step(
        self,
        agent1_policy,
        old_agent2_policy: np.ndarray,
        agent2_env,
    ) -> np.ndarray:
        agent1_policy = self._add_noise(agent1_policy)
        return super().step(agent1_policy, old_agent2_policy, agent2_env)

    def _add_noise(self, policy: np.ndarray):
        noise = np.random.normal(0, self.NOISE_STD, policy.shape)
        noisy_policy = policy + noise
        noisy_policy = np.maximum(0, noisy_policy)  # Clip negative values to 0
        row_sums = np.sum(noisy_policy, axis=1, keepdims=True)
        noisy_policy /= row_sums
        return noisy_policy
