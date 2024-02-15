from typing import Any, Tuple

import numpy as np


class SingleRunConfig:
    def __init__(self):
        self.beta: float
        self.regularizer: float
        self.gamma: float
        self.num_trajectories: int
        self.num_ftrl_steps: int
        self.b: float
        self.k: int
        self.agent2_algorithm: Tuple[str, Any]
        self.meta_policy: str
        self.seed_seq: np.random.SeedSequence
        self.v: float  # v for the MDRR algorithm

    def get_str(self, num_steps: int):
        result = f"steps={num_steps}"
        for arg, val in self.__dict__.items():
            if arg == "seed_seq":
                continue
            result += "_{}={}".format(arg, val)
        return result

    def print_info(self):
        for arg, val in self.__dict__.items():
            if arg == "seed_seq":
                continue
            print("  {} = {}".format(arg, val))
