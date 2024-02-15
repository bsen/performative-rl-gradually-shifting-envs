from typing import Any, List, Tuple


class ExperimentConfig:
    def __init__(self):
        self.beta: List[float] = []
        self.regularizer: List[float] = []
        self.gamma: List[float] = [0.9]
        self.num_trajectories: List[int] = []
        self.num_ftrl_steps: List[int] = []
        self.b: List[float] = []
        self.k: List[int] = []
        self.agent2_algorithm: List[Tuple[str, Any]] = []
        self.meta_policy: List[str] = []
        self.v: List[float] = []
        self.seed: int = 1234
