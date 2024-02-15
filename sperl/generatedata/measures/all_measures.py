from typing import Dict, List, Type

from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig
from sperl.data.occupancy import OccupancyMeasure
from sperl.generatedata.measures.base import BaseMeasures
from sperl.generatedata.measures.reward_values import RewardValues
from sperl.generatedata.measures.stepwise_occupancy import StepwiseOccupancy
from sperl.generatedata.measures.to_last_occupancy import ToLastOccupancy


class AllMeasures:
    DISTANCES: List[Type[BaseMeasures]] = [
        StepwiseOccupancy,
        ToLastOccupancy,
        RewardValues,
    ]

    def __init__(self, run_config: SingleRunConfig, general_config: GeneralConfig):
        self._distances_dict: Dict[str, BaseMeasures] = {}
        for d in self.DISTANCES:
            self._distances_dict[d.name()] = d(run_config, general_config)  # type: ignore

    def step(
        self,
        occupancy: OccupancyMeasure,
        step,
        num_samples,
        got_updated: bool,
        reward_value: float,
    ):
        for d in self._distances_dict.values():
            d.step(occupancy, step, num_samples, got_updated, reward_value)

    @property
    def distance_dict(self) -> Dict[str, BaseMeasures]:
        return self._distances_dict

    @staticmethod
    def get_distance(name: str) -> Type[BaseMeasures]:
        for d in AllMeasures.DISTANCES:
            if d.name() == name:
                return d
        raise ValueError('Distance "{}" is not supported'.format(name))
