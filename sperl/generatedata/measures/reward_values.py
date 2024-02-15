from typing import List

from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig
from sperl.data.measure_entry import MeasureEntry
from sperl.data.occupancy import OccupancyMeasure
from sperl.generatedata.measures.base import BaseMeasures


class RewardValues(BaseMeasures):
    def __init__(self, run_config: SingleRunConfig, general_config: GeneralConfig):
        super().__init__(run_config, general_config)
        self._rewards_list: List[MeasureEntry] = []

    def step(
        self,
        occupancy: OccupancyMeasure,
        step,
        num_samples,
        got_updated: bool,
        reward_value: float,
    ):
        super().step(occupancy, step, num_samples, got_updated, reward_value)
        self._rewards_list.append(
            MeasureEntry(
                reward_value,
                step,
                self._total_num_samples,
                self._num_retrainings,
            )
        )

    @property
    def measures_list(self) -> List[MeasureEntry]:
        return self._rewards_list

    @staticmethod
    def name() -> str:
        return "reward"

    @staticmethod
    def axis_text() -> str:
        return "$V_t^{d_t}$"
