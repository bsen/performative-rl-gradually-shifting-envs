from typing import List

import numpy as np

from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig
from sperl.data.measure_entry import MeasureEntry
from sperl.data.occupancy import OccupancyMeasure
from sperl.generatedata.measures.base import BaseMeasures


class StepwiseOccupancy(BaseMeasures):
    def __init__(self, run_config: SingleRunConfig, general_config: GeneralConfig):
        super().__init__(run_config, general_config)
        self._distance_list: List[MeasureEntry] = []

        self._first_round = True
        self._previous_occ: np.ndarray

    def step(
        self,
        occupancy: OccupancyMeasure,
        step,
        num_samples,
        got_updated: bool,
        reward_value: float,
    ):
        super().step(occupancy, step, num_samples, got_updated, reward_value)
        occ_flat = occupancy.occ.flatten()
        if not self._first_round and got_updated:
            dist = np.linalg.norm(
                occ_flat - self._previous_occ  # type: ignore
            ) / np.linalg.norm(self._previous_occ)
            self._distance_list.append(
                MeasureEntry(dist, step, self._total_num_samples, self._num_retrainings)
            )

        self._first_round = False
        self._previous_occ = occ_flat

    @staticmethod
    def name() -> str:
        return "occupancy"

    @property
    def measures_list(self) -> List[MeasureEntry]:
        return self._distance_list

    @staticmethod
    def axis_text() -> str:
        return "$c_t||d_{t+1} - d_t||^2$"
