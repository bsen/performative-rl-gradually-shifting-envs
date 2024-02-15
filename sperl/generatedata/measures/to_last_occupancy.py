from typing import List, cast

import numpy as np

from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig
from sperl.data.measure_entry import MeasureEntry
from sperl.data.occupancy import OccupancyMeasure
from sperl.generatedata.measures.base import BaseMeasures


class ToLastOccupancy(BaseMeasures):
    def __init__(self, run_config: SingleRunConfig, general_config: GeneralConfig):
        super().__init__(run_config, general_config)
        self._occupancy_list: List[np.ndarray] = []
        self._step_list: List[int] = []
        self._num_samples_list: List[int] = []
        self._computed: bool = False
        self._num_retrainings_list: List[int] = []

    def step(
        self,
        occupancy: OccupancyMeasure,
        step,
        num_samples,
        got_updated: bool,
        reward_value: float,
    ):
        super().step(occupancy, step, num_samples, got_updated, reward_value)
        self._occupancy_list.append(occupancy.occ.flatten())
        self._step_list.append(step)
        self._num_samples_list.append(self._total_num_samples)
        self._num_retrainings_list.append(self._num_retrainings)

    def compute(self):
        self._distance_list = []
        last_occupancies = self._occupancy_list[-10:]
        limiting_occ = np.mean(last_occupancies, axis=0)
        for i, occ in enumerate(self._occupancy_list[:-1]):
            dist = cast(float, np.linalg.norm(occ - limiting_occ))  # type: ignore
            self._distance_list.append(
                MeasureEntry(
                    dist,
                    self._step_list[i],
                    self._num_samples_list[i],
                    self._num_retrainings_list[i],
                )
            )
        del self._occupancy_list
        self._computed = True

    @property
    def measures_list(self) -> List[MeasureEntry]:
        if not self._computed:
            self.compute()
        return self._distance_list

    @staticmethod
    def name() -> str:
        return "occupancy-last"

    @staticmethod
    def axis_text() -> str:
        return "$||d_t - d_{\\operatorname{last}}||_2$"
