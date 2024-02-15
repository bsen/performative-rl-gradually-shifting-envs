from typing import List

from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig
from sperl.data.measure_entry import MeasureEntry
from sperl.data.occupancy import OccupancyMeasure


class BaseMeasures:
    def __init__(self, run_config: SingleRunConfig, general_config: GeneralConfig):
        self.run_config = run_config
        self.general_config = general_config
        self._distance_list: List[MeasureEntry]
        self._num_retrainings: int = 0
        self._total_num_samples: int = 0

    def step(
        self,
        occupancy: OccupancyMeasure,
        step,
        num_samples,
        got_updated: bool,
        reward_value: float,
    ):
        self._total_num_samples += num_samples
        if got_updated:
            self._num_retrainings += 1

    @property
    def measures_list(self) -> List[MeasureEntry]:
        raise NotImplementedError()

    @staticmethod
    def name() -> str:
        raise NotImplementedError()

    @staticmethod
    def axis_text() -> str:
        raise NotImplementedError()
