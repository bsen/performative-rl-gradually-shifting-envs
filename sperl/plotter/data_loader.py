import os
import pickle
from typing import Any, Iterator, List, Tuple

import numpy as np

from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig
from sperl.data.measure_summary_entry import MeasureSummaryEntry
from sperl.generatedata.measures.measure_statistics import MeasureStatistics


class DataLoader:
    def __init__(
        self,
        folder: str,
        files: List[str],
        variable: str,
        distance: str,
        over_resource: str,
    ):
        self._folder = folder
        self._files = files
        self._variable = variable
        self._distance: str = distance
        self._over_resource: str = over_resource

    def load_values(
        self,
    ) -> Iterator[Tuple[Any, np.ndarray, np.ndarray, List]]:
        for file in self._files:
            fn = os.path.join(self._folder, file)
            yield self._load_file(fn)

    def _load_file(self, filename: str) -> Tuple[Any, np.ndarray, np.ndarray, List]:
        run_config: SingleRunConfig
        distance_statistics: MeasureStatistics
        _, run_config, distance_statistics = self._config_n_distance(filename)
        val = getattr(run_config, self._variable)
        dist_sum = self._list_distance_entries(distance_statistics)
        means: np.ndarray = np.array([d.mean for d in dist_sum])
        std_errs: np.ndarray = np.array([d.std_err for d in dist_sum])
        over_values = [d.resource for d in dist_sum]
        return val, means, std_errs, over_values

    def _config_n_distance(
        self, filename: str
    ) -> Tuple[GeneralConfig, SingleRunConfig, MeasureStatistics]:
        with open(filename, "rb") as f:
            result = pickle.load(f)
        return result

    def _list_distance_entries(
        self, distance_statistics: MeasureStatistics
    ) -> List[MeasureSummaryEntry]:
        ds = distance_statistics.distance_summaries[self._distance]
        return getattr(ds, "over_{}".format(self._over_resource))
