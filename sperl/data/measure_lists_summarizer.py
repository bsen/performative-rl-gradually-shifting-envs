from typing import List, Tuple

import numpy as np

from sperl.data.measure_summary_entry import MeasureSummaryEntry
from sperl.data.measure_entry import MeasureEntry


class MeasureListSummarizer:
    ERROR_FACTOR = 1.96

    def __init__(self, distance_lists: List[List[MeasureEntry]], over_variable: str):
        self._distance_lists = distance_lists
        self._cur_indices = [0 for _ in distance_lists]
        self._over_variable = over_variable

    def __iter__(self):
        return self

    def _get_current_entries(self) -> List[MeasureEntry]:
        return [l[i] for l, i in zip(self._distance_lists, self._cur_indices)]

    def _get_next_entries(self):
        for l, i in zip(self._distance_lists, self._cur_indices):
            if i == len(l) - 1:
                yield None
            else:
                yield l[i]

    def _smallest_next_entries(self) -> Tuple[List[int], int]:
        indices = []
        over_var_val = 9223372036854775807  # 2^63 -1
        for i, entry in enumerate(self._get_next_entries()):
            if entry is not None:
                value = getattr(entry, self._over_variable)
                if value == over_var_val:
                    indices.append(i)
                elif value < over_var_val:
                    over_var_val = value
                    indices = [i]
        if not indices:
            raise StopIteration
        return indices, over_var_val

    def __next__(self) -> MeasureSummaryEntry:
        next_indices, var_value = self._smallest_next_entries()

        for idx in next_indices:
            self._cur_indices[idx] += 1
        cur_entries = self._get_current_entries()
        cur_distncs = [e.distance for e in cur_entries]

        std_err = (
            self.ERROR_FACTOR * np.std(cur_distncs) / np.sqrt(len(self._distance_lists))
        )
        mean = np.mean(cur_distncs)
        return MeasureSummaryEntry(mean, std_err, var_value)

    def get_summary(self) -> List[MeasureSummaryEntry]:
        return [e for e in self]
