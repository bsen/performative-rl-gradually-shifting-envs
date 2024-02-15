from typing import Dict, List

from sperl.data.measure_entry import MeasureEntry
from sperl.data.measure_summary import MeasureSummary
from sperl.generatedata.measures.all_measures import AllMeasures


class MeasureStatistics:
    def __init__(self, distances: List[AllMeasures]):
        dist_by_name = self.raw_distances_by_name(distances)
        self.distance_summaries: Dict[str, MeasureSummary] = {}
        for name, raw_dist in dist_by_name.items():
            self.distance_summaries[name] = MeasureSummary(raw_dist)

    def raw_distances_by_name(
        self, distances: List[AllMeasures]
    ) -> Dict[str, List[List[MeasureEntry]]]:
        distance_names: List[str] = list(distances[0].distance_dict.keys())
        num_meta_traj = len(distances)

        raw_distances_by_name: Dict[str, List[List[MeasureEntry]]] = {}
        for name in distance_names:
            raw_distances: List[List[MeasureEntry]] = []
            for i in range(num_meta_traj):
                raw_distances.append(distances[i].distance_dict[name].measures_list)
            raw_distances_by_name[name] = raw_distances
        return raw_distances_by_name
