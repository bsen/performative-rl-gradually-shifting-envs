from typing import List

from sperl.data.measure_summary_entry import MeasureSummaryEntry
from sperl.data.measure_entry import MeasureEntry
from sperl.data.measure_lists_summarizer import MeasureListSummarizer


class MeasureSummary:
    def __init__(self, distance_lists: List[List[MeasureEntry]]):
        summarizer_step = MeasureListSummarizer(distance_lists, "step")
        summarizer_samples = MeasureListSummarizer(distance_lists, "samples")
        summarizer_retrainings = MeasureListSummarizer(distance_lists, "retrainings")

        self.over_step: List[MeasureSummaryEntry] = summarizer_step.get_summary()
        self.over_samples: List[MeasureSummaryEntry] = summarizer_samples.get_summary()
        self.over_retrainings: List[
            MeasureSummaryEntry
        ] = summarizer_retrainings.get_summary()
