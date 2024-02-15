class MeasureSummaryEntry:
    def __init__(self, mean: float, std_err: float, resource: int):
        self.mean = mean
        self.std_err = std_err
        self.resource = resource
