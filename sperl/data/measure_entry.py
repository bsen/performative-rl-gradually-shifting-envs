class MeasureEntry:
    def __init__(self, distance: float, step: int, samples: int, retrainings: int):
        self.distance = distance
        self.step = step
        self.samples = samples
        self.retrainings = retrainings
