from collections import defaultdict
from typing import DefaultDict


class AnalyticsManager:
    metrics: DefaultDict[str, list[float]]

    def __init__(self):
        self.metrics = defaultdict(lambda: [])

    def log(self, metrics: dict[str, float]):
        for k, v in metrics.items():
            self.metrics[k].append(v)
