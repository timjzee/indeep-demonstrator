"""Defines the `AbstractModel`, the parent class of all other types of models that are used by the Demonstrator (ASR, TTS, VAD, etc.)."""

from abc import ABC

# Allows MetricTracker to be imported in both src and eval folders
try:
    from src.metrics import MetricTracker
except:
    from metrics import MetricTracker

class AbstractModel(ABC):
    """The parent class of all other types of models that are used by the Demonstrator (ASR, TTS, VAD, etc.).

    Attributes:
        metric_tracker: An object that tracks the metrics of a model.
    """

    def __init__(self) -> None:
        super().__init__()
        
        self.metric_tracker = MetricTracker()
        