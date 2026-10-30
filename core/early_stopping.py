"""Validation-only early stopping with serializable resume state."""
import math
from dataclasses import dataclass, asdict


@dataclass
class EarlyStopping:
    patience: int = 3
    min_delta: float = 0.0
    best: float = -math.inf
    bad_rounds: int = 0
    stopped: bool = False

    def __post_init__(self):
        if self.patience < 1 or self.min_delta < 0 or not math.isfinite(self.min_delta):
            raise ValueError('Patience must be positive and min_delta finite and nonnegative')

    def update(self, score):
        if not math.isfinite(score):
            raise ValueError('Nonfinite validation score')
        if self.stopped:
            return True
        if score > self.best + self.min_delta:
            self.best, self.bad_rounds = score, 0
        else:
            self.bad_rounds += 1
        self.stopped = self.bad_rounds >= self.patience
        return self.stopped

    def state_dict(self):
        return asdict(self)
