from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SmoothedScalarEstimator:
    alpha: float = 0.9
    epsilon: float = 1e-9
    _cross: np.ndarray | None = field(default=None, init=False, repr=False)
    _auto: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    def estimate(self, prototype: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prototype = np.asarray(prototype, dtype=np.complex128)
        source = np.asarray(source, dtype=np.complex128)
        if prototype.shape != source.shape:
            raise ValueError(f"prototype shape {prototype.shape} != source shape {source.shape}")

        instant_cross = prototype * np.conjugate(source)
        instant_auto = np.abs(source) ** 2

        if self._cross is None:
            self._cross = instant_cross
            self._auto = instant_auto
        else:
            self._cross = (1.0 - self.alpha) * instant_cross + self.alpha * self._cross
            self._auto = (1.0 - self.alpha) * instant_auto + self.alpha * self._auto

        weights = np.real(self._cross) / (self._auto + self.epsilon)
        estimated = weights * source
        return estimated, weights
