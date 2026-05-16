from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chorus.estimation import SmoothedScalarEstimator
from chorus.prototypes import center_prototype, surround_prototype
from chorus.transforms import STFTConfig, STFTTransform


@dataclass(frozen=True)
class ChorusConfig:
    sample_rate: int
    transform: str = "stft"
    frame_size: int = 1024
    hop_size: int = 512
    smoothing_alpha: float = 0.9
    epsilon: float = 1e-9


@dataclass(frozen=True)
class ChorusResult:
    center: np.ndarray
    only: np.ndarray
    surround: np.ndarray
    contributions: dict[str, np.ndarray]
    metadata: dict[str, object]


class ChorusProcessor:
    def __init__(self, config: ChorusConfig) -> None:
        if config.transform != "stft":
            raise NotImplementedError("full splitting is supported only for STFT in v0")
        self.config = config
        self.transform = STFTTransform(STFTConfig(config.frame_size, config.hop_size))

    def process(self, stereo: np.ndarray) -> ChorusResult:
        representation = self.transform.forward(stereo)
        left = representation.data[0]
        right = representation.data[1]

        center_proto = center_prototype(left, right)
        surround_proto = surround_prototype(left, right)

        left_center = self._estimate_channel(center_proto, left)
        right_center = self._estimate_channel(center_proto, right)
        left_surround = self._estimate_channel(surround_proto, left)
        right_surround = -self._estimate_channel(surround_proto, right)

        center_components = np.stack([left_center, right_center], axis=0)
        surround_components = np.stack([left_surround, right_surround], axis=0)
        input_components = np.stack([left, right], axis=0)
        only_components = input_components - center_components - surround_components

        center = self.transform.inverse_components(
            center_components, representation.original_shape
        )
        surround = self.transform.inverse_components(
            surround_components, representation.original_shape
        )
        only = self.transform.inverse_components(only_components, representation.original_shape)
        contributions = {
            "Lc": center[:, 0],
            "Rc": center[:, 1],
            "Lo": only[:, 0],
            "Ro": only[:, 1],
            "Ls": surround[:, 0],
            "Rs": surround[:, 1],
        }

        return ChorusResult(
            center=center,
            only=only,
            surround=surround,
            contributions=contributions,
            metadata={
                "sample_rate": self.config.sample_rate,
                "transform": representation.metadata,
                "smoothing_alpha": self.config.smoothing_alpha,
                "epsilon": self.config.epsilon,
            },
        )

    def _estimate_channel(self, prototype: np.ndarray, source: np.ndarray) -> np.ndarray:
        estimator = SmoothedScalarEstimator(
            alpha=self.config.smoothing_alpha,
            epsilon=self.config.epsilon,
        )
        output = np.empty_like(source)
        for frame_index in range(source.shape[-1]):
            output[..., frame_index], _weights = estimator.estimate(
                prototype[..., frame_index],
                source[..., frame_index],
            )
        return output
