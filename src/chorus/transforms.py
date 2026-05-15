from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pywt
from scipy.fft import fft
from scipy.signal import ShortTimeFFT


@dataclass(frozen=True)
class STFTConfig:
    frame_size: int = 1024
    hop_size: int = 512
    fft_size: int | None = None


@dataclass(frozen=True)
class TransformRepresentation:
    data: np.ndarray
    original_shape: tuple[int, int]
    metadata: dict[str, object]


class TimeFrequencyTransform(Protocol):
    def forward(self, stereo: np.ndarray) -> TransformRepresentation:
        """Convert stereo samples shaped (samples, 2) into transform coefficients."""

    def inverse(self, representation: TransformRepresentation) -> np.ndarray:
        """Convert transform coefficients back into stereo samples shaped (samples, 2)."""


def _validate_stereo(stereo: np.ndarray) -> np.ndarray:
    array = np.asarray(stereo, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"expected stereo array shaped (samples, 2), got {array.shape}")
    return array


class STFTTransform:
    def __init__(self, config: STFTConfig | None = None) -> None:
        self.config = config or STFTConfig()
        fft_size = self.config.fft_size or self.config.frame_size
        if self.config.frame_size <= 0:
            raise ValueError("v0 STFT requires frame_size to be positive")
        if self.config.hop_size <= 0:
            raise ValueError("v0 STFT requires hop_size to be positive")
        if fft_size != self.config.frame_size:
            raise ValueError("v0 STFT requires fft_size to equal frame_size")
        window = np.sqrt(np.hanning(self.config.frame_size))
        self._stft = ShortTimeFFT(
            win=window,
            hop=self.config.hop_size,
            fs=1.0,
            fft_mode="onesided",
            mfft=self.config.frame_size,
        )
        if not self._stft.invertible:
            raise ValueError(
                "v0 STFT requires an invertible frame_size/hop_size configuration"
            )

    def forward(self, stereo: np.ndarray) -> TransformRepresentation:
        stereo = _validate_stereo(stereo)
        if stereo.shape[0] < self.config.frame_size:
            raise ValueError(
                f"v0 STFT requires at least {self.config.frame_size} samples, "
                f"got {stereo.shape[0]}"
            )
        data = np.stack([self._stft.stft(channel) for channel in stereo.T], axis=0)
        return TransformRepresentation(
            data=data,
            original_shape=stereo.shape,
            metadata={
                "transform": "stft",
                "frame_size": self.config.frame_size,
                "hop_size": self.config.hop_size,
                "fft_size": self.config.frame_size,
                "window": "sqrt_hann",
                "experimental": False,
            },
        )

    def inverse(self, representation: TransformRepresentation) -> np.ndarray:
        channels = [
            self._stft.istft(representation.data[index], k1=representation.original_shape[0])
            for index in range(2)
        ]
        return np.column_stack(channels)

    def inverse_components(
        self, components: np.ndarray, original_shape: tuple[int, int]
    ) -> np.ndarray:
        representation = TransformRepresentation(
            data=components,
            original_shape=original_shape,
            metadata={
                "transform": "stft",
                "frame_size": self.config.frame_size,
                "hop_size": self.config.hop_size,
                "fft_size": self.config.frame_size,
                "window": "sqrt_hann",
                "experimental": False,
            },
        )
        return self.inverse(representation)


class FrFTTransform:
    def __init__(self, order: float = 1.0) -> None:
        self.order = order

    def forward(self, stereo: np.ndarray) -> TransformRepresentation:
        stereo = _validate_stereo(stereo)
        bins = fft(stereo.T, axis=-1)
        phase = np.exp(-0.5j * np.pi * self.order)
        data = bins * phase
        return TransformRepresentation(
            data=data,
            original_shape=stereo.shape,
            metadata={"transform": "frft", "order": self.order, "experimental": True},
        )

    def inverse(self, representation: TransformRepresentation) -> np.ndarray:
        raise NotImplementedError("FrFT full splitting is experimental and unsupported in v0")


class WaveletTransform:
    def __init__(self, wavelet: str = "db4", level: int = 3) -> None:
        self.wavelet = wavelet
        self.level = level

    def forward(self, stereo: np.ndarray) -> TransformRepresentation:
        stereo = _validate_stereo(stereo)
        coeff_arrays = []
        coeff_slices = []
        for channel in stereo.T:
            coeffs = pywt.wavedec(channel, self.wavelet, level=self.level, mode="periodization")
            coeff_array, slices = pywt.coeffs_to_array(coeffs)
            coeff_arrays.append(coeff_array)
            coeff_slices.append(slices)
        return TransformRepresentation(
            data=np.stack(coeff_arrays, axis=0),
            original_shape=stereo.shape,
            metadata={
                "transform": "wavelet",
                "wavelet": self.wavelet,
                "level": self.level,
                "coeff_slices": coeff_slices,
                "experimental": True,
            },
        )

    def inverse(self, representation: TransformRepresentation) -> np.ndarray:
        raise NotImplementedError("Wavelet full splitting is experimental and unsupported in v0")
