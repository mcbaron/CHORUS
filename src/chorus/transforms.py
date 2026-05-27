from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pywt
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


def _frft_1d(x: np.ndarray, order: float) -> np.ndarray:
    n = len(x)
    if order == 0.0:
        return np.asarray(x, dtype=complex)
    if order == 1.0:
        return np.fft.fft(x)
    phi = order * np.pi / 2.0
    indices = np.arange(n, dtype=np.float64)
    cot_phi = np.cos(phi) / np.sin(phi)
    csc_phi = 1.0 / np.sin(phi)
    chirp = np.exp(-1j * np.pi * cot_phi * indices**2 / n)
    norm = np.sqrt((1.0 - 1j * cot_phi) / (n * abs(csc_phi)))
    return norm * np.fft.ifft(chirp * np.fft.fft(chirp * x))


def _ifrft_1d(y: np.ndarray, order: float) -> np.ndarray:
    n = len(y)
    if order == 0.0:
        return np.asarray(y, dtype=complex)
    if order == 1.0:
        return np.fft.ifft(y)
    phi = order * np.pi / 2.0
    indices = np.arange(n, dtype=np.float64)
    cot_phi = np.cos(phi) / np.sin(phi)
    csc_phi = 1.0 / np.sin(phi)
    chirp = np.exp(-1j * np.pi * cot_phi * indices**2 / n)
    norm = np.sqrt((1.0 - 1j * cot_phi) / (n * abs(csc_phi)))
    return (1.0 / norm) * np.conj(chirp) * np.fft.ifft(np.conj(chirp) * np.fft.fft(y))


class FrFTTransform:
    def __init__(self, order: float = 0.5, frame_size: int = 1024, hop_size: int | None = None) -> None:
        self.order = order
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else frame_size // 2
        if not (0 < self.hop_size <= frame_size // 2):
            raise ValueError(f"hop_size must be between 1 and frame_size // 2 ({frame_size // 2}), got {self.hop_size}")

    def forward(self, stereo: np.ndarray) -> TransformRepresentation:
        stereo = _validate_stereo(stereo)
        n_samples = stereo.shape[0]
        N = self.frame_size
        H = self.hop_size
        window = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N))

        n_frames = math.ceil(n_samples / H) + 1
        total_length = (n_frames - 1) * H + N

        padded = np.zeros((total_length, 2), dtype=np.float64)
        padded[H : H + n_samples, :] = stereo

        out = np.empty((2, n_frames, N), dtype=complex)
        for ch in range(2):
            for i in range(n_frames):
                start = i * H
                out[ch, i] = _frft_1d(window * padded[start : start + N, ch], self.order)

        return TransformRepresentation(
            data=out,
            original_shape=stereo.shape,
            metadata={
                "transform": "frft",
                "order": self.order,
                "frame_size": N,
                "hop_size": H,
                "window": "sqrt_hann",
                "experimental": True,
            },
        )

    def inverse(self, representation: TransformRepresentation) -> np.ndarray:
        n_samples = representation.original_shape[0]
        N = self.frame_size
        H = self.hop_size
        window = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N))

        n_frames = representation.data.shape[1]
        total_length = (n_frames - 1) * H + N

        buf = np.zeros((total_length, 2), dtype=np.float64)
        for ch in range(2):
            for i in range(n_frames):
                start = i * H
                frame = _ifrft_1d(representation.data[ch, i], self.order).real
                buf[start : start + N, ch] += window * frame

        return buf[H : H + n_samples, :]

    def inverse_components(
        self, components: np.ndarray, original_shape: tuple[int, int]
    ) -> np.ndarray:
        N = self.frame_size
        H = self.hop_size
        representation = TransformRepresentation(
            data=components,
            original_shape=original_shape,
            metadata={
                "transform": "frft",
                "order": self.order,
                "frame_size": N,
                "hop_size": H,
                "window": "sqrt_hann",
                "experimental": True,
            },
        )
        return self.inverse(representation)


class WaveletTransform:
    def __init__(self, wavelet: str = "db4", level: int = 3) -> None:
        self.wavelet = wavelet
        self.level = level

    def forward(self, stereo: np.ndarray) -> TransformRepresentation:
        stereo = _validate_stereo(stereo)
        coeff_arrays = []
        coeff_slices = []
        for channel in stereo.T:
            coeffs = pywt.wavedec(channel, self.wavelet, level=self.level, mode="reflect")
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
        coeff_slices = representation.metadata["coeff_slices"]
        channels = []
        for channel_data, slices in zip(representation.data, coeff_slices, strict=True):
            coeffs = pywt.array_to_coeffs(channel_data, slices, output_format="wavedec")
            reconstructed = pywt.waverec(coeffs, self.wavelet, mode="reflect")
            channels.append(reconstructed[: representation.original_shape[0]])
        return np.column_stack(channels)

    def inverse_components(
        self, components: np.ndarray, original_shape: tuple[int, int]
    ) -> np.ndarray:
        original = np.zeros(original_shape, dtype=np.float64)
        representation = self.forward(original)
        representation = TransformRepresentation(
            data=components,
            original_shape=original_shape,
            metadata=representation.metadata,
        )
        return self.inverse(representation)
