from __future__ import annotations

import numpy as np
from chorus.core import ChorusConfig, ChorusProcessor


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def test_processor_returns_three_stereo_stems(
    stereo_identical: np.ndarray, sample_rate: int
) -> None:
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate))

    result = processor.process(stereo_identical)

    assert result.center.shape == stereo_identical.shape
    assert result.only.shape == stereo_identical.shape
    assert result.surround.shape == stereo_identical.shape


def test_identical_stereo_routes_primarily_to_center(
    stereo_identical: np.ndarray, sample_rate: int
) -> None:
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0))

    result = processor.process(stereo_identical)

    assert _rms(result.center) > 10.0 * _rms(result.only)
    assert _rms(result.center) > 10.0 * max(_rms(result.surround), 1e-12)


def test_hard_panned_left_routes_primarily_to_only(
    sine_440: np.ndarray, sample_rate: int
) -> None:
    stereo = np.column_stack([sine_440, np.zeros_like(sine_440)])
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0))

    result = processor.process(stereo)

    assert _rms(result.only[:, 0]) > 10.0 * max(_rms(result.center), 1e-12)
    np.testing.assert_allclose(result.only[:, 1], np.zeros_like(sine_440), atol=1e-10)


def test_hard_panned_right_routes_primarily_to_only(
    sine_440: np.ndarray, sample_rate: int
) -> None:
    stereo = np.column_stack([np.zeros_like(sine_440), sine_440])
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0))

    result = processor.process(stereo)

    assert _rms(result.only[:, 1]) > 10.0 * max(_rms(result.center), 1e-12)
    np.testing.assert_allclose(result.only[:, 0], np.zeros_like(sine_440), atol=1e-10)


def test_phase_inverted_stereo_routes_primarily_to_surround(
    sine_440: np.ndarray, sample_rate: int
) -> None:
    stereo = np.column_stack([sine_440, -sine_440])
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0))

    result = processor.process(stereo)

    assert _rms(result.surround) > 10.0 * max(_rms(result.center), 1e-12)


def test_contributions_reconstruct_input(
    stereo_identical: np.ndarray, sample_rate: int
) -> None:
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0))

    result = processor.process(stereo_identical)
    reconstructed = result.center + result.only + result.surround

    np.testing.assert_allclose(reconstructed, stereo_identical, atol=1e-8, rtol=1e-8)


def test_silence_is_stable(sample_rate: int) -> None:
    stereo = np.zeros((4096, 2), dtype=np.float64)
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate))

    result = processor.process(stereo)

    assert np.all(np.isfinite(result.center))
    assert np.all(np.isfinite(result.only))
    assert np.all(np.isfinite(result.surround))
