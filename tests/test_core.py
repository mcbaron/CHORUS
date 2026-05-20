from __future__ import annotations

import numpy as np
import pytest
from chorus.core import ChorusConfig, ChorusProcessor
from chorus.filters import normalize_filter_config


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


def test_processor_exposes_six_named_contributions(
    sine_440: np.ndarray, sample_rate: int
) -> None:
    stereo = np.column_stack([sine_440, 0.5 * sine_440])
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0))

    result = processor.process(stereo)

    assert set(result.contributions) == {"Lc", "Rc", "Lo", "Ro", "Ls", "Rs"}
    for contribution in result.contributions.values():
        assert contribution.shape == (stereo.shape[0],)
    np.testing.assert_allclose(
        result.center,
        np.column_stack([result.contributions["Lc"], result.contributions["Rc"]]),
    )
    np.testing.assert_allclose(
        result.only,
        np.column_stack([result.contributions["Lo"], result.contributions["Ro"]]),
    )
    np.testing.assert_allclose(
        result.surround,
        np.column_stack([result.contributions["Ls"], result.contributions["Rs"]]),
    )


def test_contribution_arrays_do_not_alias_stem_arrays(
    stereo_identical: np.ndarray, sample_rate: int
) -> None:
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0))
    result = processor.process(stereo_identical)
    original_center_left = result.center[0, 0]

    result.contributions["Lc"][0] = original_center_left + 1.0

    assert result.center[0, 0] == original_center_left


@pytest.mark.parametrize("transform_name", ["frft", "wavelet"])
def test_processor_can_run_reconstruction_capable_research_transforms(
    transform_name: str, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    processor = ChorusProcessor(
        ChorusConfig(sample_rate=sample_rate, transform=transform_name, smoothing_alpha=0.0)
    )

    result = processor.process(stereo_identical)

    assert result.metadata["transform"]["transform"] == transform_name
    reconstructed = result.center + result.only + result.surround
    np.testing.assert_allclose(reconstructed, stereo_identical, atol=1e-6, rtol=1e-6)


def test_unity_filter_chains_match_unfiltered_v1_output(
    stereo_identical: np.ndarray, sample_rate: int
) -> None:
    baseline = ChorusProcessor(
        ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0)
    ).process(stereo_identical)
    filtered = ChorusProcessor(
        ChorusConfig(
            sample_rate=sample_rate,
            smoothing_alpha=0.0,
            filter_chains=normalize_filter_config(None),
        )
    ).process(stereo_identical)

    np.testing.assert_allclose(filtered.center, baseline.center, atol=1e-10)
    np.testing.assert_allclose(filtered.only, baseline.only, atol=1e-10)
    np.testing.assert_allclose(filtered.surround, baseline.surround, atol=1e-10)
    assert filtered.metadata["filters"]["transparent"] is True


def test_gain_filter_changes_only_target_contribution(
    stereo_identical: np.ndarray, sample_rate: int
) -> None:
    chains = normalize_filter_config({"Lc": [{"type": "gain", "db": -6.0}]})
    result = ChorusProcessor(
        ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0, filter_chains=chains)
    ).process(stereo_identical)

    assert result.metadata["filters"]["transparent"] is False
    assert result.contributions["Lc"].shape == result.contributions["Rc"].shape
    assert not np.allclose(result.contributions["Lc"], result.contributions["Rc"])


def test_silence_is_stable(sample_rate: int) -> None:
    stereo = np.zeros((4096, 2), dtype=np.float64)
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate))

    result = processor.process(stereo)

    assert np.all(np.isfinite(result.center))
    assert np.all(np.isfinite(result.only))
    assert np.all(np.isfinite(result.surround))
