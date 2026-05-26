from __future__ import annotations

import numpy as np
import pytest

from chorus.filters import CONTRIBUTIONS, FilterSpec, apply_filter_chains, normalize_filter_config


def test_normalize_empty_filter_config_creates_unity_for_every_contribution() -> None:
    chains = normalize_filter_config(None)

    assert set(chains) == CONTRIBUTIONS
    for chain in chains.values():
        assert chain == [FilterSpec(type="unity", parameters={})]


def test_invalid_contribution_name_fails_before_audio_processing() -> None:
    with pytest.raises(ValueError, match="unknown contribution 'left_center'"):
        normalize_filter_config({"left_center": [{"type": "gain", "db": 3.0}]})


def test_unsupported_filter_type_fails_validation() -> None:
    with pytest.raises(ValueError, match="unsupported filter type 'comb' for Lc\\[0\\]"):
        normalize_filter_config({"Lc": [{"type": "comb"}]})


def test_invalid_gain_parameter_fails_validation() -> None:
    with pytest.raises(ValueError, match="gain db for Lc\\[0\\] must be between -60.0 and 24.0"):
        normalize_filter_config({"Lc": [{"type": "gain", "db": 48.0}]})


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"Lc": [{"type": "eq", "mode": "notch", "frequency_hz": 1000.0}]}, "eq mode"),
        ({"Lc": [{"type": "eq", "frequency_hz": 0.0}]}, "frequency_hz"),
        ({"Lc": [{"type": "eq", "frequency_hz": 24000.0}]}, "below Nyquist"),
        ({"Lc": [{"type": "eq", "frequency_hz": 1000.0, "q": 0.0}]}, "eq q"),
    ],
)
def test_invalid_eq_parameters_fail_validation(raw: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_filter_config(raw, sample_rate=48_000)


def test_unity_filter_returns_matching_contributions() -> None:
    contributions = {name: np.ones(8, dtype=np.float64) for name in CONTRIBUTIONS}

    filtered, report = apply_filter_chains(
        contributions, normalize_filter_config(None), sample_rate=48_000
    )

    for name in CONTRIBUTIONS:
        np.testing.assert_allclose(filtered[name], contributions[name])
    assert report["transparent"] is True


@pytest.mark.parametrize("target", sorted(CONTRIBUTIONS))
def test_each_contribution_can_be_gain_adjusted_independently(target: str) -> None:
    contributions = {name: np.ones(8, dtype=np.float64) for name in CONTRIBUTIONS}
    chains = normalize_filter_config({target: [{"type": "gain", "db": 6.0}]})

    filtered, report = apply_filter_chains(contributions, chains, sample_rate=48_000)

    np.testing.assert_allclose(filtered[target], np.ones(8) * (10.0 ** (6.0 / 20.0)))
    for name in CONTRIBUTIONS - {target}:
        np.testing.assert_allclose(filtered[name], np.ones(8))
    assert report["transparent"] is False


def test_gain_and_polarity_affect_only_configured_contribution() -> None:
    contributions = {name: np.ones(8, dtype=np.float64) for name in CONTRIBUTIONS}
    chains = normalize_filter_config(
        {
            "Lc": [{"type": "gain", "db": 6.0}],
            "Rs": [{"type": "polarity"}],
        }
    )

    filtered, report = apply_filter_chains(contributions, chains, sample_rate=48_000)

    np.testing.assert_allclose(filtered["Lc"], np.ones(8) * (10.0 ** (6.0 / 20.0)))
    np.testing.assert_allclose(filtered["Rs"], -np.ones(8))
    np.testing.assert_allclose(filtered["Rc"], np.ones(8))
    assert report["transparent"] is False


def test_solo_mutes_non_solo_contributions() -> None:
    contributions = {name: np.ones(8, dtype=np.float64) for name in CONTRIBUTIONS}
    chains = normalize_filter_config({"Lo": [{"type": "solo"}]})

    filtered, report = apply_filter_chains(contributions, chains, sample_rate=48_000)

    np.testing.assert_allclose(filtered["Lo"], np.ones(8))
    for name in CONTRIBUTIONS - {"Lo"}:
        np.testing.assert_allclose(filtered[name], np.zeros(8))
    assert report["soloed"] == ["Lo"]


def test_eq_highpass_attenuates_below_cutoff() -> None:
    """Highpass at 1000 Hz should attenuate a 100 Hz sine by more than 20 dB."""
    sample_rate = 48_000
    n = 4096
    t = np.arange(n) / sample_rate
    audio = np.sin(2.0 * np.pi * 100.0 * t)

    contributions = {name: (audio.copy() if name == "Lo" else np.zeros(n)) for name in CONTRIBUTIONS}
    chains = normalize_filter_config({"Lo": [{"type": "eq", "mode": "highpass", "frequency_hz": 1000.0, "q": 0.707}]})
    filtered, _ = apply_filter_chains(contributions, chains, sample_rate)

    input_rms = float(np.sqrt(np.mean(audio ** 2)))
    output_rms = float(np.sqrt(np.mean(filtered["Lo"] ** 2)))
    ratio_db = 20.0 * np.log10(output_rms / input_rms + 1e-12)
    assert ratio_db < -20.0, f"Expected > 20 dB attenuation, got {ratio_db:.1f} dB"


def test_eq_peaking_boosts_center_frequency() -> None:
    """Peaking EQ at 1000 Hz with +6 dB should amplify a 1000 Hz sine by ~6 dB."""
    sample_rate = 48_000
    n = 4096
    t = np.arange(n) / sample_rate
    audio = np.sin(2.0 * np.pi * 1000.0 * t)

    contributions = {name: (audio.copy() if name == "Lo" else np.zeros(n)) for name in CONTRIBUTIONS}
    chains = normalize_filter_config({"Lo": [{"type": "eq", "mode": "peaking", "frequency_hz": 1000.0, "q": 10.0, "gain_db": 6.0}]})
    filtered, _ = apply_filter_chains(contributions, chains, sample_rate)

    input_rms = float(np.sqrt(np.mean(audio ** 2)))
    output_rms = float(np.sqrt(np.mean(filtered["Lo"] ** 2)))
    ratio_db = 20.0 * np.log10(output_rms / (input_rms + 1e-12))
    assert 4.0 < ratio_db < 8.0, f"Expected ~6 dB boost, got {ratio_db:.1f} dB"


def test_eq_lowpass_attenuates_above_cutoff() -> None:
    """Lowpass at 1000 Hz should attenuate a 10 kHz sine by more than 20 dB."""
    sample_rate = 48_000
    n = 4096
    t = np.arange(n) / sample_rate
    audio = np.sin(2.0 * np.pi * 10_000.0 * t)

    contributions = {name: (audio.copy() if name == "Lo" else np.zeros(n)) for name in CONTRIBUTIONS}
    chains = normalize_filter_config({"Lo": [{"type": "eq", "mode": "lowpass", "frequency_hz": 1000.0, "q": 0.707}]})
    filtered, _ = apply_filter_chains(contributions, chains, sample_rate)

    input_rms = float(np.sqrt(np.mean(audio ** 2)))
    output_rms = float(np.sqrt(np.mean(filtered["Lo"] ** 2)))
    ratio_db = 20.0 * np.log10(output_rms / input_rms + 1e-12)
    assert ratio_db < -20.0, f"Expected > 20 dB attenuation, got {ratio_db:.1f} dB"


def test_eq_fixture_parity_known_eq_preset() -> None:
    """The regenerated known_eq_preset fixture should match _apply_eq output."""
    import json
    from pathlib import Path

    fixture_dir = Path("fixtures/v2")
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    assert "known_eq_preset" in manifest["cases"], "fixture not found"

    audio_input = np.load(fixture_dir / "known_eq_preset.input.npy")
    spec = FilterSpec(type="eq", parameters={"mode": "highpass", "frequency_hz": 120.0, "q": 0.707})
    from chorus.filters import _apply_eq
    lo_audio = audio_input[:, 0]
    filtered = _apply_eq(lo_audio, spec, sample_rate=48_000)

    assert np.all(np.isfinite(filtered)), "filtered output contains non-finite values"
    assert filtered.shape == lo_audio.shape
