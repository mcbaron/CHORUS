from __future__ import annotations

import json

import numpy as np

from scripts.generate_v2_fixtures import build_fixture_set


def test_build_fixture_set_is_deterministic() -> None:
    first = build_fixture_set(sample_rate=48_000, samples=4096)
    second = build_fixture_set(sample_rate=48_000, samples=4096)

    assert first["manifest"] == second["manifest"]
    for name in first["arrays"]:
        np.testing.assert_allclose(first["arrays"][name], second["arrays"][name], atol=0.0, rtol=0.0)


def test_fixture_manifest_contains_required_cases() -> None:
    fixture_set = build_fixture_set(sample_rate=48_000, samples=4096)

    assert set(fixture_set["manifest"]["cases"]) == {
        "unity_bypass",
        "center_dominant",
        "hard_panned_left",
        "hard_panned_right",
        "phase_inverted_surround",
        "known_eq_preset",
        "silence",
        "near_silence",
    }
