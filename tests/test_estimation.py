from __future__ import annotations

import numpy as np
from chorus.estimation import SmoothedScalarEstimator


def test_estimator_recovers_unit_gain_for_matching_signal() -> None:
    estimator = SmoothedScalarEstimator(alpha=0.0, epsilon=1e-12)
    source = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    prototype = source.copy()

    estimated, weights = estimator.estimate(prototype, source)

    np.testing.assert_allclose(weights, np.ones_like(source, dtype=np.float64), atol=1e-12)
    np.testing.assert_allclose(estimated, source, atol=1e-12)


def test_estimator_is_stable_for_silence() -> None:
    estimator = SmoothedScalarEstimator(alpha=0.9, epsilon=1e-9)
    source = np.zeros((4,), dtype=np.complex128)
    prototype = np.zeros((4,), dtype=np.complex128)

    estimated, weights = estimator.estimate(prototype, source)

    assert np.all(np.isfinite(weights))
    assert np.all(np.isfinite(estimated))
    np.testing.assert_allclose(weights, np.zeros_like(weights))
    np.testing.assert_allclose(estimated, np.zeros_like(estimated))


def test_estimator_smooths_weight_changes() -> None:
    estimator = SmoothedScalarEstimator(alpha=0.5, epsilon=1e-12)
    source = np.array([1.0 + 0.0j])

    _, first = estimator.estimate(np.array([1.0 + 0.0j]), source)
    _, second = estimator.estimate(np.array([0.0 + 0.0j]), source)

    assert first[0] > second[0] > 0.0
