from __future__ import annotations

import numpy as np

from chorus.prototypes import center_prototype, surround_prototype


def test_center_prototype_keeps_equal_in_phase_content() -> None:
    left = np.array([1.0 + 0.0j, 0.5 + 0.0j])
    right = np.array([1.0 + 0.0j, 0.5 + 0.0j])

    center = center_prototype(left, right)

    np.testing.assert_allclose(center, left)


def test_center_prototype_rejects_hard_panned_content() -> None:
    left = np.array([1.0 + 0.0j])
    right = np.array([0.0 + 0.0j])

    center = center_prototype(left, right)

    np.testing.assert_allclose(center, np.array([0.0 + 0.0j]))


def test_center_prototype_rejects_phase_inverted_content() -> None:
    left = np.array([1.0 + 0.0j])
    right = np.array([-1.0 + 0.0j])

    center = center_prototype(left, right)

    np.testing.assert_allclose(center, np.array([0.0 + 0.0j]), atol=1e-12)


def test_surround_prototype_keeps_equal_out_of_phase_content() -> None:
    left = np.array([1.0 + 0.0j])
    right = np.array([-1.0 + 0.0j])

    surround = surround_prototype(left, right)

    np.testing.assert_allclose(surround, np.array([1.0 + 0.0j]))


def test_surround_prototype_rejects_equal_in_phase_content() -> None:
    left = np.array([1.0 + 0.0j])
    right = np.array([1.0 + 0.0j])

    surround = surround_prototype(left, right)

    np.testing.assert_allclose(surround, np.array([0.0 + 0.0j]), atol=1e-12)
