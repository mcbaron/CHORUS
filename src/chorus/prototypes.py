from __future__ import annotations

import numpy as np


def _safe_unit_phase(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    return np.divide(values, magnitude, out=np.zeros_like(values), where=magnitude > 0.0)


def _matched_magnitude(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.minimum(np.abs(left), np.abs(right))


def center_prototype(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the local in-phase shared prototype for complex components."""
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    shared = _matched_magnitude(left, right)
    left_part = shared * _safe_unit_phase(left)
    right_part = shared * _safe_unit_phase(right)
    return 0.5 * (left_part + right_part)


def surround_prototype(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the local out-of-phase shared prototype for complex components."""
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    shared = _matched_magnitude(left, right)
    left_part = shared * _safe_unit_phase(left)
    right_part = shared * _safe_unit_phase(right)
    return 0.5 * (left_part - right_part)
