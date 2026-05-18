from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_rate() -> int:
    return 44_100


@pytest.fixture
def one_second_time(sample_rate: int) -> np.ndarray:
    return np.arange(sample_rate, dtype=np.float64) / sample_rate


@pytest.fixture
def sine_440(one_second_time: np.ndarray) -> np.ndarray:
    return 0.25 * np.sin(2.0 * np.pi * 440.0 * one_second_time)


@pytest.fixture
def stereo_identical(sine_440: np.ndarray) -> np.ndarray:
    return np.column_stack([sine_440, sine_440])
