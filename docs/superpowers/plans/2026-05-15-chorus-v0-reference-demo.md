# CHORUS v0 Reference Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Python offline CHORUS v0 demo that splits a stereo WAV into center, only, and surround contribution stems using a patent-shaped shared DSP core.

**Architecture:** Create a small `src/chorus` package with a state-shaped offline core, a reference STFT transform, experimental FrFT/wavelet adapters, prototype generation, smoothed least-squares estimation, report generation, and a `chorus split` CLI. Keep file I/O outside the DSP core so the core can later inform a real-time plugin implementation.

**Tech Stack:** Python 3.12, Poetry, NumPy, SciPy, PyWavelets, pytest, Ruff.

---

## File Structure

- Create `pyproject.toml`: Poetry package metadata, dependencies, CLI entrypoint, pytest/Ruff config.
- Create `README.md`: minimal usage and v0 scope.
- Create `src/chorus/__init__.py`: package version export.
- Create `src/chorus/core.py`: config/result dataclasses and `ChorusProcessor`.
- Create `src/chorus/transforms.py`: transform protocol, `STFTTransform`, `FrFTTransform`, `WaveletTransform`.
- Create `src/chorus/prototypes.py`: center and surround prototype generation.
- Create `src/chorus/estimation.py`: smoothed least-squares coefficient estimator.
- Create `src/chorus/io.py`: WAV read/write, level metrics, report generation.
- Create `src/chorus/cli.py`: `chorus split` command.
- Create `tests/conftest.py`: deterministic audio fixtures.
- Create `tests/test_transforms.py`: STFT pass-through gate and experimental adapter smoke tests.
- Create `tests/test_prototypes.py`: prototype routing behavior.
- Create `tests/test_estimation.py`: coefficient estimation and silence stability.
- Create `tests/test_core.py`: three-stem output contract and synthetic routing fixtures.
- Create `tests/test_io_cli.py`: report and CLI integration tests.

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/chorus/__init__.py`
- Create: `src/chorus/cli.py`
- Create: `tests/conftest.py`

- [x] **Step 1: Write the package scaffold**

Create `pyproject.toml`:

```toml
[project]
name = "chorus"
version = "0.1.0"
description = "CHORUS reference demo for reconstructed upmixed stereo contributions"
readme = "README.md"
authors = [{ name = "CHORUS contributors" }]
requires-python = ">=3.12,<4.0"
dependencies = [
    "numpy>=1.26,<2.0",
    "scipy>=1.13,<2.0",
    "PyWavelets>=1.6,<2.0",
]

[project.scripts]
chorus = "chorus.cli:main"

[tool.poetry]
packages = [{ include = "chorus", from = "src" }]

[tool.poetry.group.dev.dependencies]
pytest = "^8.2"
ruff = "^0.4"

[build-system]
requires = ["poetry-core>=1.9.0"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Create `README.md`:

```markdown
# CHORUS

CHORUS Harmonizes Outputs from Reconstructed Upmixed Signals.

v0 is an offline Python reference demo. It reads stereo audio and writes three stereo stems:

- `center.wav`: `[Lc, Rc]`
- `only.wav`: `[Lo, Ro]`
- `surround.wav`: `[Ls, Rs]`

The reference transform is STFT. FrFT and wavelet adapters are experimental.
```

Create `src/chorus/__init__.py`:

```python
"""CHORUS reference demo package."""

__version__ = "0.1.0"
```

Create `src/chorus/cli.py`:

```python
from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chorus", description="CHORUS reference demo")
    parser.parse_args(argv)
    return 0
```

Create `tests/conftest.py`:

```python
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
```

- [x] **Step 2: Install dependencies**

Run:

```bash
poetry install
```

Expected: Poetry creates an environment, writes `poetry.lock`, and installs `numpy`, `scipy`, `PyWavelets`, `pytest`, and `ruff`.

- [x] **Step 3: Run the empty test suite**

Run:

```bash
poetry run chorus --help
poetry run pytest -q
```

Expected: `poetry run chorus --help` exits 0 and prints argparse help. Pytest exits with code 5 and prints `no tests ran`. This confirms the scaffold imports before tests are added.

- [x] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock README.md src/chorus/__init__.py src/chorus/cli.py tests/conftest.py
git commit -m "chore: scaffold chorus python package"
```

## Task 2: Reference STFT Transform

**Files:**
- Create: `src/chorus/transforms.py`
- Test: `tests/test_transforms.py`

- [x] **Step 1: Write failing STFT pass-through tests**

Create `tests/test_transforms.py`:

```python
from __future__ import annotations

import numpy as np

from chorus.transforms import FrFTTransform, STFTConfig, STFTTransform, WaveletTransform


def test_stft_round_trip_pass_through_stereo(stereo_identical: np.ndarray) -> None:
    transform = STFTTransform(STFTConfig(frame_size=1024, hop_size=512))

    representation = transform.forward(stereo_identical)
    reconstructed = transform.inverse(representation)

    assert reconstructed.shape == stereo_identical.shape
    np.testing.assert_allclose(reconstructed, stereo_identical, atol=1e-10, rtol=1e-10)


def test_stft_metadata_records_reference_settings(stereo_identical: np.ndarray) -> None:
    transform = STFTTransform(STFTConfig(frame_size=1024, hop_size=512))

    representation = transform.forward(stereo_identical)

    assert representation.metadata["transform"] == "stft"
    assert representation.metadata["frame_size"] == 1024
    assert representation.metadata["hop_size"] == 512
    assert representation.metadata["window"] == "sqrt_hann"


def test_experimental_frft_adapter_smoke(stereo_identical: np.ndarray) -> None:
    transform = FrFTTransform(order=0.75)

    representation = transform.forward(stereo_identical)

    assert representation.data.shape[0] == 2
    assert representation.metadata["experimental"] is True
    assert representation.metadata["transform"] == "frft"


def test_experimental_wavelet_adapter_smoke(stereo_identical: np.ndarray) -> None:
    transform = WaveletTransform(wavelet="db4", level=3)

    representation = transform.forward(stereo_identical)

    assert representation.data.shape[0] == 2
    assert representation.metadata["experimental"] is True
    assert representation.metadata["transform"] == "wavelet"
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
poetry run pytest tests/test_transforms.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'chorus.transforms'`.

- [x] **Step 3: Implement transform classes**

Create `src/chorus/transforms.py`:

```python
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

    def forward(self, stereo: np.ndarray) -> TransformRepresentation:
        stereo = _validate_stereo(stereo)
        channel_first = stereo.T
        data = np.stack([self._stft.stft(channel) for channel in channel_first], axis=0)
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


class FrFTTransform:
    def __init__(self, order: float = 1.0) -> None:
        self.order = order

    def forward(self, stereo: np.ndarray) -> TransformRepresentation:
        stereo = _validate_stereo(stereo)
        # v0 smoke adapter: phase-rotate FFT bins by fractional order. This is not the
        # final reference FrFT reconstruction path.
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
```

- [x] **Step 4: Run transform tests**

Run:

```bash
poetry run pytest tests/test_transforms.py -q
```

Expected: PASS. If the STFT round trip fails because `ShortTimeFFT` boundary behavior differs, adjust only `STFTTransform.forward` and `STFTTransform.inverse` until this pass-through test meets `1e-10`; do not loosen the test first.

- [x] **Step 5: Commit**

```bash
git add src/chorus/transforms.py tests/test_transforms.py
git commit -m "feat: add reference stft transform"
```

## Task 3: Prototype Generation

**Files:**
- Create: `src/chorus/prototypes.py`
- Test: `tests/test_prototypes.py`

- [x] **Step 1: Write failing prototype tests**

Create `tests/test_prototypes.py`:

```python
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
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
poetry run pytest tests/test_prototypes.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'chorus.prototypes'`.

- [x] **Step 3: Implement prototype functions**

Create `src/chorus/prototypes.py`:

```python
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
```

- [x] **Step 4: Run prototype tests**

Run:

```bash
poetry run pytest tests/test_prototypes.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/chorus/prototypes.py tests/test_prototypes.py
git commit -m "feat: add stereo prototype generation"
```

## Task 4: Smoothed Least-Squares Estimation

**Files:**
- Create: `src/chorus/estimation.py`
- Test: `tests/test_estimation.py`

- [x] **Step 1: Write failing estimation tests**

Create `tests/test_estimation.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

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


def test_estimator_uses_epsilon_as_denominator_floor() -> None:
    estimator = SmoothedScalarEstimator(alpha=0.0, epsilon=0.1)
    source = np.array([1.0 + 0.0j])
    prototype = source.copy()

    estimated, weights = estimator.estimate(prototype, source)

    np.testing.assert_allclose(weights, np.array([1.0]))
    np.testing.assert_allclose(estimated, source)


def test_estimator_smooths_weight_changes() -> None:
    estimator = SmoothedScalarEstimator(alpha=0.5, epsilon=1e-12)
    source = np.array([1.0 + 0.0j])

    _, first = estimator.estimate(np.array([1.0 + 0.0j]), source)
    _, second = estimator.estimate(np.array([0.0 + 0.0j]), source)

    assert first[0] > second[0] > 0.0


def test_estimator_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\)"):
        SmoothedScalarEstimator(alpha=1.0)


def test_estimator_rejects_invalid_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon must be positive"):
        SmoothedScalarEstimator(epsilon=0.0)


def test_estimator_rejects_shape_mismatch() -> None:
    estimator = SmoothedScalarEstimator()

    with pytest.raises(ValueError, match=r"prototype shape \(2,\) != source shape \(1,\)"):
        estimator.estimate(
            np.array([1.0 + 0.0j, 2.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
        )
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
poetry run pytest tests/test_estimation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'chorus.estimation'`.

- [x] **Step 3: Implement scalar estimator**

Create `src/chorus/estimation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SmoothedScalarEstimator:
    alpha: float = 0.9
    epsilon: float = 1e-9
    _cross: np.ndarray | None = field(default=None, init=False, repr=False)
    _auto: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    def estimate(self, prototype: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prototype = np.asarray(prototype, dtype=np.complex128)
        source = np.asarray(source, dtype=np.complex128)
        if prototype.shape != source.shape:
            raise ValueError(f"prototype shape {prototype.shape} != source shape {source.shape}")

        instant_cross = prototype * np.conjugate(source)
        instant_auto = np.abs(source) ** 2

        if self._cross is None:
            self._cross = instant_cross
            self._auto = instant_auto
        else:
            self._cross = (1.0 - self.alpha) * instant_cross + self.alpha * self._cross
            self._auto = (1.0 - self.alpha) * instant_auto + self.alpha * self._auto

        weights = np.real(self._cross) / np.maximum(self._auto, self.epsilon)
        estimated = weights * source
        return estimated, weights
```

- [x] **Step 4: Run estimation tests**

Run:

```bash
poetry run pytest tests/test_estimation.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/chorus/estimation.py tests/test_estimation.py
git commit -m "feat: add smoothed least squares estimator"
```

## Task 5: Core Split Processor

**Files:**
- Create: `src/chorus/core.py`
- Modify: `src/chorus/transforms.py`
- Test: `tests/test_core.py`

- [x] **Step 1: Write failing core tests**

Create `tests/test_core.py`:

```python
from __future__ import annotations

import numpy as np

from chorus.core import ChorusConfig, ChorusProcessor


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def test_processor_returns_three_stereo_stems(stereo_identical: np.ndarray, sample_rate: int) -> None:
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


def test_hard_panned_left_routes_primarily_to_only(sine_440: np.ndarray, sample_rate: int) -> None:
    stereo = np.column_stack([sine_440, np.zeros_like(sine_440)])
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0))

    result = processor.process(stereo)

    assert _rms(result.only[:, 0]) > 10.0 * max(_rms(result.center), 1e-12)
    np.testing.assert_allclose(result.only[:, 1], np.zeros_like(sine_440), atol=1e-10)


def test_hard_panned_right_routes_primarily_to_only(sine_440: np.ndarray, sample_rate: int) -> None:
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


def test_contributions_reconstruct_input(stereo_identical: np.ndarray, sample_rate: int) -> None:
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
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
poetry run pytest tests/test_core.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'chorus.core'`.

- [x] **Step 3: Add inverse helpers for component arrays**

Modify `src/chorus/transforms.py` by adding this method to `STFTTransform`:

```python
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
```

- [x] **Step 4: Implement core processor**

Create `src/chorus/core.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chorus.estimation import SmoothedScalarEstimator
from chorus.prototypes import center_prototype, surround_prototype
from chorus.transforms import STFTConfig, STFTTransform


@dataclass(frozen=True)
class ChorusConfig:
    sample_rate: int
    transform: str = "stft"
    frame_size: int = 1024
    hop_size: int = 512
    smoothing_alpha: float = 0.9
    epsilon: float = 1e-9


@dataclass(frozen=True)
class ChorusResult:
    center: np.ndarray
    only: np.ndarray
    surround: np.ndarray
    metadata: dict[str, object]


class ChorusProcessor:
    def __init__(self, config: ChorusConfig) -> None:
        if config.transform != "stft":
            raise NotImplementedError("full splitting is supported only for STFT in v0")
        self.config = config
        self.transform = STFTTransform(STFTConfig(config.frame_size, config.hop_size))

    def process(self, stereo: np.ndarray) -> ChorusResult:
        representation = self.transform.forward(stereo)
        left = representation.data[0]
        right = representation.data[1]

        center_proto = center_prototype(left, right)
        surround_proto = surround_prototype(left, right)

        lc_components = self._estimate_channel(center_proto, left)
        rc_components = self._estimate_channel(center_proto, right)
        ls_components = self._estimate_channel(surround_proto, left)
        rs_components = -self._estimate_channel(surround_proto, right)

        center_components = np.stack([lc_components, rc_components], axis=0)
        surround_components = np.stack([ls_components, rs_components], axis=0)
        input_components = np.stack([left, right], axis=0)
        only_components = input_components - center_components - surround_components

        center = self.transform.inverse_components(center_components, representation.original_shape)
        surround = self.transform.inverse_components(surround_components, representation.original_shape)
        only = self.transform.inverse_components(only_components, representation.original_shape)

        return ChorusResult(
            center=center,
            only=only,
            surround=surround,
            metadata={
                "sample_rate": self.config.sample_rate,
                "transform": representation.metadata,
                "smoothing_alpha": self.config.smoothing_alpha,
                "epsilon": self.config.epsilon,
            },
        )

    def _estimate_channel(self, prototype: np.ndarray, source: np.ndarray) -> np.ndarray:
        estimator = SmoothedScalarEstimator(
            alpha=self.config.smoothing_alpha,
            epsilon=self.config.epsilon,
        )
        output = np.empty_like(source)
        for frame_index in range(source.shape[-1]):
            output[..., frame_index], _weights = estimator.estimate(
                prototype[..., frame_index],
                source[..., frame_index],
            )
        return output
```

- [x] **Step 5: Run core tests**

Run:

```bash
poetry run pytest tests/test_core.py -q
```

Expected: PASS.

- [x] **Step 6: Run transform and core tests together**

Run:

```bash
poetry run pytest tests/test_transforms.py tests/test_core.py -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add src/chorus/core.py src/chorus/transforms.py tests/test_core.py
git commit -m "feat: add chorus split processor"
```

## Task 6: WAV I/O And Report Generation

**Files:**
- Create: `src/chorus/io.py`
- Test: `tests/test_io_cli.py`

- [x] **Step 1: Write failing I/O and report tests**

Create `tests/test_io_cli.py`:

```python
from __future__ import annotations

import json

import numpy as np
from scipy.io import wavfile

from chorus.core import ChorusConfig, ChorusProcessor
from chorus.io import build_report, read_stereo_wav, write_stems


def test_wav_read_write_and_report(tmp_path, stereo_identical: np.ndarray, sample_rate: int) -> None:
    input_path = tmp_path / "input.wav"
    out_dir = tmp_path / "out"
    wavfile.write(input_path, sample_rate, stereo_identical.astype(np.float32))

    loaded_rate, loaded = read_stereo_wav(input_path)
    processor = ChorusProcessor(ChorusConfig(sample_rate=loaded_rate, smoothing_alpha=0.0))
    result = processor.process(loaded)

    stem_paths = write_stems(out_dir, sample_rate, result)
    report = build_report(
        input_path=input_path,
        output_dir=out_dir,
        sample_rate=sample_rate,
        input_audio=loaded,
        result=result,
        stem_paths=stem_paths,
    )

    assert set(stem_paths) == {"center", "only", "surround"}
    assert (out_dir / "center.wav").exists()
    assert (out_dir / "only.wav").exists()
    assert (out_dir / "surround.wav").exists()
    assert report["checks"]["reconstruction"]["passed"] is True
    assert report["levels"]["center"]["rms"] > report["levels"]["only"]["rms"]

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    loaded_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded_report["input"]["sample_rate"] == sample_rate
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
poetry run pytest tests/test_io_cli.py::test_wav_read_write_and_report -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'chorus.io'`.

- [x] **Step 3: Implement I/O helpers**

Create `src/chorus/io.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from chorus.core import ChorusResult


def read_stereo_wav(path: str | Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    array = np.asarray(data)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"expected stereo WAV shaped (samples, 2), got {array.shape}")
    if np.issubdtype(array.dtype, np.unsignedinteger):
        info = np.iinfo(array.dtype)
        midpoint = float(info.max + 1) / 2.0
        array = (array.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(array.dtype, np.signedinteger):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        array = array.astype(np.float64) / scale
    else:
        array = array.astype(np.float64)
    return int(sample_rate), array


def write_stems(
    output_dir: str | Path,
    sample_rate: int,
    result: ChorusResult,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stems = {
        "center": out / "center.wav",
        "only": out / "only.wav",
        "surround": out / "surround.wav",
    }
    for name, path in stems.items():
        audio = getattr(result, name)
        wavfile.write(path, sample_rate, np.asarray(audio, dtype=np.float32))
    return stems


def _level_metrics(audio: np.ndarray) -> dict[str, float]:
    return {
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
    }


def _residual_metrics(input_audio: np.ndarray, result: ChorusResult) -> dict[str, object]:
    reconstructed = result.center + result.only + result.surround
    residual = input_audio - reconstructed
    left = residual[:, 0]
    right = residual[:, 1]
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0
    return {
        "passed": bool(max_abs <= 1e-6),
        "max_abs": max_abs,
        "rms": rms,
        "left": {
            "max_abs": float(np.max(np.abs(left))) if left.size else 0.0,
            "rms": float(np.sqrt(np.mean(np.square(left)))) if left.size else 0.0,
        },
        "right": {
            "max_abs": float(np.max(np.abs(right))) if right.size else 0.0,
            "rms": float(np.sqrt(np.mean(np.square(right)))) if right.size else 0.0,
        },
    }


def build_report(
    input_path: str | Path,
    output_dir: str | Path,
    sample_rate: int,
    input_audio: np.ndarray,
    result: ChorusResult,
    stem_paths: dict[str, Path],
) -> dict[str, object]:
    duration_seconds = float(input_audio.shape[0] / sample_rate)
    return {
        "input": {
            "path": str(input_path),
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "channels": 2,
        },
        "output": {
            "directory": str(output_dir),
            "stems": {name: str(path) for name, path in stem_paths.items()},
        },
        "settings": result.metadata,
        "levels": {
            "center": _level_metrics(result.center),
            "only": _level_metrics(result.only),
            "surround": _level_metrics(result.surround),
        },
        "checks": {
            "reconstruction": _residual_metrics(input_audio, result),
        },
    }
```

- [x] **Step 4: Run I/O test**

Run:

```bash
poetry run pytest tests/test_io_cli.py::test_wav_read_write_and_report -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/chorus/io.py tests/test_io_cli.py
git commit -m "feat: add wav io and reports"
```

## Task 7: CLI

**Files:**
- Modify: `src/chorus/cli.py`
- Modify: `tests/test_io_cli.py`

- [x] **Step 1: Add failing CLI test**

Add `from chorus.cli import main` to the import block near the top of `tests/test_io_cli.py`, then append this test:

```python
def test_cli_split_writes_expected_outputs(tmp_path, stereo_identical: np.ndarray, sample_rate: int) -> None:
    input_path = tmp_path / "input.wav"
    out_dir = tmp_path / "out"
    wavfile.write(input_path, sample_rate, stereo_identical.astype(np.float32))

    exit_code = main(["split", str(input_path), "--out-dir", str(out_dir), "--transform", "stft"])

    assert exit_code == 0
    assert (out_dir / "center.wav").exists()
    assert (out_dir / "only.wav").exists()
    assert (out_dir / "surround.wav").exists()
    assert (out_dir / "report.json").exists()
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
poetry run pytest tests/test_io_cli.py::test_cli_split_writes_expected_outputs -q
```

Expected: FAIL because the placeholder CLI does not recognize the `split` command.

- [x] **Step 3: Implement CLI**

Create `src/chorus/cli.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chorus.core import ChorusConfig, ChorusProcessor
from chorus.io import build_report, read_stereo_wav, write_stems


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chorus")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split", help="split stereo WAV into CHORUS stems")
    split.add_argument("input", type=Path)
    split.add_argument("--out-dir", type=Path, required=True)
    split.add_argument("--transform", choices=["stft", "frft", "wavelet"], default="stft")
    split.add_argument("--frame-size", type=int, default=1024)
    split.add_argument("--hop-size", type=int, default=512)
    split.add_argument("--smoothing-alpha", type=float, default=0.9)
    split.add_argument("--epsilon", type=float, default=1e-9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "split":
        sample_rate, audio = read_stereo_wav(args.input)
        config = ChorusConfig(
            sample_rate=sample_rate,
            transform=args.transform,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            smoothing_alpha=args.smoothing_alpha,
            epsilon=args.epsilon,
        )
        processor = ChorusProcessor(config)
        result = processor.process(audio)
        stem_paths = write_stems(args.out_dir, sample_rate, result)
        report = build_report(args.input, args.out_dir, sample_rate, audio, result, stem_paths)
        report_path = args.out_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    parser.error(f"unsupported command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 4: Run CLI test**

Run:

```bash
poetry run pytest tests/test_io_cli.py::test_cli_split_writes_expected_outputs -q
```

Expected: PASS.

- [x] **Step 5: Run full I/O/CLI tests**

Run:

```bash
poetry run pytest tests/test_io_cli.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/chorus/cli.py tests/test_io_cli.py
git commit -m "feat: add chorus split cli"
```

## Task 8: Full Verification And Documentation Polish

**Files:**
- Modify: `README.md`
- Inspect: `src/chorus/*.py`
- Inspect: `tests/*.py`

- [x] **Step 1: Update README usage**

Replace `README.md` with:

````markdown
# CHORUS

CHORUS Harmonizes Outputs from Reconstructed Upmixed Signals.

v0 is an offline Python reference demo for stereo contribution splitting. It reads a stereo WAV `(L, R)` and writes:

- `center.wav`: `[Lc, Rc]`
- `only.wav`: `[Lo, Ro]`
- `surround.wav`: `[Ls, Rs]`
- `report.json`: settings, levels, and reconstruction checks

## Install

```bash
poetry install
```

## Run

```bash
poetry run chorus split input.wav --out-dir out --transform stft
```

STFT is the reference path. FrFT and wavelet adapters are experimental in v0 and are not supported for full splitting until their inverse/reconstruction behavior is promoted to reference quality.

## Acceptance Gate

The STFT pass-through test is mandatory. If unchanged stereo cannot round-trip through the STFT analyzer/reconstructor within tolerance, the splitter is not considered valid.
````

- [x] **Step 2: Run the full test suite**

Run:

```bash
poetry run pytest -q
```

Expected: all tests PASS.

- [x] **Step 3: Run Ruff**

Run:

```bash
poetry run ruff check .
```

Expected: PASS. If Ruff reports import ordering, run `poetry run ruff check . --fix`, inspect the diff, then rerun `poetry run ruff check .`.

- [x] **Step 4: Run the CLI against a generated fixture**

Run:

```bash
poetry run python - <<'PY'
from pathlib import Path
import numpy as np
from scipy.io import wavfile

sample_rate = 44100
t = np.arange(sample_rate, dtype=np.float64) / sample_rate
tone = 0.25 * np.sin(2 * np.pi * 440 * t)
audio = np.column_stack([tone, tone]).astype(np.float32)
Path("tmp").mkdir(exist_ok=True)
wavfile.write("tmp/identical.wav", sample_rate, audio)
PY
poetry run chorus split tmp/identical.wav --out-dir tmp/chorus-out --transform stft
```

Expected: `tmp/chorus-out/center.wav`, `tmp/chorus-out/only.wav`, `tmp/chorus-out/surround.wav`, and `tmp/chorus-out/report.json` exist.

- [x] **Step 5: Inspect report reconstruction status**

Run:

```bash
poetry run python - <<'PY'
import json

with open("tmp/chorus-out/report.json", encoding="utf-8") as handle:
    report = json.load(handle)
print(report["checks"]["reconstruction"])
assert report["checks"]["reconstruction"]["passed"] is True
PY
```

Expected: printed reconstruction metrics with `"passed": True`.

- [x] **Step 6: Commit final polish**

```bash
git add README.md src/chorus tests
git commit -m "docs: document chorus v0 usage"
```

## Final Verification

Run:

```bash
poetry run pytest -q
poetry run ruff check .
git status --short
```

Expected:

- all tests pass
- Ruff passes
- `git status --short` shows only intentional untracked demo artifacts under `tmp/`, or a clean tree if `tmp/` is removed after verification

If `tmp/` is created during verification, delete it before final handoff:

```bash
rm -rf tmp
git status --short
```

Expected: clean working tree.

## Plan Self-Review Notes

- Spec coverage: scaffold, STFT reference transform, experimental FrFT/wavelet adapters, prototype generation, smoothed least-squares estimation, three stereo stems, report generation, CLI, and required synthetic tests are covered.
- Pass-through gate: Task 2 places STFT round-trip reconstruction before contribution math and forbids loosening the tolerance first.
- Plugin path: the plan keeps core processing separate from file I/O and uses config/result boundaries that can inform a future real-time implementation.
