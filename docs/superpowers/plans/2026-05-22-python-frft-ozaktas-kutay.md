# Python FrFT Ozaktas-Kutay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python `FrFTTransform` phase-shifted FFT approximation with the true Ozaktas-Kutay 2-chirp algorithm, add `frft_order` and `frame_size` to `ChorusConfig`, generate FrFT fixtures, and un-ignore the Rust FrFT fixture test.

**Architecture:** A private `_frft_1d` helper in `transforms.py` implements the Ozaktas-Kutay formula; `FrFTTransform` uses it to process non-overlapping frames of size `frame_size`, matching Rust's `StreamingFrft`. `ChorusConfig` gains `frft_order: float = 0.5`, and `ChorusProcessor` passes it through. The Rust fixture test in `fixtures.rs` is un-ignored and wired up with the newly generated `frft_unity_bypass` fixture files.

**Tech Stack:** Python 3, NumPy (`np.fft.fft/ifft`), pytest, soundfile, poetry; Rust (no source changes — fixture test only, run via `run-rust-remote` skill)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/chorus/transforms.py` | Modify | Add `_frft_1d` helper; rewrite `FrFTTransform` constructor, `forward`, `inverse`, `inverse_components` |
| `src/chorus/core.py` | Modify | Add `frft_order: float = 0.5` to `ChorusConfig`; wire `FrFTTransform(order=config.frft_order, frame_size=config.frame_size)` in `ChorusProcessor` |
| `tests/test_transforms.py` | Modify | Add round-trip test, WAV round-trip tests, and special-case tests (`order=0.0`, `order=1.0`) for the new `FrFTTransform` |
| `scripts/generate_v2_fixtures.py` | Modify | Add `frft_unity_bypass` case using `ChorusConfig(transform="frft", frft_order=0.5, frame_size=1024)` |
| `rust/chorus-dsp/tests/fixtures.rs` | Modify | Un-ignore `rust_matches_python_frft_fixture` and implement it |

---

## Task 1: Add `_frft_1d` helper and rewrite `FrFTTransform`

**Files:**
- Modify: `src/chorus/transforms.py`

- [ ] **Step 1: Write the failing round-trip test**

In `tests/test_transforms.py`, add these tests after the existing `test_frft_round_trip_pass_through_stereo` test. These will fail because `FrFTTransform` does not yet accept `frame_size` and the algorithm is wrong.

```python
def test_frft_ozaktas_round_trip_synthetic() -> None:
    """New Ozaktas-Kutay FrFT must round-trip a synthetic signal within 1e-10."""
    rng = np.random.default_rng(42)
    audio = rng.standard_normal((4096, 2))
    t = FrFTTransform(order=0.5, frame_size=1024)
    rep = t.forward(audio)
    recovered = t.inverse(rep)
    np.testing.assert_allclose(recovered, audio, atol=1e-10)


def test_frft_identity_order_zero() -> None:
    """order=0.0 forward must return input unchanged."""
    rng = np.random.default_rng(7)
    audio = rng.standard_normal((1024, 2))
    t = FrFTTransform(order=0.0, frame_size=1024)
    rep = t.forward(audio)
    recovered = t.inverse(rep)
    np.testing.assert_allclose(recovered, audio, atol=1e-12)


def test_frft_order_one_matches_fft() -> None:
    """order=1.0 forward must match np.fft.fft per frame per channel."""
    rng = np.random.default_rng(13)
    audio = rng.standard_normal((1024, 2))
    t = FrFTTransform(order=1.0, frame_size=1024)
    rep = t.forward(audio)
    # Single frame: data shape (2, 1024), compare against np.fft.fft
    expected_left = np.fft.fft(audio[:, 0])
    expected_right = np.fft.fft(audio[:, 1])
    np.testing.assert_allclose(rep.data[0], expected_left, atol=1e-10)
    np.testing.assert_allclose(rep.data[1], expected_right, atol=1e-10)
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
cd /Volumes/mcbaron/repos/CHORUS
poetry run pytest tests/test_transforms.py::test_frft_ozaktas_round_trip_synthetic tests/test_transforms.py::test_frft_identity_order_zero tests/test_transforms.py::test_frft_order_one_matches_fft -v
```

Expected: all three FAIL — `TypeError: FrFTTransform.__init__() got an unexpected keyword argument 'frame_size'` or similar.

- [ ] **Step 3: Add `_frft_1d` helper to `transforms.py`**

In `src/chorus/transforms.py`, add the following function directly before the `FrFTTransform` class (after `_validate_stereo`). Also remove the `from scipy.fft import fft, ifft` import — the new code uses `np.fft` only (NumPy FFT). The `from scipy.signal import ShortTimeFFT` import is still needed for `STFTTransform`.

Replace the import block at the top of the file:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pywt
from scipy.signal import ShortTimeFFT
```

Then add `_frft_1d` immediately before `class FrFTTransform`:

```python
def _frft_1d(x: np.ndarray, order: float) -> np.ndarray:
    """Ozaktas-Kutay 2-chirp 1D FrFT.

    Args:
        x: 1D real or complex array of length N.
        order: rotation order α (angle φ = α·π/2).

    Returns:
        1D complex array of length N.

    Special cases:
        order=0.0 → identity (returns x as complex)
        order=1.0 → plain FFT (unnormalized sum, divides by nothing)
    """
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
```

- [ ] **Step 4: Rewrite `FrFTTransform`**

Replace the entire `FrFTTransform` class in `src/chorus/transforms.py`:

```python
class FrFTTransform:
    """Fractional Fourier Transform using the Ozaktas-Kutay 2-chirp algorithm.

    Processes stereo audio in non-overlapping frames of ``frame_size`` samples,
    matching the frame structure of Rust's ``StreamingFrft``.

    Args:
        order: Rotation order α (angle φ = α·π/2). Default 0.5 (45°).
        frame_size: Number of samples per non-overlapping frame. Default 1024.
    """

    def __init__(self, order: float = 0.5, frame_size: int = 1024) -> None:
        self.order = order
        self.frame_size = frame_size

    def forward(self, stereo: np.ndarray) -> TransformRepresentation:
        """Forward FrFT: stereo (n_samples, 2) → TransformRepresentation.

        Zero-pads to a multiple of frame_size, then applies _frft_1d per frame
        per channel. Output data shape: (2, n_padded) complex.
        """
        stereo = _validate_stereo(stereo)
        n_samples = stereo.shape[0]
        import math
        n_padded = math.ceil(n_samples / self.frame_size) * self.frame_size
        n_pad = n_padded - n_samples

        stereo_padded = np.zeros((n_padded, 2), dtype=np.float64)
        stereo_padded[:n_samples, :] = stereo

        n_frames = n_padded // self.frame_size
        out = np.empty((2, n_padded), dtype=complex)

        for ch in range(2):
            for i in range(n_frames):
                start = i * self.frame_size
                end = start + self.frame_size
                frame = stereo_padded[start:end, ch]
                out[ch, start:end] = _frft_1d(frame, self.order)

        return TransformRepresentation(
            data=out,
            original_shape=stereo.shape,
            metadata={
                "transform": "frft",
                "order": self.order,
                "frame_size": self.frame_size,
                "n_pad": n_pad,
                "experimental": True,
            },
        )

    def inverse(self, representation: TransformRepresentation) -> np.ndarray:
        """Inverse FrFT: TransformRepresentation → stereo (n_samples, 2).

        Applies _frft_1d(frame, -order) per frame per channel, takes real part,
        then trims padding to original_shape.
        """
        n_pad: int = representation.metadata["n_pad"]  # type: ignore[assignment]
        n_padded = representation.data.shape[1]
        n_frames = n_padded // self.frame_size
        n_samples = representation.original_shape[0]

        out = np.empty((n_padded, 2), dtype=np.float64)
        for ch in range(2):
            for i in range(n_frames):
                start = i * self.frame_size
                end = start + self.frame_size
                frame = representation.data[ch, start:end]
                out[start:end, ch] = _frft_1d(frame, -self.order).real

        return out[:n_samples, :]

    def inverse_components(
        self, components: np.ndarray, original_shape: tuple[int, int]
    ) -> np.ndarray:
        """Inverse FrFT for component arrays (no metadata available).

        Computes n_pad from original_shape and frame_size rather than reading
        it from metadata.
        """
        import math
        n_padded = math.ceil(original_shape[0] / self.frame_size) * self.frame_size
        n_pad = n_padded - original_shape[0]
        representation = TransformRepresentation(
            data=components,
            original_shape=original_shape,
            metadata={
                "transform": "frft",
                "order": self.order,
                "frame_size": self.frame_size,
                "n_pad": n_pad,
                "experimental": True,
            },
        )
        return self.inverse(representation)
```

- [ ] **Step 5: Run the three new tests — all must pass**

```bash
cd /Volumes/mcbaron/repos/CHORUS
poetry run pytest tests/test_transforms.py::test_frft_ozaktas_round_trip_synthetic tests/test_transforms.py::test_frft_identity_order_zero tests/test_transforms.py::test_frft_order_one_matches_fft -v
```

Expected: PASS, PASS, PASS.

- [ ] **Step 6: Run the full pre-existing FrFT test suite to confirm nothing regressed**

```bash
cd /Volumes/mcbaron/repos/CHORUS
poetry run pytest tests/test_transforms.py -k "frft" -v
```

Expected: all pass. The existing tests (`test_experimental_frft_adapter_smoke`, `test_frft_round_trip_pass_through_stereo`, `test_frft_round_trip_wav`) must continue to pass. Note that `test_frft_round_trip_pass_through_stereo` uses `stereo_identical` (shape `(44100, 2)`) with `order=0.75` — no `frame_size` kwarg, so the default `frame_size=1024` applies. The test checks `recovered.shape == stereo_identical.shape` and allclose — this will pass since the new code pads then trims correctly.

- [ ] **Step 7: Commit**

```bash
git add src/chorus/transforms.py tests/test_transforms.py
git commit -m "feat: replace FrFTTransform with Ozaktas-Kutay 2-chirp algorithm"
```

---

## Task 2: Add `frft_order` to `ChorusConfig` and wire `ChorusProcessor`

**Files:**
- Modify: `src/chorus/core.py`
- Test: `tests/test_transforms.py` (no new test file needed — use existing test infrastructure)

- [ ] **Step 1: Write a failing test for the new config field**

Add to `tests/test_transforms.py`:

```python
def test_frft_transform_respects_order_from_config() -> None:
    """FrFTTransform constructed via ChorusConfig must use frft_order."""
    from chorus.core import ChorusConfig, ChorusProcessor

    rng = np.random.default_rng(99)
    audio = rng.standard_normal((4096, 2))

    # Processor with frft_order=0.3 must not crash and must reconstruct audio
    config = ChorusConfig(sample_rate=48_000, transform="frft", frft_order=0.3, frame_size=1024)
    processor = ChorusProcessor(config)
    result = processor.process(audio)
    # center + only + surround should approximately reconstruct input
    reconstructed = result.center + result.only + result.surround
    np.testing.assert_allclose(reconstructed, audio, atol=1e-6)
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd /Volumes/mcbaron/repos/CHORUS
poetry run pytest tests/test_transforms.py::test_frft_transform_respects_order_from_config -v
```

Expected: FAIL — `TypeError: ChorusConfig.__init__() got an unexpected keyword argument 'frft_order'`.

- [ ] **Step 3: Add `frft_order` to `ChorusConfig` in `core.py`**

In `src/chorus/core.py`, modify the `ChorusConfig` dataclass to add the new field (insert after `smoothing_alpha`):

```python
@dataclass(frozen=True)
class ChorusConfig:
    sample_rate: int
    transform: str = "stft"
    frame_size: int = 1024
    hop_size: int = 512
    smoothing_alpha: float = 0.9
    frft_order: float = 0.5
    epsilon: float = 1e-9
    filter_chains: FilterChains | None = None
```

- [ ] **Step 4: Wire `frft_order` and `frame_size` into `ChorusProcessor.__init__`**

In `src/chorus/core.py`, update the `elif config.transform == "frft":` branch in `ChorusProcessor.__init__`:

```python
        elif config.transform == "frft":
            self.transform = FrFTTransform(order=config.frft_order, frame_size=config.frame_size)
```

The complete `__init__` block after the change:

```python
    def __init__(self, config: ChorusConfig) -> None:
        self.config = config
        if config.transform == "stft":
            self.transform = STFTTransform(STFTConfig(config.frame_size, config.hop_size))
        elif config.transform == "frft":
            self.transform = FrFTTransform(order=config.frft_order, frame_size=config.frame_size)
        elif config.transform == "wavelet":
            self.transform = WaveletTransform()
        else:
            raise ValueError(f"unsupported transform {config.transform!r}")
```

- [ ] **Step 5: Run the new test to confirm it passes**

```bash
cd /Volumes/mcbaron/repos/CHORUS
poetry run pytest tests/test_transforms.py::test_frft_transform_respects_order_from_config -v
```

Expected: PASS.

- [ ] **Step 6: Run the full test suite to confirm no regressions**

```bash
cd /Volumes/mcbaron/repos/CHORUS
poetry run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/chorus/core.py tests/test_transforms.py
git commit -m "feat: add frft_order to ChorusConfig and wire FrFTTransform"
```

---

## Task 3: Add WAV round-trip tests for the new FrFT

**Files:**
- Modify: `tests/test_transforms.py`

The existing `test_frft_round_trip_wav` test uses `FrFTTransform(order=0.5)` without `frame_size`. After Task 1, the default `frame_size=1024` applies, so the test already exercises the new algorithm. This task adds a parametrized WAV test explicitly naming both parameters to lock in the contract.

- [ ] **Step 1: Verify the existing WAV test passes with the new implementation**

```bash
cd /Volumes/mcbaron/repos/CHORUS
poetry run pytest tests/test_transforms.py::test_frft_round_trip_wav -v
```

Expected: PASS for both `PinkPanther.wav` and `TVSong.wav`. The existing test at line 108–115 uses `FrFTTransform(order=0.5)` — default `frame_size=1024` applies, so it exercises the new code path. If this test passes, no new test is needed. If it fails, investigate and fix `FrFTTransform` before continuing.

- [ ] **Step 2: Add an explicit parametrized WAV test with both parameters named**

Add to `tests/test_transforms.py`:

```python
@pytest.mark.parametrize("wav_path", WAV_FILES)
def test_frft_ozaktas_round_trip_wav(wav_path: str) -> None:
    """Ozaktas-Kutay FrFT must round-trip real WAV files within 1e-10."""
    audio = _load_wav_stereo(wav_path)
    t = FrFTTransform(order=0.5, frame_size=1024)
    rep = t.forward(audio)
    recovered = t.inverse(rep)
    np.testing.assert_allclose(recovered, audio, atol=1e-10)
```

- [ ] **Step 3: Run the new WAV test**

```bash
cd /Volumes/mcbaron/repos/CHORUS
poetry run pytest tests/test_transforms.py::test_frft_ozaktas_round_trip_wav -v
```

Expected: PASS for both WAV files.

- [ ] **Step 4: Commit**

```bash
git add tests/test_transforms.py
git commit -m "test: add explicit Ozaktas-Kutay WAV round-trip tests for FrFTTransform"
```

---

## Task 4: Generate `frft_unity_bypass` fixtures

**Files:**
- Modify: `scripts/generate_v2_fixtures.py`

- [ ] **Step 1: Write a test that the fixture generator produces the new case**

Add to `tests/test_transforms.py`:

```python
def test_frft_unity_bypass_fixture_generation() -> None:
    """frft_unity_bypass fixture must be generatable without error."""
    import math
    from chorus.core import ChorusConfig, ChorusProcessor
    from chorus.filters import normalize_filter_config

    sample_rate = 48_000
    samples = 4096
    t = np.arange(samples, dtype=np.float64) / sample_rate
    tone = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    audio = np.column_stack([tone, tone])

    config = ChorusConfig(
        sample_rate=sample_rate,
        transform="frft",
        frft_order=0.5,
        frame_size=1024,
        smoothing_alpha=0.0,
        filter_chains=normalize_filter_config(None),
    )
    processor = ChorusProcessor(config)
    result = processor.process(audio)

    # Output arrays must have the same number of samples as input
    assert result.center.shape == audio.shape
    assert result.only.shape == audio.shape
    assert result.surround.shape == audio.shape

    # center + only + surround must reconstruct input within 1e-6
    reconstructed = result.center + result.only + result.surround
    np.testing.assert_allclose(reconstructed, audio, atol=1e-6)
```

- [ ] **Step 2: Run the test to confirm it passes (verifying config plumbing is correct)**

```bash
cd /Volumes/mcbaron/repos/CHORUS
poetry run pytest tests/test_transforms.py::test_frft_unity_bypass_fixture_generation -v
```

Expected: PASS. If it fails, the issue is in Task 2 plumbing — fix there before continuing.

- [ ] **Step 3: Add `frft_unity_bypass` case to the fixture generator**

In `scripts/generate_v2_fixtures.py`, the `_cases` function currently returns a `dict[str, tuple[np.ndarray, dict | None]]`. The `frft_unity_bypass` case requires a `ChorusConfig`, not just a filter dict. Refactor `_cases` and `build_fixture_set` to support per-case configs.

Replace the `_cases` function signature and body, and update `build_fixture_set`:

```python
def _cases(
    sample_rate: int, samples: int
) -> dict[str, tuple[np.ndarray, dict | None, ChorusConfig | None]]:
    """Return fixture cases as (audio, filters, override_config).

    If override_config is None, build_fixture_set uses the default STFT config.
    """
    tone = _sine(sample_rate, samples, 440.0, 0.25)
    quiet = _sine(sample_rate, samples, 330.0, 1e-8)
    default_filters = None
    return {
        "unity_bypass": (np.column_stack([tone, tone]), default_filters, None),
        "center_dominant": (np.column_stack([tone, tone * 0.95]), default_filters, None),
        "hard_panned_left": (np.column_stack([tone, np.zeros_like(tone)]), default_filters, None),
        "hard_panned_right": (np.column_stack([np.zeros_like(tone), tone]), default_filters, None),
        "phase_inverted_surround": (np.column_stack([tone, -tone]), default_filters, None),
        "known_eq_preset": (
            np.column_stack([tone, tone]),
            {
                "Lc": [{"type": "gain", "db": -6.0}],
                "Lo": [{"type": "eq", "mode": "highpass", "frequency_hz": 120.0, "q": 0.707}],
                "Rs": [{"type": "polarity"}],
            },
            None,
        ),
        "silence": (np.zeros((samples, 2), dtype=np.float64), default_filters, None),
        "near_silence": (np.column_stack([quiet, -quiet]), default_filters, None),
        "frft_unity_bypass": (
            np.column_stack([tone, tone]),
            default_filters,
            ChorusConfig(
                sample_rate=sample_rate,
                transform="frft",
                frft_order=0.5,
                frame_size=1024,
                smoothing_alpha=0.0,
            ),
        ),
    }
```

Update `build_fixture_set` to handle the optional override config:

```python
def build_fixture_set(sample_rate: int = 48_000, samples: int = 4096) -> dict[str, object]:
    arrays: dict[str, np.ndarray] = {}
    manifest = {
        "sample_rate": sample_rate,
        "samples": samples,
        "tolerances": {
            "bypass_max_abs": 1e-6,
            "filtered_max_abs": 5e-5,
        },
        "cases": {},
    }
    for name, (audio, filters, override_config) in _cases(sample_rate, samples).items():
        if override_config is not None:
            config = override_config
        else:
            config = ChorusConfig(
                sample_rate=sample_rate,
                smoothing_alpha=0.0,
                filter_chains=normalize_filter_config(filters),
            )
        processor = ChorusProcessor(config)
        result = processor.process(audio)
        arrays[f"{name}.input"] = audio
        arrays[f"{name}.center"] = result.center
        arrays[f"{name}.only"] = result.only
        arrays[f"{name}.surround"] = result.surround
        manifest["cases"][name] = {
            "filters": filters or {},
            "outputs": {
                "input": f"{name}.input.npy",
                "center": f"{name}.center.npy",
                "only": f"{name}.only.npy",
                "surround": f"{name}.surround.npy",
            },
        }
    return {"manifest": manifest, "arrays": arrays}
```

Note: The existing cases previously used `ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0, filter_chains=normalize_filter_config(filters))`. Now, `normalize_filter_config` is called inside `build_fixture_set` for non-override cases. The import for `normalize_filter_config` is already present at the top of the file.

- [ ] **Step 4: Run the fixture generator**

```bash
cd /Volumes/mcbaron/repos/CHORUS
poetry run python scripts/generate_v2_fixtures.py
```

Expected: exits cleanly. Verify the new fixture files exist:

```bash
ls /Volumes/mcbaron/repos/CHORUS/fixtures/v2/frft_unity_bypass.*.npy
```

Expected output:
```
fixtures/v2/frft_unity_bypass.center.npy
fixtures/v2/frft_unity_bypass.input.npy
fixtures/v2/frft_unity_bypass.only.npy
fixtures/v2/frft_unity_bypass.surround.npy
```

- [ ] **Step 5: Run the full Python test suite to confirm no regressions**

```bash
cd /Volumes/mcbaron/repos/CHORUS
poetry run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_v2_fixtures.py fixtures/v2/frft_unity_bypass.input.npy fixtures/v2/frft_unity_bypass.center.npy fixtures/v2/frft_unity_bypass.only.npy fixtures/v2/frft_unity_bypass.surround.npy fixtures/v2/manifest.json tests/test_transforms.py
git commit -m "feat: add frft_unity_bypass fixture case and generate fixture files"
```

---

## Task 5: Un-ignore and implement the Rust fixture test

**Files:**
- Modify: `rust/chorus-dsp/tests/fixtures.rs`

No Rust source changes are required — only the test in `fixtures.rs` is updated.

- [ ] **Step 1: Verify the fixture files are present (prerequisite from Task 4)**

```bash
ls /Volumes/mcbaron/repos/CHORUS/fixtures/v2/frft_unity_bypass.*.npy
```

If not present, complete Task 4 first.

- [ ] **Step 2: Replace the `#[ignore]`d stub with a real implementation**

In `rust/chorus-dsp/tests/fixtures.rs`, replace:

```rust
#[test]
#[ignore = "FrFT fixture parity blocked: Python uses phase-shifted FFT, not true FrFT; see TODO in frft.rs"]
fn rust_matches_python_frft_fixture() {
    // Enable once Python FrFTTransform is updated to Ozaktas-Kutay algorithm.
    // Until then, Rust and Python FrFT outputs are not comparable.
    todo!()
}
```

with:

```rust
#[test]
fn rust_matches_python_frft_fixture() {
    let input = load_stereo("frft_unity_bypass.input.npy");
    let config = DspConfig {
        transform: TransformKind::Frft {
            order: 0.5,
            frame_size: 1024,
            smoothing_alpha: 0.0,
        },
        ..DspConfig::default()
    };
    let mut dsp = ChorusDsp::new(config);
    let output = dsp.process(&input).unwrap();
    let skip = 1024; // warm-up: first frame has no prior context
    let compare_len = output.len().min(input.len()).saturating_sub(skip);
    assert!(
        compare_len > 0,
        "not enough output samples to compare after warm-up skip"
    );
    assert_close(
        &output[skip..skip + compare_len],
        &load_stereo("frft_unity_bypass.center.npy")[skip..skip + compare_len],
        1e-6,
    );
}
```

Also remove the TODO comment at the top of `rust/chorus-dsp/src/transforms/frft.rs`:

```rust
// TODO(follow-on): Python FrFTTransform uses phase-shifted FFT (FFT * exp(-0.5j*pi*order)),
// not true FrFT. Rust implements a unitary 2-chirp discrete FrFT.  Python must be updated
// before Rust/Python FrFT fixture parity is possible.
```

- [ ] **Step 3: Run the Rust fixture tests via the `run-rust-remote` skill**

```bash
# Run via run-rust-remote skill
cargo test --test fixtures -- rust_matches_python_frft_fixture --nocapture
```

Expected: `test rust_matches_python_frft_fixture ... ok`.

- [ ] **Step 4: Run the full Rust test suite via the `run-rust-remote` skill**

```bash
# Run via run-rust-remote skill
cargo test -p chorus-dsp
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add rust/chorus-dsp/tests/fixtures.rs rust/chorus-dsp/src/transforms/frft.rs
git commit -m "feat: un-ignore and implement rust_matches_python_frft_fixture test"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task |
|---|---|
| `_frft_1d` helper with Ozaktas-Kutay formula | Task 1, Step 3 |
| Special cases `order=0.0` (identity) and `order=1.0` (FFT) | Task 1, Steps 1 & 4 |
| `FrFTTransform` constructor with `order=0.5`, `frame_size=1024` defaults | Task 1, Step 4 |
| `forward`: zero-pad, process frames, return `(2, n_padded)` complex | Task 1, Step 4 |
| `forward` metadata includes `n_pad` | Task 1, Step 4 |
| `inverse`: extract `n_pad`, apply `_frft_1d(-order)`, trim | Task 1, Step 4 |
| `inverse_components`: compute `n_pad` from `original_shape` | Task 1, Step 4 |
| `ChorusConfig.frft_order: float = 0.5` | Task 2, Step 3 |
| `ChorusProcessor` passes `order` and `frame_size` to `FrFTTransform` | Task 2, Step 4 |
| Python round-trip test `test_frft_round_trip` | Task 1, Steps 1–5 |
| Python WAV tests (PinkPanther, TVSong) | Task 3 |
| Python special cases `order=0.0`, `order=1.0` | Task 1, Step 1 |
| `frft_unity_bypass` fixture case in generator | Task 4, Step 3 |
| Four fixture `.npy` files generated | Task 4, Step 4 |
| Rust fixture test un-ignored and implemented | Task 5, Step 2 |
| Rust TODO comment removed from `frft.rs` | Task 5, Step 2 |
| No Rust source changes (only fixture test) | Task 5 (fixture test only) |

### Placeholder scan

No "TBD", "TODO" (in plan), "implement later", "add validation", or placeholder steps found. All code blocks are complete.

### Type consistency check

- `_frft_1d(x: np.ndarray, order: float) -> np.ndarray` — used as `_frft_1d(frame, self.order)` and `_frft_1d(frame, -self.order)` in `forward` and `inverse`. Consistent.
- `FrFTTransform.forward` returns `TransformRepresentation` with `data` shape `(2, n_padded)` complex. `inverse` reads `data[ch, start:end]` — consistent.
- `metadata["n_pad"]` written in `forward`, read in `inverse` as `int`. `inverse_components` recomputes it independently. Consistent.
- `ChorusConfig.frft_order: float = 0.5` → `ChorusProcessor` passes `order=config.frft_order` → `FrFTTransform.__init__(self, order: float = 0.5, ...)`. Consistent.
- `ChorusConfig.frame_size: int = 1024` → `ChorusProcessor` passes `frame_size=config.frame_size` → `FrFTTransform.__init__(self, ..., frame_size: int = 1024)`. Consistent.
- Rust `TransformKind::Frft { order: 0.5, frame_size: 1024, smoothing_alpha: 0.0 }` matches `StreamingFrft::new(order, frame_size, smoothing_alpha, epsilon, filter_chains)`. Consistent.
- `_cases` return type changed from `dict[str, tuple[np.ndarray, dict | None]]` to `dict[str, tuple[np.ndarray, dict | None, ChorusConfig | None]]`. The `build_fixture_set` unpacking updated to `for name, (audio, filters, override_config) in _cases(...)`. Consistent.
