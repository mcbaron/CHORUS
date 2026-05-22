# Python FrFT Ozaktas-Kutay Design

**Date:** 2026-05-22
**Status:** Approved
**Scope:** Replace the Python `FrFTTransform` phase-shifted FFT approximation with the true Ozaktas-Kutay algorithm, restructure to process in non-overlapping frames matching Rust's `StreamingFrft`, generate FrFT fixtures, and un-ignore the Rust FrFT fixture test.

---

## Background

The current Python `FrFTTransform` computes `FFT(x) * exp(-0.5j*π*order)` — a phase-shifted FFT, not a true fractional Fourier transform. The Rust `StreamingFrft` implements the Ozaktas-Kutay 2-chirp algorithm. Because the algorithms differ, Rust and Python FrFT outputs are incomparable, and the Rust fixture test is `#[ignore]`d.

---

## Algorithm: Ozaktas-Kutay 2-Chirp FrFT

For signal `x` of length N, rotation angle `φ = order * π/2`:

```
chirp[n] = exp(-iπ · cot(φ) · n² / N)    for n = 0, …, N-1
norm = sqrt((1 - i·cot(φ)) / (N · |csc(φ)|))

FrFT(x) = norm · ifft(chirp · fft(chirp · x))
```

Normalization convention: `fft` is unnormalized (sum), `ifft` divides by N — matching both numpy and rustfft behavior.

Special cases:
- `order = 0.0` → identity (return `x` unchanged)
- `order = 1.0` → plain FFT

Inverse FrFT at order α uses order −α, i.e., `FrFT_{-α}`. With order negated:
- `cot(−φ) = −cot(φ)`, so `chirp_inv = conj(chirp)`
- `norm_inv = conj(norm)`

This matches Rust's inverse formula: `(1/norm) · conj(chirp) · ifft(conj(chirp) · fft(Y))`.

---

## Helper: `_frft_1d`

New module-private function in `transforms.py`:

```python
def _frft_1d(x: np.ndarray, order: float) -> np.ndarray:
    """Ozaktas-Kutay 1D FrFT. x must be 1D complex or real."""
    N = len(x)
    if order == 0.0:
        return x.astype(complex)
    if order == 1.0:
        return np.fft.fft(x)
    phi = order * np.pi / 2
    n = np.arange(N, dtype=np.float64)
    cot_phi = np.cos(phi) / np.sin(phi)
    csc_phi = 1.0 / np.sin(phi)
    chirp = np.exp(-1j * np.pi * cot_phi * n**2 / N)
    norm = np.sqrt((1.0 - 1j * cot_phi) / (N * abs(csc_phi)))
    return norm * np.fft.ifft(chirp * np.fft.fft(chirp * x))
```

Inverse: `_frft_1d(x, -order)`.

---

## `FrFTTransform` Changes

### Constructor

```python
class FrFTTransform:
    def __init__(self, order: float = 0.5, frame_size: int = 1024) -> None:
```

**Note:** Default `order` changes from `1.0` (plain FFT) to `0.5` (true FrFT at 45°). The old default produced no useful test coverage of the algorithm.

### `forward(stereo)`

1. Validate stereo: shape `(n_samples, 2)`
2. Zero-pad to `n_padded = ceil(n_samples / frame_size) * frame_size`
3. For each non-overlapping frame `i` (0 to `n_frames - 1`):
   - Extract `frame = stereo_padded[i*frame_size : (i+1)*frame_size, :]`
   - Apply `_frft_1d(frame[:, ch], order)` per channel
4. Return `TransformRepresentation`:
   - `data`: shape `(2, n_padded)` complex — flat concatenation of per-frame FrFT output per channel
   - `original_shape`: `stereo.shape`
   - `metadata`: includes `n_pad = n_padded - n_samples`

**Why flat output:** `center_prototype` and `ComplexEstimator` are both pointwise operations. With `smoothing_alpha=0` (used in fixtures), global processing is mathematically equivalent to per-frame processing. The existing `ChorusProcessor` pipeline works without modification.

### `inverse(representation)`

1. Extract `n_pad` from `metadata`
2. Reshape `data` from `(2, n_padded)` to `(n_frames, frame_size, 2)` (per frame, per channel)
3. For each frame, apply `_frft_1d(frame[:, ch], -order)` per channel, take `.real`
4. Flatten and trim to `original_shape[0]` samples

### `inverse_components(components, original_shape)`

Same as `inverse()` but `n_pad` is computed from `original_shape[0]` and `frame_size`:
`n_padded = ceil(original_shape[0] / frame_size) * frame_size`, `n_pad = n_padded - original_shape[0]`.

---

## `ChorusConfig` Addition

```python
@dataclass
class ChorusConfig:
    ...
    frft_order: float = 0.5
    frame_size: int = 1024  # shared by STFT and FrFT
```

`ChorusProcessor` passes `order=config.frft_order, frame_size=config.frame_size` to `FrFTTransform`.

---

## Fixture Generation

### New fixture case: `frft_unity_bypass`

Add to `generate_v2_fixtures.py`:

```python
"frft_unity_bypass": (
    np.column_stack([tone, tone]),
    None,
    ChorusConfig(transform="frft", frft_order=0.5, frame_size=1024),
),
```

The `build_fixture_set` function is extended to accept per-case `ChorusConfig` overrides, or the FrFT cases are generated in a separate loop.

Output files (in `fixtures/v2/`):
- `frft_unity_bypass.input.npy`
- `frft_unity_bypass.center.npy`
- `frft_unity_bypass.only.npy`
- `frft_unity_bypass.surround.npy`

Manifest gains a `frft_unity_bypass` entry with `"transform": "frft"` in the metadata.

---

## Rust Fixture Test (`fixtures.rs`)

`rust_matches_python_frft_fixture` is un-`#[ignore]`d and implemented:

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
    let skip = 1024; // warm-up
    let compare_len = output.len().min(input.len()).saturating_sub(skip);
    // For a stereo-identical tone, center ≈ input and only ≈ 0. Compare center output.
    assert_close(&output[skip..skip + compare_len], &load_stereo("frft_unity_bypass.center.npy")[skip..skip + compare_len], 1e-6);
}
```

---

## Testing

### Python round-trip (`tests/test_transforms.py`)

```python
def test_frft_round_trip():
    audio = np.column_stack([sine(440, 4096), sine(440, 4096)])
    t = FrFTTransform(order=0.5, frame_size=1024)
    rep = t.forward(audio)
    recovered = t.inverse(rep)
    np.testing.assert_allclose(recovered, audio, atol=1e-10)
```

### Python WAV tests

Load `tests/test_tracks_wav/PinkPanther.wav` and `TVSong.wav` via `soundfile.read()`. For each:
```python
rep = frft.forward(audio)
recovered = frft.inverse(rep)
np.testing.assert_allclose(recovered, audio, atol=1e-10)
```

### Python special cases

- `order=0.0`: `forward` returns input unchanged (identity)
- `order=1.0`: `forward` returns `np.fft.fft` of each frame

### Rust round-trip and special cases

Already present in `transforms/frft.rs`. Verify they still pass after no Rust changes are required for this task.

---

## Non-Goals

- Updating Rust `StreamingFrft` — it already implements Ozaktas-Kutay correctly
- Matching fixture parity with `smoothing_alpha > 0` — per-frame vs. global estimation diverges; fixture generation uses `alpha=0`
- FrFT with EQ filters — handled by the EQ biquad spec
