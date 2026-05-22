# EQ Biquad Implementation Design

**Date:** 2026-05-22
**Status:** Approved
**Scope:** Implement `FilterSpec::Eq` in both Python and Rust using per-bin complex frequency response, enabling `known_eq_preset` fixture parity.

---

## Background

The Rust DSP core panics when a filter chain contains `FilterSpec::Eq`. The Python path uses `scipy.signal.sosfilt` (time-domain IIR). These approaches are not comparable at the sample level, so the `known_eq_preset` fixture parity test is currently `#[ignore]`d.

This design switches both sides to a **per-bin complex H(ω) spectral approach**: compute the biquad transfer function H(z) from filter parameters, evaluate it at each spectral bin frequency as a complex value, and apply it as a multiplicative gain on the complex spectral bins. Both sides use identical coefficient formulas, giving exact Python–Rust fixture parity.

---

## Approach: Per-bin Complex H

For a biquad with coefficients `b = [b0, b1, b2]`, `a = [1, a1, a2]`:

```
H(e^jω) = (b0 + b1·e^{-jω} + b2·e^{-2jω}) / (1 + a1·e^{-jω} + a2·e^{-2jω})
```

At STFT bin k: `ω_k = 2π·k/N` where N = `frame_size`. For SOS filters (multiple stages), multiply H values per stage.

This is a circular-convolution approximation per STFT frame. The approximation degrades for filters with very long effective IRs relative to frame size (e.g., low-frequency highpass), but is within the `filtered_max_abs: 5e-05` fixture tolerance for the target parameter range.

---

## Supported EQ Modes

Matching current Python implementation:

| Mode | Python API | Coefficient formula |
|---|---|---|
| `highpass` | `scipy.signal.butter(2, fc, btype="highpass", fs=sr, output="sos")` | 2nd-order Butterworth HP via bilinear transform (see below) |
| `lowpass` | `scipy.signal.butter(2, fc, btype="lowpass", fs=sr, output="sos")` | 2nd-order Butterworth LP via bilinear transform (see below) |
| `peaking` (default) | `scipy.signal.iirpeak(f0, Q, fs)` | H_total(ω) = 1 + H_peak(ω)·(10^{gain_db/20} − 1) |

### Coefficient Formulas (Butterworth 2nd-order)

For highpass with pre-warped `k = tan(π·fc/fs)`:

```
b0 =  1 / (1 + sqrt(2)·k + k²)
b1 = -2·b0
b2 =  b0
a1 =  2·(k² − 1) · b0
a2 =  (1 − sqrt(2)·k + k²) · b0
```

For lowpass: same `k`, but `b0 = k²/(1 + sqrt(2)·k + k²)`, `b1 = 2·b0`, `b2 = b0`, same `a1`/`a2`.

These match scipy's output exactly. Implementation should verify against scipy for a known `(fc, fs)` pair in unit tests.

For `iirpeak`: `w0 = 2π·f0/fs`, `bw = w0/Q`:

```
b = [tan(bw/2), 0, -tan(bw/2)]     (then normalize by a0)
a = [1 + tan(bw/2), -2·cos(w0), 1 - tan(bw/2)]
```

Peaking H_total is applied as: `1 + H_peak(ω_k) · (A − 1)` where `A = 10^{gain_db/20}`.

---

## Rust Changes (`filters.rs`)

### `apply_chains_complex` (STFT path)

The `FilterSpec::Eq` arm currently panics. Replace with:

```rust
FilterSpec::Eq { mode, frequency_hz, q, gain_db } => {
    let h_bins = eq_frequency_response(mode, *frequency_hz, *q, *gain_db, sample_rate, bins.len());
    for (bin, h) in bins.iter_mut().zip(h_bins.iter()) {
        *bin *= h;
    }
}
```

### New function `eq_frequency_response`

```rust
/// Compute complex H(e^jω) at each bin frequency for the given EQ spec.
/// bins_count = frame_size / 2 + 1 (one-sided).
pub fn eq_frequency_response(
    mode: &str,
    frequency_hz: f64,
    q: f64,
    gain_db: f64,
    sample_rate: u32,
    bins_count: usize,
) -> Vec<Complex<f64>>;
```

Internally:
1. Compute biquad coefficients `(b, a)` from filter parameters using the same bilinear-transform formulas as scipy
2. For k in `0..bins_count`: `ω = 2π·k / ((bins_count - 1) * 2)`, evaluate H(e^{jω}) via complex polynomial evaluation
3. For `peaking`: return `1 + H_peak(ω) * (10^{gain_db/20} - 1.0)` per bin

### `chain_scalar` (non-STFT path)

The `FilterSpec::Eq` arm currently `unimplemented!()`. The non-STFT path (wavelet, FrFT) operates on real time-domain audio after reconstruction. For these paths, implement as full-signal spectral EQ:
- `rfft(audio)` → apply per-bin H → `irfft` (using `rustfft` on the full contribution vector)

---

## Python Changes (`filters.py`)

`_apply_eq(audio, spec, sample_rate)` switches from `signal.sosfilt` / `signal.lfilter` to:

```python
def _apply_eq(audio: np.ndarray, spec: FilterSpec, sample_rate: int) -> np.ndarray:
    h = _eq_frequency_response(spec, sample_rate, len(audio))
    spectrum = np.fft.rfft(audio)
    return np.fft.irfft(spectrum * h, n=len(audio))
```

`_eq_frequency_response(spec, sample_rate, n)` returns `np.ndarray` of shape `(n//2 + 1,)` complex, computed from the same biquad formulas as Rust.

---

## Fixture Regeneration

The `known_eq_preset` fixture uses a highpass EQ at 120Hz on the `Lo` contribution. After Python's `_apply_eq` changes, the fixture `.npy` files must be regenerated:

```bash
python scripts/generate_v2_fixtures.py
```

Rust's `rust_matches_python_known_eq_preset_fixture` test is un-`#[ignore]`d and wired to compare Rust output against the regenerated files at `filtered_max_abs: 5e-05`.

---

## Parity Note

Python applies H globally (full-signal FFT, N = 4096 for fixture), Rust applies per STFT frame (N = 1024). These are the same circular-convolution approximation but at different granularities. Outputs will not be sample-identical but should be within fixture tolerance.

If sample-exact parity is required in future, Python would need to apply the filter using STFT-frame-granularity. That is a follow-on change and not in scope here.

---

## Testing

### Rust unit tests (`filters.rs`)

- `eq_highpass_attenuates_below_cutoff`: 440Hz sine → highpass at 1000Hz → verify bin 0–20 (≤440Hz) has |H| < −20dB
- `eq_peaking_boosts_center_bin`: peaking at 1000Hz, +6dB → verify center bin has |H| ≈ 2.0
- `eq_chain_compose`: two EQ filters in one chain compose multiplicatively

### Rust fixture test (`fixtures.rs`)

- `rust_matches_python_known_eq_preset_fixture`: un-ignored, loads regenerated fixture files, tolerance `5e-05`

### Rust WAV tests (`stft.rs`)

- Apply `known_eq_preset` filter chain to `PinkPanther.wav` and `TVSong.wav` via the STFT path; verify output is finite and `Lo` contribution has less low-frequency energy than input (highpass effect)

### Python unit tests (`tests/test_filters.py`)

- `test_eq_highpass_attenuates_below_cutoff`: same assertion as Rust version against audio output
- `test_eq_fixture_parity`: regenerated fixture matches the updated `_apply_eq` output

### Python WAV tests (`tests/test_filters.py`)

- Apply `known_eq_preset` chains to `tests/test_tracks_wav/PinkPanther.wav` and `TVSong.wav`; verify finite output and that `Lo` is high-pass filtered relative to input

---

## Follow-on

Sample-exact Python–Rust EQ parity would require Python to apply EQ at STFT-frame granularity. Not in scope; tracked in design comments.
