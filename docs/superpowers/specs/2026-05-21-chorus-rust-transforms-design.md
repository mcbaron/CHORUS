# CHORUS Rust Transform Port Design

**Date:** 2026-05-21
**Status:** Approved
**Scope:** Port STFT, FrFT, and Wavelet transforms to Rust with streaming/real-time support for DJ and plugin integration.

---

## Background

The Python v2 `ChorusProcessor` performs stereo decomposition (center/only/surround) in transform domain using complex per-coefficient prototype computation and estimation. The Rust v3 port initially used time-domain processing, which diverged from Python fixture outputs. This design ports all three Python v2 transforms to Rust with first-class streaming support.

---

## Goals

- Rust transforms produce output that matches Python v2 fixtures within tolerance (STFT and Wavelet)
- All three transforms operate in streaming mode: compatible with real-time audio callbacks (JUCE, DJ software)
- Algorithmic latency ≤ ~21ms at 48kHz (one STFT frame) for STFT; ≤ ~6ms for Wavelet
- FrFT implements the true Ozaktas-Kutay algorithm (Python uses a phase-shifted FFT approximation — see follow-on task below)

---

## Module Structure

```
rust/chorus-dsp/src/
  transforms/
    mod.rs       — Transform trait, TransformKind enum
    stft.rs      — StreamingStft
    frft.rs      — StreamingFrft
    wavelet.rs   — StreamingWavelet
  prototypes.rs  — updated: complex matched-magnitude algorithm
  estimation.rs  — updated: per-coefficient complex estimation
  processor.rs   — updated: holds Box<dyn Transform>
  lib.rs         — adds pub mod transforms
```

New dev dependency: `hound = "3"` for WAV loading in tests.
New regular dependency: `rustfft = "6"`, `num-complex = "0.4"`.

---

## Transform Trait

```rust
pub trait Transform: Send {
    /// Push one stereo block. Returns processed stereo output samples
    /// when enough output is available; may return empty vec during warm-up.
    fn process_block(&mut self, input: &[[f64; 2]]) -> Vec<[f64; 2]>;
    fn reset(&mut self);
}
```

Callers must not assume sample-in = sample-out synchrony. Output must be drained on every call. Warm-up produces no output until the first full frame is available.

`TransformKind` enum in `DspConfig`:

```rust
pub enum TransformKind {
    Stft { frame_size: usize, hop_size: usize, smoothing_alpha: f64 },
    Frft { order: f64, frame_size: usize, smoothing_alpha: f64 },
    Wavelet { wavelet: WaveletKind, level: usize, frame_size: usize, smoothing_alpha: f64 },
}

pub enum WaveletKind {
    Db4,
}
```

Default: `TransformKind::Stft { frame_size: 1024, hop_size: 512 }`.

---

## DspConfig Changes

```rust
pub struct DspConfig {
    pub sample_rate: u32,
    pub epsilon: f64,
    pub filter_chains: FilterChains,
    pub transform: TransformKind,
}
```

`smoothing_alpha` moves inside each `TransformKind` variant. Each transform variant carries its own `smoothing_alpha: f64` field because the smoothing semantics differ: STFT smooths across time frames, FrFT and Wavelet smooth across processing blocks.

---

## STFT Streaming (`stft.rs`)

**Parameters:** `frame_size=1024`, `hop_size=512`, `sqrt_hann` window, one-sided real FFT (`frame_size/2 + 1 = 513` complex bins per channel).

**Internal state:**
- `input_buf: VecDeque<[f64; 2]>` — accumulates incoming samples
- `output_buf: VecDeque<[f64; 2]>` — overlap-added output
- `overlap: Vec<[f64; 2]>` — OLA tail of length `frame_size - hop_size`
- `window: Vec<f64>` — precomputed sqrt-Hann coefficients
- Four `ComplexEstimator` instances (Lc, Rc, Ls, Rs), each with `frame_size/2 + 1` coefficient slots

**Per-hop processing** (triggers when `input_buf.len() >= frame_size`):
1. Extract `frame_size` samples, apply window, real-FFT → 513 complex bins per channel
2. Compute `center_prototype` and `surround_prototype` per bin (complex matched-magnitude)
3. Run per-bin `ComplexEstimator::estimate` → Lc, Rc, Ls, Rs bin arrays
4. Derive Lo = Left − Lc − Ls, Ro = Right − Rc − Rs (per bin)
5. Apply filter chains per contribution (gain/polarity/mute/solo are bin-independent scalings)
6. IFFT each of the 6 contribution arrays, apply synthesis window, overlap-add into output buffer
7. Advance input ring by `hop_size`; emit `hop_size` samples from output buffer

**Latency:** `frame_size` samples = ~21ms at 48kHz. First output after `frame_size` input samples; steady-state emits `hop_size` output per hop.

**EQ filters** (`FilterSpec::Eq`) remain panicking placeholder — deferred to a follow-on task requiring per-bin biquad coefficients.

---

## FrFT Streaming (`frft.rs`)

Implements the Ozaktas-Kutay algorithm. The transform of order `α` (where the rotation angle is `α * π/2`) decomposes into:

1. Chirp pre-multiply: `x[n] * exp(-iπ * cot(φ) * (n/N)²)` where `φ = α * π/2`
2. FFT
3. Chirp post-multiply in frequency domain: `X[k] * exp(-iπ * cot(φ) * (k/N)²)`
4. Normalization: `sqrt((1 - i*cot(φ)) / N)`
5. IFFT to get transform-domain coefficients for prototype/estimation

Chirp tables are precomputed at construction from `order` and `frame_size`.

**Special cases:** `order = 1.0` uses plain FFT; `order = 0.0` is identity — neither uses the chirp path.

**Streaming model:** same ring-buffer pattern as STFT. `frame_size` is configurable (default 1024). No overlap needed — FrFT is a global transform per block; boundary effects are accepted.

**Python callout comment in `frft.rs`:**
```rust
// TODO(follow-on): Python FrFTTransform uses phase-shifted FFT (FFT * exp(-0.5j*pi*order)),
// not true FrFT. Rust implements Ozaktas-Kutay. Python must be updated before
// Rust/Python FrFT fixture parity is possible.
```

---

## Wavelet Streaming (`wavelet.rs`)

**Filter bank:** db4 Daubechies-4 filter coefficients hardcoded (8 taps, well-known constants). No external wavelet crate. Implements `wavedec` as `level` successive applications of the two-channel filter bank (lowpass + highpass, downsample by 2) with periodic boundary extension. `waverec` is the matching synthesis bank.

**Overlap-save parameters:**
- Filter support at level 3 = `(filter_len - 1) * (2^level - 1) = 7 * 7 = 49` samples of boundary contamination per side
- Default `frame_size = 512`; overlap = 2 × 49 = 98 samples
- Each call accumulates samples; when `input_buf.len() >= frame_size + overlap`, extract padded block, decompose, process coefficients, reconstruct, discard first and last 49 samples, emit clean center

**Internal state:**
- `input_buf: VecDeque<[f64; 2]>` — pre-loaded with `overlap` zeros at init
- `output_buf: VecDeque<[f64; 2]>` — output ring buffer
- Four `ComplexEstimator` instances (per wavelet coefficient position, stateful across blocks)

**Latency:** `frame_size/2 + overlap/2` ≈ 305 samples (~6ms at 48kHz).

---

## Complex Prototypes (`prototypes.rs`)

Existing real-valued functions renamed to `center_prototype_real` / `surround_prototype_real`. New complex versions match the Python exactly:

```rust
pub fn center_prototype(left: &[Complex<f64>], right: &[Complex<f64>]) -> Vec<Complex<f64>> {
    left.iter().zip(right).map(|(l, r)| {
        let shared = l.norm().min(r.norm());
        (safe_unit_phase(*l) * shared + safe_unit_phase(*r) * shared) * 0.5
    }).collect()
}

pub fn surround_prototype(left: &[Complex<f64>], right: &[Complex<f64>]) -> Vec<Complex<f64>> {
    left.iter().zip(right).map(|(l, r)| {
        let shared = l.norm().min(r.norm());
        (safe_unit_phase(*l) * shared - safe_unit_phase(*r) * shared) * 0.5
    }).collect()
}

fn safe_unit_phase(v: Complex<f64>) -> Complex<f64> {
    let mag = v.norm();
    if mag > 0.0 { v / mag } else { Complex::new(0.0, 0.0) }
}
```

---

## Complex Estimation (`estimation.rs`)

New `ComplexEstimator` struct (the existing `SmoothedScalarEstimator` is kept unchanged for real-domain fallback):

```rust
pub struct ComplexEstimator {
    alpha: f64,
    epsilon: f64,
    cross: Vec<Complex<f64>>,  // per-coefficient smoothed cross-correlation
    auto: Vec<f64>,            // per-coefficient smoothed auto-correlation
}
```

Per-call, for each coefficient index `k`:
```
instant_cross[k] = prototype[k] * conj(source[k])
instant_auto[k]  = |source[k]|²
cross[k]  = (1−α)*instant_cross[k] + α*cross[k]
auto[k]   = (1−α)*instant_auto[k]  + α*auto[k]
weight[k] = Re(cross[k]) / max(auto[k], ε)
output[k] = weight[k] * source[k]
```

`ComplexEstimator::new(alpha, epsilon, num_coeffs)` allocates zeroed state arrays. The four estimator instances (Lc, Rc, Ls, Rs) live inside each transform struct, not in `ChorusDsp`.

---

## Processor Integration (`processor.rs`)

```rust
pub struct ChorusDsp {
    config: DspConfig,
    transform: Box<dyn Transform>,
}
```

`ChorusDsp::process` delegates entirely to `self.transform.process_block(input)`. All prototype, estimation, filter chain, and OLA logic moves into the transform impls. `ChorusDsp` becomes a thin config-to-transform router.

`ChorusDsp::reset` calls `self.transform.reset()` — needed when the host resets the plugin (e.g., stop/start in DAW).

---

## Testing

### Python round-trip tests (add to `tests/test_transforms.py`)

Load `tests/test_tracks_wav/PinkPanther.wav` and `tests/test_tracks_wav/TVSong.wav` via `soundfile.read()`. For each of STFT, FrFT, and Wavelet:

```python
representation = transform.forward(audio)
reconstructed = transform.inverse(representation)
np.testing.assert_allclose(reconstructed, audio, atol=1e-10, rtol=1e-10)
```

Same tolerance as the existing `stereo_identical` sine wave tests. These establish the Python baseline before Rust is compared against it.

### Rust unit tests (in each transform file)

- **Sine wave round-trip:** generate 4096 samples of 440Hz sine, push through transform, verify reconstruction within `1e-6`
- **WAV round-trip:** load `../../tests/test_tracks_wav/PinkPanther.wav` and `TVSong.wav` via `hound`, push through streaming transform in 512-sample chunks, collect all output, compare reconstruction within `1e-6`
- **Streaming consistency:** same signal fed in 512-sample chunks vs. one block produces identical output (modulo warm-up)
- **FrFT special cases:** identity at `order=0.0`, FFT equivalence at `order=1.0`, round-trip at `order=0.5`
- **Wavelet boundary:** verify discarded boundary samples are not present in output

### Fixture parity tests (`tests/fixtures.rs`)

- STFT and Wavelet: `unity_bypass` + 6 non-EQ cases at `1e-6` tolerance
- FrFT: `#[ignore]` — see Python callout above
- `known_eq_preset`: `#[ignore]` — pending EQ biquad implementation

---

## Follow-on Tasks

1. **Python FrFT:** Replace `FrFTTransform` phase-shifted FFT with true Ozaktas-Kutay algorithm so Python and Rust FrFT produce fixture-comparable output.
2. **EQ biquad:** Implement `FilterSpec::Eq` in both Rust and Python paths using per-bin biquad coefficients, enabling `known_eq_preset` fixture parity.
3. **Real-time buffer sizing:** Profile streaming latency under JUCE callbacks and tune default `frame_size`/`hop_size` for specific DJ host requirements.
