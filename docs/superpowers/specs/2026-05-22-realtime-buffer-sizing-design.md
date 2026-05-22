# Real-Time Buffer Sizing Design

**Date:** 2026-05-22
**Status:** Approved
**Scope:** Profile CHORUS streaming DSP under host-realistic callback buffer sizes, document per-host latency characteristics, and tune `frame_size`/`hop_size` defaults for DJ plugin use.

---

## Background

CHORUS's `StreamingStft` and `StreamingFrft` accumulate input in ring buffers and emit output in `hop_size`-sample chunks. The JUCE plugin host drives the plugin with callback buffers whose size varies by host and user settings. The current defaults (`frame_size=1024`, `hop_size=512`) provide ~21ms algorithmic latency at 48kHz — within budget, but not measured against realistic host callback patterns.

The four target DJ hosts (Traktor Pro, Serato DJ Pro, Mixxx, VirtualDJ) each have typical callback sizes and latency expectations. This task builds a standalone Rust harness to measure processing time and output latency across the full parameter grid, then recommends per-host defaults.

---

## Benchmark Harness

### Location

`rust/chorus-dsp/benches/streaming_latency.rs`

A standalone binary (not `cargo bench` criterion-style, to avoid the criterion dev dependency). Run with:

```bash
cargo run --release --bin streaming_latency
```

Add to `Cargo.toml` as a `[[bin]]` target.

### Audio Sources

Three audio sources drive the benchmark for each `(callback_size, frame_size, hop_size)` combination:

1. **Synthetic sine tone:** 440Hz stereo sine, 5 seconds at 48kHz (deterministic baseline)
2. **`tests/test_tracks_wav/PinkPanther.wav`:** loaded via `hound`, truncated to 5 seconds
3. **`tests/test_tracks_wav/TVSong.wav`:** loaded via `hound`, truncated to 5 seconds

Real audio exercises the DSP's spectral processing (prototype computation, estimator state) more representatively than a pure sine tone.

### Measurement

For each audio source and each `(callback_size, frame_size, hop_size)` combination:

- Feed the audio in `callback_size`-sample chunks via `ChorusDsp::process()`
- Measure wall-clock time per call using `std::time::Instant`
- Collect: mean callback time (μs), p95 callback time (μs), max callback time (μs)
- Record `samples_until_first_output`: the total input samples consumed before the first non-empty `Vec` is returned (algorithmic latency in samples)

### Pass/Fail Criterion

For real-time viability: `p95 callback time ≤ callback_duration` where `callback_duration = callback_size / 48000 * 1e6` microseconds.

Report: ✓ real-time safe / ✗ not safe per combination.

---

## Parameter Grid

### Callback sizes (simulating host behavior)

| Host | Typical callback sizes |
|---|---|
| Traktor Pro | 256, 512 |
| Serato DJ Pro | 128, 256, 512 |
| Mixxx | 64, 128, 256, 512 |
| VirtualDJ | 256, 512, 1024 |

Sweep `callback_size ∈ [64, 128, 256, 512, 1024]`.

### STFT parameter grid

| frame_size | hop_size | Algorithmic latency (samples) | Latency (ms @ 48kHz) | Freq bins |
|---|---|---|---|---|
| 256 | 128 | 256 | ~5.3ms | 129 |
| 512 | 256 | 512 | ~10.7ms | 257 |
| 1024 | 512 | 1024 | ~21.3ms | 513 |

---

## Output

The harness prints a results table to stdout and writes `docs/profiling/YYYY-MM-DD-streaming-latency.md` with:

1. Full results table (callback_size × frame_size, per audio source)
2. Per-host recommendation based on measured results
3. Summary: which configurations are real-time safe on the development machine (Apple Silicon or x86, reported from `uname`)

---

## Recommended Defaults (to be finalized post-profiling)

Pending profiling results, the expected recommendation is:

| Profile | frame_size | hop_size | Rationale |
|---|---|---|---|
| Default | 1024 | 512 | Current; ~21ms, 513 frequency bins |
| Low-latency | 512 | 256 | ~10ms, 257 bins — suitable for Traktor/Serato |

If profiling confirms that `frame_size=512` meets the real-time constraint for all target callback sizes, update `DspConfig::default()` or add `DspConfig::low_latency()` preset.

Host-specific preset constants in `processor.rs`:

```rust
impl DspConfig {
    /// Low-latency preset suitable for Traktor Pro / Serato DJ Pro (~10ms algorithmic latency).
    pub fn low_latency() -> Self { ... }
}
```

Only add this if profiling results show a meaningful difference from the default.

---

## Sanity Test

Before timing begins, the harness runs a correctness check: feed 4096 samples of sine, verify the process returns non-empty output and all samples are finite. Panics with a helpful message if the DSP is misconfigured. This ensures the benchmark exercises real processing rather than a no-op.

---

## Deliverables

1. `rust/chorus-dsp/benches/streaming_latency.rs` — benchmark binary
2. `docs/profiling/2026-05-22-streaming-latency.md` — results and recommendations
3. `DspConfig` changes (if profiling recommends) in `processor.rs`
4. Updated `DspConfig::default()` or new preset if warranted

---

## Non-Goals

- Measuring latency inside an actual JUCE plugin host (host installs required; deferred)
- Profiling `StreamingFrft` or `StreamingWavelet` (STFT is the primary plugin transform)
- Automated CI integration of the benchmark (results are machine-dependent)
