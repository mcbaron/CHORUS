# CHORUS v0 Reference Design

Date: 2026-05-15

Project: CHORUS, "CHORUS Harmonizes Outputs from Reconstructed Upmixed Signals"

Source reference: `US9078077.pdf`, "Estimation of Synthetic Audio Prototypes with Frequency-Based Input Signal Decomposition"

## Summary

CHORUS v0 is a Python offline reference demo for the patent-shaped stereo contribution design. It reads a stereo audio file `(L, R)` and writes three stereo output files:

- `center.wav`: `[Lc, Rc]`
- `only.wav`: `[Lo, Ro]`
- `surround.wav`: `[Ls, Rs]`

The six contribution signals are:

- `Lc`: left channel contribution to center
- `Rc`: right channel contribution to center
- `Lo`: residual left-only contribution
- `Ro`: residual right-only contribution
- `Ls`: left channel contribution to surround
- `Rs`: right channel contribution to surround

The v0 implementation is offline-first, but the design keeps the DSP core separate from file I/O and shaped around explicit frame/block state. That boundary is required so the same algorithmic structure can later be ported to a real-time plugin core for DJ software hosts such as Traktor, Mixxx, VirtualDJ, or Serato DJ.

## Goals

- Implement a repeatable offline Python demo that processes a stereo file into three stereo contribution files.
- Keep the algorithmic core independent of the offline CLI.
- Use STFT as the reference transform path.
- Include FrFT and wavelet transform adapters as experimental options behind the same interface.
- Produce a compact report with settings, output levels, and reconstruction checks.
- Treat STFT pass-through reconstruction as a hard acceptance gate.

## Non-Goals

- No VST, AU, CLAP, JUCE, or DJ-host integration in v0.
- No GUI in v0.
- No real-time audio callback in v0.
- No claim that FrFT or wavelet paths are reference-quality in v0.
- No subjective audio-quality scoring automation in v0.

## Architecture

The repo will contain a small Python package with a shared DSP core and a thin offline CLI wrapper.

The offline CLI:

1. Reads a stereo WAV file.
2. Builds a processing configuration.
3. Runs the stereo samples through the shared core.
4. Writes `center.wav`, `only.wav`, `surround.wav`.
5. Writes `report.json` and optionally `report.md`.

The shared core:

1. Accepts stereo blocks or frames.
2. Converts them into a time-frequency representation.
3. Generates local nonlinear prototypes for center and surround-like content.
4. Estimates smoothed linear contributions from the original input components.
5. Computes only/residual contributions.
6. Reconstructs contribution signals back to time-domain stereo stems.

Even though the v0 CLI may read a complete file into memory, the DSP objects should expose explicit stateful processing boundaries: construct with config, process input frames/blocks, maintain estimator state, and finalize or flush output. The design should not bake in offline-only assumptions.

## Package Surface

Expected modules:

- `chorus.core`: configuration dataclasses, block processor, shared contracts, result objects.
- `chorus.transforms`: transform interface plus `STFTTransform`, `FrFTTransform`, and `WaveletTransform`.
- `chorus.prototypes`: center, surround, and only/residual prototype logic.
- `chorus.estimation`: smoothed least-squares covariance and coefficient estimation.
- `chorus.io`: WAV read/write helpers and report generation.
- `chorus.cli`: command-line entrypoint.

Expected CLI:

```bash
chorus split input.wav --out-dir out --transform stft
```

Optional flags should cover output sample format, frame size, hop size, smoothing coefficient or time constant, denominator floor, and report format.

## Transform Interface

The transform interface must hide representation details from the rest of the core while preserving enough metadata for reconstruction and reporting.

The reference STFT transform should use patent-aligned defaults:

- 1024-sample frame size
- 512-sample hop size
- sqrt-Hann analysis/synthesis window
- overlap-add reconstruction
- 44.1 kHz compatible defaults, without forbidding other sample rates

The FrFT and wavelet transform adapters are experimental in v0. They must conform to the same high-level contracts, but their reports must identify them as experimental and include any reconstruction limitations. If their inverse paths are not robust enough for contribution reconstruction, they may initially be limited to smoke-test support and explicit unsupported-operation errors for full splitting.

## DSP Behavior

For each time-frequency tile, CHORUS works from input components `L` and `R`.

The center prototype `C` should represent shared, in-phase, center-like content. The patent's example scales the larger-magnitude input toward the smaller-magnitude input and averages the equal-length parts. The result is strongest when `L` and `R` are equal level and in phase, weaker when level or phase differs, and zero for hard-panned or phase-reversed content.

The surround prototype `S` should represent shared, out-of-phase, surround-like content. It is symmetric with the center prototype: strongest when `L` and `R` are equal level and out of phase, weaker as level differences increase or phase differences decrease.

The center and surround contribution estimates should be formed from the original input components using smoothed least-squares estimation. For v0, the reference six-output design estimates single-channel contributions:

- `Lc` from `L` using the center prototype
- `Rc` from `R` using the center prototype
- `Ls` from `L` using the surround prototype
- `Rs` from `R` using the surround prototype, with the sign/phase convention needed for surround asymmetry

Only/residual contributions are then:

- `Lo = L - Lc - Ls`
- `Ro = R - Rc - Rs`

The reconstructed stems are grouped as `[Lc, Rc]`, `[Lo, Ro]`, and `[Ls, Rs]`.

The estimation layer should compute and smooth auto/cross statistics over time. Defaults should include causal exponential smoothing and a denominator floor for silence and near-silence stability. These constants must be configurable and recorded in the report.

## Reporting

The v0 report must include:

- input path
- output directory
- transform type
- transform settings
- smoothing settings
- denominator floor
- input sample rate, duration, and channel count
- output stem paths
- per-stem RMS and peak levels
- reconstruction residual for left: `L - (Lc + Lo + Ls)`
- reconstruction residual for right: `R - (Rc + Ro + Rs)`
- pass/fail status for required checks
- warnings for experimental transforms

The residual report should include at least max absolute error and RMS error.

## Testing Strategy

STFT pass-through reconstruction is critical. The implementation must include a test that analyzes and reconstructs unchanged stereo through the STFT path and checks that the output matches the input within a tight numeric tolerance. If this fails, the v0 splitter should be considered untrustworthy.

Required tests:

- stereo input produces exactly three stereo outputs with expected shapes
- STFT pass-through round trip meets tolerance
- identical `L/R` input routes primarily to center contribution
- hard-panned left input routes primarily to left-only contribution
- hard-panned right input routes primarily to right-only contribution
- phase-inverted `L/R` input routes primarily to surround contribution
- silence and near-silence are numerically stable
- report generation includes required fields and residual checks
- FrFT adapter has an experimental smoke test
- wavelet adapter has an experimental smoke test

The initial acceptance suite should prioritize deterministic synthetic fixtures over subjective listening tests.

## Future Plugin Path

The v0 Python design is not the plugin implementation. It is the reference model.

Future plugin work should preserve these boundaries:

- host wrapper: VST/AU/CLAP/JUCE or another plugin framework
- real-time block adapter: maps host buffers to core processing blocks
- DSP core: transform, prototype generation, estimator state, reconstruction
- preset/config layer: exposes transform and estimation parameters safely

If the future plugin core is written in C++ or Rust, the Python v0 tests should serve as behavioral fixtures for the port.

## Acceptance Criteria

The v0 reference demo is complete when:

- `chorus split input.wav --out-dir out --transform stft` writes the three expected stereo files.
- The CLI writes a machine-readable report.
- STFT pass-through reconstruction passes the hard acceptance tolerance.
- Synthetic routing fixtures pass for center, left-only, right-only, and surround-like cases.
- Silence and near-silence do not produce NaNs, infinities, or unstable coefficients.
- FrFT and wavelet options are present as experimental adapters or are explicitly marked unsupported for full splitting with clear tests.
- The code structure keeps DSP core logic separate from offline file I/O.
