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

## Roadmap

v1 will validate and enable FrFT and wavelet adapters. It will also add `report.md` with spectrograms for all upmixed channels.

v2 will enable the application of arbitrary filters to each upmixed component.

v3 will be a rewrite in Rust with VST/JUCE/AU/CLAP plugin integration.
