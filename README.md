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

## v1 Reports

Every `chorus split` run writes:

- `center.wav`
- `only.wav`
- `surround.wav`
- `report.json`
- `report.md`
- `spectrograms/input_left.png`
- `spectrograms/input_right.png`
- `spectrograms/Lc.png`
- `spectrograms/Rc.png`
- `spectrograms/Lo.png`
- `spectrograms/Ro.png`
- `spectrograms/Ls.png`
- `spectrograms/Rs.png`

STFT remains the default reference transform. FrFT and Wavelet can be selected for v1 reconstruction experiments, but reports label them separately from the STFT reference path.

## Acceptance Gate

Transform pass-through tests are mandatory. If unchanged stereo cannot round-trip through the STFT, FrFT, and Wavelet analyzer/reconstructor paths within tolerance, v1 transform validation is not considered valid.

STFT remains the trusted split path. FrFT and Wavelet passing the round-trip gate means they are reconstruction-capable research transforms, not promoted production split engines.

## Roadmap

v1 adds transform validation, reconstruction-capable FrFT and Wavelet experiments, and report artifacts with spectrograms for all upmixed channels.

v2 will enable the application of arbitrary filters to each upmixed component.

v3 will be a rewrite in Rust with VST/JUCE/AU/CLAP plugin integration.

## v2 Contribution Filters

Filters are configured per reconstructed contribution:

```json
{
  "Lc": [{"type": "gain", "db": -6.0}],
  "Rc": [{"type": "unity"}],
  "Lo": [{"type": "eq", "mode": "highpass", "frequency_hz": 120.0, "q": 0.707}],
  "Ro": [{"type": "polarity"}],
  "Ls": [{"type": "mute"}],
  "Rs": [{"type": "solo"}]
}
```

Run:

```bash
poetry run chorus split input.wav --out-dir out --filters filters.json
```

Valid contribution names are `Lc`, `Rc`, `Lo`, `Ro`, `Ls`, and `Rs`. Invalid contribution names, unsupported filter types, and invalid parameter ranges fail before output audio is written.

## v3 Rust/JUCE Validation

Generate Python v2 fixtures:

```bash
poetry run python scripts/generate_v2_fixtures.py
```

Run Rust validation:

```bash
cargo test --manifest-path rust/chorus-dsp/Cargo.toml
cargo test --manifest-path rust/chorus-ffi/Cargo.toml
```

Build the plugin after installing JUCE:

```bash
test -n "$JUCE_DIR"
cmake -S plugin -B plugin/build -DJUCE_DIR="$JUCE_DIR"
cmake --build plugin/build
ctest --test-dir plugin/build --output-on-failure
```

v3 is not valid until Rust fixture tests match Python v2 outputs and the JUCE wrapper loads in at least one plugin host or validator.
