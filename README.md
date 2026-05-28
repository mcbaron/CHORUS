# CHORUS

CHORUS Harmonizes Outputs from Reconstructed Upmixed Signals.

Reads a stereo WAV `(L, R)` and writes three stereo stems:

- `center.wav` — `[Lc, Rc]`
- `only.wav` — `[Lo, Ro]`
- `surround.wav` — `[Ls, Rs]`
- `report.json` — settings, levels, and reconstruction checks
- `report.md` — human-readable summary
- `spectrograms/` — per-contribution spectrograms

Available as a Python CLI (reference) and a Rust binary (streaming/real-time).

---

## Install

### Python

```bash
poetry install
```

### Rust

```bash
cargo build --release --manifest-path rust/chorus-dsp/Cargo.toml
```

The binary is at `rust/chorus-dsp/target/release/chorus_split`.

---

## Run

### Python — single file

```bash
poetry run chorus split input.wav --out-dir out/ --transform stft
```

Options:

| Flag | Default | Values |
|---|---|---|
| `--transform` | `stft` | `stft`, `frft`, `wavelet` |
| `--frame-size` | `1024` | |
| `--hop-size` | `512` | |
| `--smoothing-alpha` | `0.9` | |
| `--epsilon` | `1e-9` | |
| `--filters` | _(none)_ | path to JSON filter config |

### Rust — single file

```bash
rust/chorus-dsp/target/release/chorus_split input.wav out/ --transform stft
```

Supports `--transform stft`, `frft`, and `wavelet`.

---

## Process a directory

### Python

```bash
for f in /path/to/wavs/*.wav; do
  name=$(basename "$f" .wav)
  poetry run chorus split "$f" --out-dir "output/${name}/stft" --transform stft
  poetry run chorus split "$f" --out-dir "output/${name}/frft" --transform frft
  poetry run chorus split "$f" --out-dir "output/${name}/wavelet" --transform wavelet
done
```

### Rust

```bash
for f in /path/to/wavs/*.wav; do
  name=$(basename "$f" .wav)
  rust/chorus-dsp/target/release/chorus_split "$f" "output/rust/${name}/stft" --transform stft
  rust/chorus-dsp/target/release/chorus_split "$f" "output/rust/${name}/frft" --transform frft
  rust/chorus-dsp/target/release/chorus_split "$f" "output/rust/${name}/wavelet" --transform wavelet
done
```

---

## Contribution Filters (v2)

Filter chains are applied per-contribution after reconstruction. Configure via JSON:

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

```bash
poetry run chorus split input.wav --out-dir out/ --filters filters.json
```

Valid contributions: `Lc`, `Rc`, `Lo`, `Ro`, `Ls`, `Rs`. Invalid names, unsupported filter types, or out-of-range parameters fail before any audio is written.

Supported filter types: `unity`, `gain` (`db`), `polarity`, `mute`, `solo`, `eq` (`mode`: `highpass`/`lowpass`/`peaking`, `frequency_hz`, `q`, `gain_db`).

---

## Transforms

STFT is the reference path. FrFT and Wavelet are reconstruction-capable experimental transforms.

| Transform | Python | Rust | Notes |
|---|---|---|---|
| STFT | reference | fixture-matched | sqrt-Hann, 50% overlap, 1024-sample frames |
| FrFT | OLA (sqrt-Hann, 50%) | Ozaktas-Kutay | Rust/Python fixture parity pending algorithm alignment |
| Wavelet | db4, level 3 | fixture-matched | 512-sample frames, overlap-save |

---

## Validation

### Python tests

```bash
poetry run pytest
```

### Rust tests

```bash
cargo test --manifest-path rust/chorus-dsp/Cargo.toml
cargo test --manifest-path rust/chorus-ffi/Cargo.toml
```

### Regenerate Python v2 fixtures

```bash
poetry run python scripts/generate_v2_fixtures.py
```

---

## Known gaps

- `rust_matches_python_frft_fixture` — `#[ignore]`d: Python FrFT uses a phase-shifted FFT approximation; Rust implements true Ozaktas-Kutay. Parity requires updating Python to match.
- `rust_matches_python_known_eq_preset_fixture` — `#[ignore]`d: EQ biquad implementation pending for the non-STFT (wavelet, FrFT) paths.
- JUCE plugin wrapper — not yet built; planned for v3.

---

## Roadmap

- **v0** — Python offline splitter with STFT ✓
- **v1** — FrFT/Wavelet adapters, spectrogram reports ✓
- **v2** — Per-contribution filter chains ✓
- **v3** — Rust DSP core (streaming, real-time) ✓ / JUCE plugin wrapper (pending)
