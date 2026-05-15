# CHORUS Version Roadmap

Date: 2026-05-15

Project: CHORUS, "CHORUS Harmonizes Outputs from Reconstructed Upmixed Signals"

## Summary

CHORUS development after v0 proceeds through three sequential versions:

1. v1 validates non-STFT transforms and adds rich reporting.
2. v2 adds DJ-oriented contribution filter chains.
3. v3 ports the DSP semantics into a Rust core embedded in a JUCE plugin wrapper.

Each version depends on the previous version's acceptance gates. Later versions should not bypass earlier validation work.

## Version Dependencies

### v0 Baseline

v0 is the offline Python reference splitter. It reads stereo input, uses STFT as the trusted transform path, produces three stereo stems, and proves pass-through reconstruction and synthetic routing fixtures.

### v1 Depends On v0

v1 does not replace the STFT splitter. It adds transform diagnostics and report artifacts around the trusted v0 behavior. FrFT and Wavelet adapters remain research-grade analyzers until evidence supports promotion.

### v2 Depends On v1

v2 uses the v1 reporting and contribution visibility to add filters on the six reconstructed contribution signals. It must prove transparent unity-bypass behavior before any EQ or filtering output is trusted.

### v3 Depends On v2

v3 ports the Python reference semantics into Rust and embeds that core in JUCE. Rust behavior must be validated against Python v2 reference fixtures before plugin behavior is considered valid.

## Promotion Gates

### v1 Promotion Gate

- STFT v0 split output remains correct.
- `report.md` and associated plot artifacts are generated deterministically.
- FrFT and Wavelet diagnostics run without claiming production splitting support.
- Transform comparison metrics are clear enough to decide whether a non-STFT path should be promoted later.

### v2 Promotion Gate

- Unity-bypass filtering matches v1 outputs within tolerance.
- Each of `Lc`, `Rc`, `Lo`, `Ro`, `Ls`, and `Rs` can be filtered independently.
- Reports record filter chains and pre/post levels per contribution.
- Invalid filter configurations fail before audio is written.

### v3 Promotion Gate

- Rust DSP fixtures match Python v2 reference fixtures within tolerance.
- The JUCE wrapper loads in at least one plugin host.
- Bypass is transparent within tolerance in plugin form.
- Contribution EQ controls affect the intended components.
- Plugin state saves and reloads without parameter drift.

## Non-Goals

- v1 does not make FrFT or Wavelet authoritative split engines.
- v2 does not add transform-bin, tile-level, or spectral-mask filtering.
- v3 does not require every DJ host to expose six contributions or three stereo output buses.

## Spec Files

- `2026-05-15-chorus-v1-transform-validation-reporting.md`
- `2026-05-15-chorus-v2-contribution-filtering.md`
- `2026-05-15-chorus-v3-rust-juce-plugin.md`
