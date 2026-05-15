# CHORUS v1 Transform Validation And Reporting Design

Date: 2026-05-15

## Summary

CHORUS v1 expands the offline Python reference demo with research-grade transform comparison and richer report artifacts. STFT remains the only trusted splitting path. FrFT and Wavelet adapters are enabled as analyzers for diagnostics, comparison metrics, and report generation, not as production split engines.

The primary output addition is a human-readable `report.md` with spectrogram images for the input stereo channels and all six reconstructed contribution signals.

## Goals

- Preserve v0 STFT splitting behavior and acceptance tests.
- Generate `report.md` alongside `report.json`.
- Generate deterministic spectrogram image artifacts for:
  - input left
  - input right
  - `Lc`
  - `Rc`
  - `Lo`
  - `Ro`
  - `Ls`
  - `Rs`
- Run FrFT and Wavelet analyzers in comparison mode.
- Report transform diagnostics without promoting non-STFT transforms to trusted split engines.

## Non-Goals

- FrFT and Wavelet do not write authoritative split stems in v1.
- FrFT and Wavelet do not need production-quality inverse reconstruction in v1.
- v1 does not add contribution filters.
- v1 does not add plugin or real-time host integration.

## Architecture

v1 adds a reporting and analysis layer around the v0 processor.

The STFT processor remains responsible for:

- input splitting
- prototype generation
- contribution estimation
- stem reconstruction
- reconstruction checks

The v1 report layer is responsible for:

- collecting six contribution signals from the STFT split result
- rendering spectrograms
- running transform analyzers
- generating `report.md`
- expanding `report.json` with transform comparison metadata

FrFT and Wavelet adapters should expose analyzer results through a shared interface. That interface should report metadata such as transform name, configuration, output dimensions, energy summaries, execution time, reconstruction support status, and warnings.

## Data Flow

1. Read stereo input.
2. Run the v0 STFT split.
3. Collect input channels and six contribution signals.
4. Generate spectrogram image files.
5. Run STFT, FrFT, and Wavelet analyzers for comparison metrics.
6. Write `report.json`.
7. Write `report.md` referencing the generated images and metrics.

## Report Requirements

`report.md` must include:

- input file metadata
- output stem paths
- v0 reconstruction status
- per-stem RMS and peak levels
- per-contribution spectrogram links or embeds
- transform comparison table
- clear warnings that FrFT and Wavelet are research analyzers in v1

`report.json` must include:

- all v0 report fields
- spectrogram artifact paths
- transform analyzer metadata
- transform comparison metrics
- warning list

## Error Handling

- If spectrogram generation fails, the CLI should fail and not write a misleading complete report.
- If a research analyzer fails, the STFT split may still complete, but the report must record the analyzer failure clearly.
- Analyzer failures must not change STFT output stems.
- Missing or unsupported inverse reconstruction for FrFT/Wavelet must be reported as unsupported, not as failed STFT behavior.

## Testing

Required tests:

- v0 STFT split fixtures still pass.
- `report.md` is generated for a deterministic fixture.
- `report.md` references every generated spectrogram artifact.
- Spectrogram generation produces deterministic file names and non-empty image files.
- `report.json` includes transform comparison metadata.
- FrFT analyzer runs and is labeled research/experimental.
- Wavelet analyzer runs and is labeled research/experimental.
- FrFT/Wavelet analyzer failure is reported without corrupting STFT stem output.

## Acceptance Criteria

v1 is complete when:

- STFT split output remains compatible with v0 fixtures.
- CLI output includes `report.md`, `report.json`, and spectrogram images.
- The report includes input stereo and all six contribution spectrograms.
- FrFT and Wavelet analyzers run in comparison mode.
- Reports do not claim FrFT or Wavelet are authoritative split engines.
- Test coverage proves deterministic reporting and analyzer metadata behavior.
