# CHORUS v2 Contribution Filter Chains Design

Date: 2026-05-15

## Summary

CHORUS v2 adds DJ-oriented filtering for the six reconstructed contribution signals:

- `Lc`
- `Rc`
- `Lo`
- `Ro`
- `Ls`
- `Rs`

Filtering happens after contribution reconstruction and before final stem writing. There is no transform-bin, spectral-mask, or time-frequency tile filtering in v2.

The purpose is to let DJ software or offline workflows apply EQ and related controls to each CHORUS contribution.

## Goals

- Add a contribution filter engine for the six contribution signals.
- Represent bypass as a unity-gain filter with no poles and no zeros.
- Support per-contribution filter chains through configuration.
- Preserve transparent output when all contribution chains use unity bypass.
- Record full filter chains in reports.
- Shape the filter model so v3 can expose it as plugin parameters.

## Non-Goals

- No transform-bin or time-frequency tile filtering.
- No spectral-mask editing.
- No plugin UI in v2.
- No Rust or JUCE rewrite in v2.
- No host automation in v2.

## Architecture

v2 inserts a filter stage between contribution reconstruction and stem assembly.

The core flow is:

1. Run the v1 STFT split.
2. Produce six contribution signals.
3. Apply one filter chain to each contribution.
4. Assemble filtered contributions into three stereo stems:
   - `center.wav = [Lc, Rc]`
   - `only.wav = [Lo, Ro]`
   - `surround.wav = [Ls, Rs]`
5. Generate reports with pre/post contribution metrics.

Every contribution always passes through a filter chain. Bypass is not a separate code path. The bypass chain contains a unity-gain filter with no poles and no zeros.

## Filter Model

The initial filter model should support:

- unity bypass
- gain
- mute
- solo
- polarity inversion
- parametric EQ-style bands
- high-pass as an EQ mode
- low-pass as an EQ mode

Filter chains should be serial and ordered. Each filter instance should have a stable type name, parameter dictionary, and validation rules.

## Configuration

Filter configuration must name contribution targets explicitly. Valid targets are:

- `Lc`
- `Rc`
- `Lo`
- `Ro`
- `Ls`
- `Rs`

Unknown contribution names must fail validation before audio is processed. Unsupported filter types or invalid parameter ranges must also fail before audio is written.

## Reporting

Reports must include:

- filter chain per contribution
- pre-filter RMS and peak per contribution
- post-filter RMS and peak per contribution
- whether the run is transparent bypass or intentionally altered
- output stem paths
- reconstruction or residual checks appropriate for filtered output

Because filtered output is intentionally allowed to differ from input, reports must distinguish between:

- v1-style transparent reconstruction checks
- v2 filtered-output checks

## Error Handling

- Invalid filter configs fail before stem writing.
- A malformed filter chain identifies the contribution and filter index that failed.
- Numeric instability, NaN, or infinity in filtered output fails the run.
- Solo and mute interactions must be deterministic and recorded in the report.

## Testing

Required tests:

- Unity bypass matches v1 outputs within tolerance.
- Each of the six contributions can be independently gain-adjusted.
- Polarity inversion affects only the configured contribution.
- Invalid contribution names fail validation.
- Unsupported filter types fail validation.
- Invalid EQ parameters fail validation.
- Reports record every filter chain and pre/post levels.
- Filtered output does not claim transparent reconstruction when non-unity filters are active.

## Acceptance Criteria

v2 is complete when:

- All six contributions pass through the filter engine.
- Unity-bypass output matches v1 output within tolerance.
- Contribution-level EQ/gain controls can be applied independently.
- Reports fully describe the applied filter chains.
- Invalid configs fail clearly before output audio is written.
- No transform-bin or tile-level filtering exists in v2.
