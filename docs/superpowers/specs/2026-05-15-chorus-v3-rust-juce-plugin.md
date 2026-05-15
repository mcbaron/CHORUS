# CHORUS v3 Rust Core And JUCE Plugin Design

Date: 2026-05-15

## Summary

CHORUS v3 ports the DSP behavior from the Python reference implementation into a Rust core and embeds that Rust core inside a JUCE plugin wrapper.

Rust owns the algorithmic core. JUCE owns plugin host integration.

The Python v2 implementation remains the behavioral reference. Rust output must match Python v2 fixtures before plugin behavior is considered valid.

## Goals

- Implement a Rust DSP core matching Python v2 behavior.
- Embed the Rust core inside a JUCE plugin wrapper.
- Expose contribution controls suitable for DJ/plugin workflows.
- Support transparent bypass within tolerance.
- Validate Rust fixtures against Python v2 reference outputs.
- Load the plugin in at least one plugin host.

## Non-Goals

- v3 does not require every DJ host to expose six contribution buses.
- v3 does not require all plugin formats to ship at once.
- v3 does not redesign the DSP algorithm.
- v3 does not add transform-bin filtering.
- v3 does not make host-specific assumptions about Traktor, Mixxx, VirtualDJ, or Serato DJ without explicit validation.

## Architecture

The v3 system has three layers:

1. Python reference fixtures
2. Rust DSP core
3. JUCE plugin wrapper

### Python Reference Fixtures

Python v2 produces deterministic fixtures for:

- raw input
- six unfiltered contributions
- unity-bypass output
- known EQ/filter presets
- report metadata where relevant

These fixtures define expected Rust behavior.

### Rust DSP Core

Rust owns:

- transform processing selected for plugin v1 behavior
- prototype generation
- smoothed estimator behavior
- six contribution signals
- contribution filter chains
- parameter smoothing
- bypass behavior
- deterministic fixture output

The Rust core should have no JUCE dependency.

### JUCE Wrapper

JUCE owns:

- plugin format targets
- audio callback integration
- parameter exposure
- preset and state serialization
- host bus layout negotiation
- host compatibility testing

The JUCE wrapper calls into the Rust core through a narrow FFI or C-compatible boundary.

## Plugin Output Strategy

DJ hosts differ in how they support multi-output plugins. v3 must not assume every host can expose six internal contributions or three stereo output buses.

The plugin should support:

- stereo input
- internal six-contribution processing
- internal contribution controls
- stereo output mixdown as the baseline host-compatible path

If host support allows, additional bus layouts may expose:

- three stereo stems
- diagnostic contribution outputs

Those layouts are optional until validated in concrete hosts.

## Real-Time Requirements

The plugin audio callback must:

- avoid heap allocation
- avoid blocking I/O
- use bounded state
- handle parameter changes smoothly
- avoid panics across the FFI boundary
- process silence and near-silence stably

Parameter updates from JUCE should be smoothed or staged before they affect DSP state.

## Validation

Validation must compare Rust output against Python v2 fixtures for:

- unity bypass
- center-dominant stereo
- hard-panned left
- hard-panned right
- phase-inverted surround-like input
- known contribution EQ presets
- silence and near-silence

Tolerances should be explicit per fixture type. Bypass fixtures should be strict. EQ fixtures may allow small floating-point differences if justified and documented.

## Error Handling

- Rust errors crossing into JUCE must become safe plugin-state errors, not panics.
- Invalid parameter states must be clamped or rejected before audio processing.
- Unsupported host bus layouts must fail negotiation cleanly.
- Plugin state load failures must preserve the previous valid state.

## Testing

Required tests:

- Rust unit tests for estimator, prototype, transform, and filter behavior.
- Rust fixture tests against Python v2 outputs.
- JUCE wrapper smoke test for plugin instantiation.
- Plugin state save/load round-trip test.
- Bypass transparency test.
- Parameter smoothing test.
- Host-load test in at least one plugin host or plugin validation tool.

## Acceptance Criteria

v3 is complete when:

- Rust core matches Python v2 reference fixtures within tolerance.
- Rust core has no JUCE dependency.
- JUCE wrapper loads in at least one plugin host or validator.
- Bypass is transparent within tolerance.
- Contribution EQ controls affect the intended components.
- Plugin state saves and reloads without parameter drift.
- Real-time processing avoids allocation and blocking in the audio callback.
