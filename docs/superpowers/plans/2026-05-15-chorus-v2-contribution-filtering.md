# CHORUS v2 Contribution Filter Chains Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add serial per-contribution filter chains for `Lc`, `Rc`, `Lo`, `Ro`, `Ls`, and `Rs` while preserving transparent unity-bypass output.

**Architecture:** Insert a filter stage after v1 contribution reconstruction and before stem assembly. Every contribution always flows through a validated chain; bypass is represented as a unity filter rather than a separate code path. Reports compare pre-filter and post-filter contribution levels and explicitly mark whether output is transparent or intentionally altered.

**Tech Stack:** Python 3.12, Poetry, NumPy, SciPy signal filters, pytest, Ruff.

---

## File Structure

- Create `src/chorus/filters.py`: filter dataclasses, validation, chain application, solo/mute handling, and numeric stability checks.
- Modify `src/chorus/core.py`: add optional filter chains to `ChorusConfig`, apply filters to contributions, assemble stems from filtered contributions, and store pre/post metadata.
- Modify `src/chorus/reporting.py`: include filter chains, pre/post levels, transparency status, and filtered-output checks.
- Modify `src/chorus/cli.py`: accept `--filters path.json` for per-contribution filter configuration.
- Create `tests/test_filters.py`: unit tests for validation and filter behavior.
- Modify `tests/test_core.py`: unity-bypass parity and independent contribution filtering tests.
- Modify `tests/test_reporting.py`: filter metadata and non-transparent report tests.
- Modify `tests/test_io_cli.py`: CLI filter config integration test.
- Modify `README.md`: document filter config JSON.

## Preflight Gate

Do not start this plan until the v1 plan is fully implemented and verified. Confirm these commands pass before Task 1:

```bash
test -f src/chorus/reporting.py
test -f tests/test_reporting.py
poetry run pytest -q
```

Expected: `src/chorus/reporting.py` exists, `tests/test_reporting.py` exists, `ChorusResult.contributions` is present in `src/chorus/core.py`, `build_v1_report()` and `write_markdown_report()` are importable, and the complete v1 test suite passes.

## Task 1: Add Filter Model And Validation

**Files:**
- Create: `src/chorus/filters.py`
- Create: `tests/test_filters.py`

- [x] **Step 1: Write failing validation tests**

Create `tests/test_filters.py`:

```python
from __future__ import annotations

import pytest

from chorus.filters import CONTRIBUTIONS, FilterSpec, normalize_filter_config


def test_normalize_empty_filter_config_creates_unity_for_every_contribution() -> None:
    chains = normalize_filter_config(None)

    assert set(chains) == CONTRIBUTIONS
    for chain in chains.values():
        assert chain == [FilterSpec(type="unity", parameters={})]


def test_invalid_contribution_name_fails_before_audio_processing() -> None:
    with pytest.raises(ValueError, match="unknown contribution 'left_center'"):
        normalize_filter_config({"left_center": [{"type": "gain", "db": 3.0}]})


def test_unsupported_filter_type_fails_validation() -> None:
    with pytest.raises(ValueError, match="unsupported filter type 'comb' for Lc\\[0\\]"):
        normalize_filter_config({"Lc": [{"type": "comb"}]})


def test_invalid_gain_parameter_fails_validation() -> None:
    with pytest.raises(ValueError, match="gain db for Lc\\[0\\] must be between -60.0 and 24.0"):
        normalize_filter_config({"Lc": [{"type": "gain", "db": 48.0}]})


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"Lc": [{"type": "eq", "mode": "notch", "frequency_hz": 1000.0}]}, "eq mode"),
        ({"Lc": [{"type": "eq", "frequency_hz": 0.0}]}, "frequency_hz"),
        ({"Lc": [{"type": "eq", "frequency_hz": 24000.0}]}, "below Nyquist"),
        ({"Lc": [{"type": "eq", "frequency_hz": 1000.0, "q": 0.0}]}, "eq q"),
    ],
)
def test_invalid_eq_parameters_fail_validation(raw: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_filter_config(raw, sample_rate=48_000)
```

- [x] **Step 2: Run the tests to verify they fail**

Run:

```bash
poetry run pytest tests/test_filters.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement filter specs and validation**

Create `src/chorus/filters.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

CONTRIBUTIONS = frozenset({"Lc", "Rc", "Lo", "Ro", "Ls", "Rs"})
FILTER_TYPES = frozenset({"unity", "gain", "mute", "solo", "polarity", "eq"})


@dataclass(frozen=True)
class FilterSpec:
    type: str
    parameters: dict[str, float | str]


FilterChains = dict[str, list[FilterSpec]]


def _as_spec(target: str, index: int, raw: dict[str, Any], sample_rate: int | None) -> FilterSpec:
    filter_type = str(raw.get("type", ""))
    if filter_type not in FILTER_TYPES:
        raise ValueError(f"unsupported filter type {filter_type!r} for {target}[{index}]")
    parameters = {key: value for key, value in raw.items() if key != "type"}
    spec = FilterSpec(type=filter_type, parameters=parameters)
    _validate_spec(target, index, spec, sample_rate)
    return spec


def _validate_spec(target: str, index: int, spec: FilterSpec, sample_rate: int | None) -> None:
    if spec.type == "gain":
        db = float(spec.parameters.get("db", 0.0))
        if db < -60.0 or db > 24.0:
            raise ValueError(f"gain db for {target}[{index}] must be between -60.0 and 24.0")
    if spec.type == "eq":
        mode = str(spec.parameters.get("mode", "peaking"))
        if mode not in {"peaking", "highpass", "lowpass"}:
            raise ValueError(f"eq mode for {target}[{index}] must be peaking, highpass, or lowpass")
        frequency = float(spec.parameters.get("frequency_hz", 0.0))
        if frequency <= 0.0:
            raise ValueError(f"eq frequency_hz for {target}[{index}] must be positive")
        if sample_rate is not None and frequency >= sample_rate / 2.0:
            raise ValueError(f"eq frequency_hz for {target}[{index}] must be below Nyquist")
        q = float(spec.parameters.get("q", 0.707))
        if q <= 0.0:
            raise ValueError(f"eq q for {target}[{index}] must be positive")


def normalize_filter_config(
    raw: dict[str, list[dict[str, Any]]] | None,
    sample_rate: int | None = None,
) -> FilterChains:
    chains: FilterChains = {
        contribution: [FilterSpec(type="unity", parameters={})]
        for contribution in sorted(CONTRIBUTIONS)
    }
    if raw is None:
        return chains
    for target, filters in raw.items():
        if target not in CONTRIBUTIONS:
            raise ValueError(f"unknown contribution {target!r}")
        chain = [_as_spec(target, index, item, sample_rate) for index, item in enumerate(filters)]
        chains[target] = chain or [FilterSpec(type="unity", parameters={})]
    return chains
```

- [x] **Step 4: Run validation tests**

Run:

```bash
poetry run pytest tests/test_filters.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/chorus/filters.py tests/test_filters.py
git commit -m "feat: add contribution filter validation"
```

## Task 2: Apply Unity, Gain, Mute, Solo, And Polarity Filters

**Files:**
- Modify: `src/chorus/filters.py`
- Modify: `tests/test_filters.py`

- [x] **Step 1: Write failing filter behavior tests**

Append to `tests/test_filters.py`:

```python
import numpy as np

from chorus.filters import apply_filter_chains


def test_unity_filter_returns_matching_contributions() -> None:
    contributions = {name: np.ones(8, dtype=np.float64) for name in CONTRIBUTIONS}

    filtered, report = apply_filter_chains(contributions, normalize_filter_config(None), sample_rate=48_000)

    for name in CONTRIBUTIONS:
        np.testing.assert_allclose(filtered[name], contributions[name])
    assert report["transparent"] is True


@pytest.mark.parametrize("target", sorted(CONTRIBUTIONS))
def test_each_contribution_can_be_gain_adjusted_independently(target: str) -> None:
    contributions = {name: np.ones(8, dtype=np.float64) for name in CONTRIBUTIONS}
    chains = normalize_filter_config({target: [{"type": "gain", "db": 6.0}]})

    filtered, report = apply_filter_chains(contributions, chains, sample_rate=48_000)

    np.testing.assert_allclose(filtered[target], np.ones(8) * (10.0 ** (6.0 / 20.0)))
    for name in CONTRIBUTIONS - {target}:
        np.testing.assert_allclose(filtered[name], np.ones(8))
    assert report["transparent"] is False


def test_gain_and_polarity_affect_only_configured_contribution() -> None:
    contributions = {name: np.ones(8, dtype=np.float64) for name in CONTRIBUTIONS}
    chains = normalize_filter_config(
        {
            "Lc": [{"type": "gain", "db": 6.0}],
            "Rs": [{"type": "polarity"}],
        }
    )

    filtered, report = apply_filter_chains(contributions, chains, sample_rate=48_000)

    np.testing.assert_allclose(filtered["Lc"], np.ones(8) * (10.0 ** (6.0 / 20.0)))
    np.testing.assert_allclose(filtered["Rs"], -np.ones(8))
    np.testing.assert_allclose(filtered["Rc"], np.ones(8))
    assert report["transparent"] is False


def test_solo_mutes_non_solo_contributions() -> None:
    contributions = {name: np.ones(8, dtype=np.float64) for name in CONTRIBUTIONS}
    chains = normalize_filter_config({"Lo": [{"type": "solo"}]})

    filtered, report = apply_filter_chains(contributions, chains, sample_rate=48_000)

    np.testing.assert_allclose(filtered["Lo"], np.ones(8))
    for name in CONTRIBUTIONS - {"Lo"}:
        np.testing.assert_allclose(filtered[name], np.zeros(8))
    assert report["soloed"] == ["Lo"]
```

- [x] **Step 2: Run the tests to verify they fail**

Run:

```bash
poetry run pytest tests/test_filters.py::test_unity_filter_returns_matching_contributions tests/test_filters.py::test_gain_and_polarity_affect_only_configured_contribution tests/test_filters.py::test_solo_mutes_non_solo_contributions -q
```

Expected: FAIL because `apply_filter_chains` does not exist.

- [x] **Step 3: Implement filter application**

Append to `src/chorus/filters.py`:

```python
def _is_unity_chain(chain: list[FilterSpec]) -> bool:
    return len(chain) == 1 and chain[0].type == "unity"


def _apply_spec(audio: np.ndarray, spec: FilterSpec, sample_rate: int) -> np.ndarray:
    if spec.type == "unity":
        return audio
    if spec.type == "gain":
        return audio * (10.0 ** (float(spec.parameters.get("db", 0.0)) / 20.0))
    if spec.type == "mute":
        return np.zeros_like(audio)
    if spec.type == "solo":
        return audio
    if spec.type == "polarity":
        return -audio
    if spec.type == "eq":
        return _apply_eq(audio, spec, sample_rate)
    raise ValueError(f"unsupported filter type {spec.type!r}")


def _apply_eq(audio: np.ndarray, spec: FilterSpec, sample_rate: int) -> np.ndarray:
    mode = str(spec.parameters.get("mode", "peaking"))
    frequency = float(spec.parameters["frequency_hz"])
    q = float(spec.parameters.get("q", 0.707))
    gain_db = float(spec.parameters.get("gain_db", 0.0))
    if mode == "highpass":
        sos = signal.butter(2, frequency, btype="highpass", fs=sample_rate, output="sos")
    elif mode == "lowpass":
        sos = signal.butter(2, frequency, btype="lowpass", fs=sample_rate, output="sos")
    else:
        b, a = signal.iirpeak(frequency, q, fs=sample_rate)
        filtered = signal.lfilter(b, a, audio)
        return audio + filtered * (10.0 ** (gain_db / 20.0) - 1.0)
    return signal.sosfilt(sos, audio)


def _ensure_finite(name: str, audio: np.ndarray) -> None:
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"filtered output for {name} contains NaN or infinity")


def _level(audio: np.ndarray) -> dict[str, float]:
    return {
        "rms": float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0,
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
    }


def apply_filter_chains(
    contributions: dict[str, np.ndarray],
    chains: FilterChains,
    sample_rate: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    soloed = sorted(
        name for name, chain in chains.items() if any(spec.type == "solo" for spec in chain)
    )
    filtered: dict[str, np.ndarray] = {}
    pre_levels = {name: _level(audio) for name, audio in contributions.items()}
    post_levels: dict[str, dict[str, float]] = {}
    for name in sorted(CONTRIBUTIONS):
        audio = np.asarray(contributions[name], dtype=np.float64).copy()
        for spec in chains[name]:
            audio = _apply_spec(audio, spec, sample_rate)
        if soloed and name not in soloed:
            audio = np.zeros_like(audio)
        _ensure_finite(name, audio)
        filtered[name] = audio
        post_levels[name] = _level(audio)
    transparent = not soloed and all(_is_unity_chain(chain) for chain in chains.values())
    return filtered, {
        "chains": {
            name: [{"type": spec.type, "parameters": spec.parameters} for spec in chain]
            for name, chain in chains.items()
        },
        "transparent": transparent,
        "soloed": soloed,
        "pre_levels": pre_levels,
        "post_levels": post_levels,
    }
```

- [x] **Step 4: Run filter tests**

Run:

```bash
poetry run pytest tests/test_filters.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/chorus/filters.py tests/test_filters.py
git commit -m "feat: apply contribution filter chains"
```

## Task 3: Apply Filters In The Processor

**Files:**
- Modify: `src/chorus/core.py`
- Modify: `tests/test_core.py`

- [x] **Step 1: Write failing core filter tests**

Append to `tests/test_core.py`:

```python
from chorus.filters import normalize_filter_config


def test_unity_filter_chains_match_unfiltered_v1_output(
    stereo_identical: np.ndarray, sample_rate: int
) -> None:
    baseline = ChorusProcessor(ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0)).process(stereo_identical)
    filtered = ChorusProcessor(
        ChorusConfig(
            sample_rate=sample_rate,
            smoothing_alpha=0.0,
            filter_chains=normalize_filter_config(None),
        )
    ).process(stereo_identical)

    np.testing.assert_allclose(filtered.center, baseline.center, atol=1e-10)
    np.testing.assert_allclose(filtered.only, baseline.only, atol=1e-10)
    np.testing.assert_allclose(filtered.surround, baseline.surround, atol=1e-10)
    assert filtered.metadata["filters"]["transparent"] is True


def test_gain_filter_changes_only_target_contribution(
    stereo_identical: np.ndarray, sample_rate: int
) -> None:
    chains = normalize_filter_config({"Lc": [{"type": "gain", "db": -6.0}]})
    result = ChorusProcessor(
        ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0, filter_chains=chains)
    ).process(stereo_identical)

    assert result.metadata["filters"]["transparent"] is False
    assert result.contributions["Lc"].shape == result.contributions["Rc"].shape
    assert not np.allclose(result.contributions["Lc"], result.contributions["Rc"])
```

- [x] **Step 2: Run the tests to verify they fail**

Run:

```bash
poetry run pytest tests/test_core.py::test_unity_filter_chains_match_unfiltered_v1_output tests/test_core.py::test_gain_filter_changes_only_target_contribution -q
```

Expected: FAIL because `ChorusConfig` has no `filter_chains`.

- [x] **Step 3: Wire filter chains into core processing**

In `src/chorus/core.py`, add imports:

```python
from chorus.filters import FilterChains, apply_filter_chains, normalize_filter_config
```

Change `ChorusConfig`:

```python
@dataclass(frozen=True)
class ChorusConfig:
    sample_rate: int
    transform: str = "stft"
    frame_size: int = 1024
    hop_size: int = 512
    smoothing_alpha: float = 0.9
    epsilon: float = 1e-9
    filter_chains: FilterChains | None = None
```

After the initial contribution dictionary is created, apply filters and assemble stems:

```python
        raw_contributions = {
            "Lc": center[:, 0],
            "Rc": center[:, 1],
            "Lo": only[:, 0],
            "Ro": only[:, 1],
            "Ls": surround[:, 0],
            "Rs": surround[:, 1],
        }
        filter_chains = self.config.filter_chains or normalize_filter_config(None)
        filtered_contributions, filter_report = apply_filter_chains(
            raw_contributions,
            filter_chains,
            self.config.sample_rate,
        )
        center = np.column_stack([filtered_contributions["Lc"], filtered_contributions["Rc"]])
        only = np.column_stack([filtered_contributions["Lo"], filtered_contributions["Ro"]])
        surround = np.column_stack([filtered_contributions["Ls"], filtered_contributions["Rs"]])
```

Set the result contributions to `filtered_contributions` and include metadata:

```python
                "filters": filter_report,
```

- [x] **Step 4: Run core and filter tests**

Run:

```bash
poetry run pytest tests/test_filters.py tests/test_core.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/chorus/core.py tests/test_core.py
git commit -m "feat: filter chorus contributions in core"
```

## Task 4: Load Filter Configs From The CLI

**Files:**
- Modify: `src/chorus/cli.py`
- Modify: `tests/test_io_cli.py`

- [x] **Step 1: Write failing CLI filter config test**

Append to `tests/test_io_cli.py`:

```python
def test_cli_split_accepts_filter_config(
    tmp_path, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    input_path = tmp_path / "input.wav"
    out_dir = tmp_path / "out"
    filters_path = tmp_path / "filters.json"
    wavfile.write(input_path, sample_rate, stereo_identical.astype(np.float32))
    filters_path.write_text(json.dumps({"Lc": [{"type": "gain", "db": -6.0}]}), encoding="utf-8")

    exit_code = main(
        [
            "split",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--transform",
            "stft",
            "--filters",
            str(filters_path),
        ]
    )

    assert exit_code == 0
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["filters"]["transparent"] is False
    assert report["filters"]["chains"]["Lc"][0]["type"] == "gain"


def test_cli_split_rejects_invalid_filter_config_before_writing_outputs(
    tmp_path, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    input_path = tmp_path / "input.wav"
    out_dir = tmp_path / "out"
    filters_path = tmp_path / "filters.json"
    wavfile.write(input_path, sample_rate, stereo_identical.astype(np.float32))
    filters_path.write_text(json.dumps({"bad": [{"type": "gain", "db": 1.0}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown contribution"):
        main(["split", str(input_path), "--out-dir", str(out_dir), "--filters", str(filters_path)])

    assert not out_dir.exists()
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
poetry run pytest tests/test_io_cli.py::test_cli_split_accepts_filter_config -q
```

Expected: FAIL because `--filters` is not accepted.

- [x] **Step 3: Add CLI config loading**

In `src/chorus/cli.py`, add to the split parser:

```python
    split.add_argument("--filters", type=Path, default=None)
```

Add imports:

```python
from chorus.filters import normalize_filter_config
```

Before constructing `ChorusConfig`, load filters:

```python
        raw_filters = None
        if args.filters is not None:
            raw_filters = json.loads(args.filters.read_text(encoding="utf-8"))
        filter_chains = normalize_filter_config(raw_filters, sample_rate=sample_rate)
```

Pass `filter_chains=filter_chains` into `ChorusConfig`.

- [x] **Step 4: Run CLI tests**

Run:

```bash
poetry run pytest tests/test_io_cli.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/chorus/cli.py tests/test_io_cli.py
git commit -m "feat: load contribution filter configs"
```

## Task 5: Report Filter Chains And Filtered Checks

**Files:**
- Modify: `src/chorus/reporting.py`
- Modify: `tests/test_reporting.py`

- [ ] **Step 1: Write failing report metadata test**

Append to `tests/test_reporting.py`:

```python
from chorus.filters import normalize_filter_config


def test_report_records_filter_chains_and_filtered_output_status(
    tmp_path, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    result = ChorusProcessor(
        ChorusConfig(
            sample_rate=sample_rate,
            filter_chains=normalize_filter_config({"Lc": [{"type": "gain", "db": -6.0}]}),
        )
    ).process(stereo_identical)
    report = build_v1_report(
        tmp_path / "input.wav",
        tmp_path,
        sample_rate,
        stereo_identical,
        result,
        {"center": tmp_path / "center.wav", "only": tmp_path / "only.wav", "surround": tmp_path / "surround.wav"},
    )

    assert report["filters"]["transparent"] is False
    assert report["checks"]["filtered_output"]["intentionally_altered"] is True
    assert report["checks"]["reconstruction"]["reference"] == "v1-transparent"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
poetry run pytest tests/test_reporting.py::test_report_records_filter_chains_and_filtered_output_status -q
```

Expected: FAIL because filter report fields are missing.

- [ ] **Step 3: Add filter data to reports**

In `src/chorus/reporting.py`, add imports:

```python
from chorus.filters import normalize_filter_config
```

Update `build_v1_report()` to include a complete fallback filter report:

```python
    filters = result.metadata.get("filters")
    if filters is None:
        filters = {
            "transparent": True,
            "chains": {
                name: [{"type": spec.type, "parameters": spec.parameters} for spec in chain]
                for name, chain in normalize_filter_config(None).items()
            },
            "pre_levels": {},
            "post_levels": {},
            "soloed": [],
        }
```

Change `checks` to:

```python
        "checks": {
            "reconstruction": {
                "reference": "v1-transparent",
                "passed": bool(_residual_summary(input_audio, reconstructed)["max_abs"] <= 1e-6),
                **_residual_summary(input_audio, reconstructed),
            },
            "filtered_output": {
                "intentionally_altered": not bool(filters.get("transparent", True)),
                "finite": bool(np.all(np.isfinite(reconstructed))),
            },
        },
        "filters": filters,
```

In `write_markdown_report()`, add after output stems:

```python
    lines.extend(["", "## Filters", f"- Transparent bypass: `{report['filters']['transparent']}`"])
    for name, chain in report["filters"]["chains"].items():
        lines.append(f"- {name}: `{chain}`")
```

- [ ] **Step 4: Run reporting tests**

Run:

```bash
poetry run pytest tests/test_reporting.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chorus/reporting.py tests/test_reporting.py
git commit -m "feat: report contribution filters"
```

## Task 6: Document Filter Configuration

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add filter usage docs**

Append to `README.md`:

```markdown
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
```

- [ ] **Step 2: Run full verification**

Run:

```bash
poetry run pytest -q
poetry run ruff check .
```

Expected: all tests pass and Ruff reports no violations.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document contribution filters"
```

## Self-Review

- Spec coverage: The plan adds six contribution filter chains, validates target names and filter types before audio processing/output writing, models bypass as unity, supports gain/mute/solo/polarity/EQ modes, tests gain across all six contributions, validates EQ parameter bounds including Nyquist, and records pre/post levels in reports.
- Placeholder scan: No `TBD`, `TODO`, or vague error-handling instructions remain.
- Type consistency: `FilterSpec`, `FilterChains`, `normalize_filter_config()`, and `apply_filter_chains()` are introduced before use. `ChorusConfig.filter_chains` stores normalized `FilterChains`; CLI and tests normalize before construction, and core only supplies the unity default when the field is `None`.
- Boundary check: Filtering is post-reconstruction contribution filtering only; no transform-bin, spectral-mask, or tile-level filtering is introduced.
