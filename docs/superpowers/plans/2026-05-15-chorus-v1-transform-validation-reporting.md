# CHORUS v1 Transform Validation And Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the offline Python reference splitter with deterministic spectrogram reports, transform diagnostics, and tolerance-gated FrFT/Wavelet reconstruction support while preserving v0 STFT behavior.

**Architecture:** Keep `ChorusProcessor` as the splitting boundary, but expand it to return six named contribution arrays before stem assembly. Add a reporting layer that renders spectrogram PNGs, writes Markdown and JSON reports, and records analyzer results for STFT, FrFT, and Wavelet. STFT remains the default reference path; FrFT and Wavelet become selectable reconstruction-capable engines only when their round-trip tolerance tests pass, and reports must still label their v1 status explicitly.

**Tech Stack:** Python 3.12, Poetry, NumPy, SciPy, PyWavelets, Matplotlib, pytest, Ruff.

---

## File Structure

- Modify `pyproject.toml`: add `matplotlib` for deterministic spectrogram rendering.
- Modify `README.md`: document v1 report outputs and selectable transforms.
- Modify `src/chorus/transforms.py`: add inverse support and component inversion for FrFT/Wavelet plus analyzer metadata helpers.
- Modify `src/chorus/core.py`: add six contribution arrays to `ChorusResult` and allow `stft`, `frft`, and `wavelet` processors.
- Create `src/chorus/reporting.py`: spectrogram rendering, transform analysis, Markdown report generation, and report JSON assembly.
- Modify `src/chorus/io.py`: keep WAV I/O, delegate report construction to `reporting`.
- Modify `src/chorus/cli.py`: write `report.json`, `report.md`, and spectrogram images for every split.
- Create `tests/test_reporting.py`: report artifact and analyzer behavior tests.
- Modify `tests/test_transforms.py`: FrFT/Wavelet round-trip and metadata tests.
- Modify `tests/test_core.py`: contribution array contract and non-STFT selectable processor tests.
- Modify `tests/test_io_cli.py`: CLI report artifact integration tests.

## Task 1: Add Reporting Dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`

- [x] **Step 1: Add the dependency**

Edit `pyproject.toml` dependencies:

```toml
dependencies = [
    "numpy>=1.26,<2.0",
    "scipy>=1.13,<2.0",
    "PyWavelets>=1.6,<2.0",
    "matplotlib>=3.8,<4.0",
]
```

- [x] **Step 2: Refresh the lockfile**

Run:

```bash
poetry lock
poetry install
```

Expected: `poetry.lock` changes and install succeeds without downgrading existing runtime dependencies.

- [x] **Step 3: Document v1 outputs**

Append this section to `README.md`:

```markdown
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
```

- [x] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock README.md
git commit -m "docs: document chorus v1 reporting outputs"
```

## Task 2: Preserve Six Contribution Signals

**Files:**
- Modify: `src/chorus/core.py`
- Modify: `tests/test_core.py`

- [x] **Step 1: Write the failing contribution contract test**

Append to `tests/test_core.py`:

```python
def test_processor_exposes_six_named_contributions(
    stereo_identical: np.ndarray, sample_rate: int
) -> None:
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0))

    result = processor.process(stereo_identical)

    assert set(result.contributions) == {"Lc", "Rc", "Lo", "Ro", "Ls", "Rs"}
    for contribution in result.contributions.values():
        assert contribution.shape == (stereo_identical.shape[0],)
    np.testing.assert_allclose(result.center, np.column_stack([result.contributions["Lc"], result.contributions["Rc"]]))
    np.testing.assert_allclose(result.only, np.column_stack([result.contributions["Lo"], result.contributions["Ro"]]))
    np.testing.assert_allclose(result.surround, np.column_stack([result.contributions["Ls"], result.contributions["Rs"]]))
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
poetry run pytest tests/test_core.py::test_processor_exposes_six_named_contributions -q
```

Expected: FAIL with `AttributeError` because `ChorusResult` has no `contributions`.

- [x] **Step 3: Add contributions to the result**

In `src/chorus/core.py`, change `ChorusResult` and the return block:

```python
@dataclass(frozen=True)
class ChorusResult:
    center: np.ndarray
    only: np.ndarray
    surround: np.ndarray
    contributions: dict[str, np.ndarray]
    metadata: dict[str, object]
```

Use this return block in `ChorusProcessor.process()` after inverse reconstruction:

```python
        contributions = {
            "Lc": center[:, 0],
            "Rc": center[:, 1],
            "Lo": only[:, 0],
            "Ro": only[:, 1],
            "Ls": surround[:, 0],
            "Rs": surround[:, 1],
        }

        return ChorusResult(
            center=center,
            only=only,
            surround=surround,
            contributions=contributions,
            metadata={
                "sample_rate": self.config.sample_rate,
                "transform": representation.metadata,
                "smoothing_alpha": self.config.smoothing_alpha,
                "epsilon": self.config.epsilon,
            },
        )
```

- [x] **Step 4: Run focused tests**

Run:

```bash
poetry run pytest tests/test_core.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/chorus/core.py tests/test_core.py
git commit -m "feat: expose chorus contribution signals"
```

## Task 3: Implement Invertible FrFT And Wavelet Adapters

**Files:**
- Modify: `src/chorus/transforms.py`
- Modify: `tests/test_transforms.py`

- [ ] **Step 1: Write failing transform round-trip tests**

Append to `tests/test_transforms.py`:

```python
def test_frft_round_trip_pass_through_stereo(stereo_identical: np.ndarray) -> None:
    transform = FrFTTransform(order=0.75)

    representation = transform.forward(stereo_identical)
    reconstructed = transform.inverse(representation)

    assert reconstructed.shape == stereo_identical.shape
    np.testing.assert_allclose(reconstructed, stereo_identical, atol=1e-10, rtol=1e-10)


def test_wavelet_round_trip_pass_through_stereo(stereo_identical: np.ndarray) -> None:
    transform = WaveletTransform(wavelet="db4", level=3)

    representation = transform.forward(stereo_identical)
    reconstructed = transform.inverse(representation)

    assert reconstructed.shape == stereo_identical.shape
    np.testing.assert_allclose(reconstructed, stereo_identical, atol=1e-8, rtol=1e-8)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
poetry run pytest tests/test_transforms.py::test_frft_round_trip_pass_through_stereo tests/test_transforms.py::test_wavelet_round_trip_pass_through_stereo -q
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Add inverse and component inverse support**

In `src/chorus/transforms.py`, add imports:

```python
from scipy.fft import fft, ifft
```

Replace `FrFTTransform.inverse()` with:

```python
    def inverse(self, representation: TransformRepresentation) -> np.ndarray:
        phase = np.exp(-0.5j * np.pi * self.order)
        channels = ifft(representation.data / phase, axis=-1).real
        return channels.T[: representation.original_shape[0], :]

    def inverse_components(
        self, components: np.ndarray, original_shape: tuple[int, int]
    ) -> np.ndarray:
        representation = TransformRepresentation(
            data=components,
            original_shape=original_shape,
            metadata={"transform": "frft", "order": self.order, "experimental": True},
        )
        return self.inverse(representation)
```

Replace `WaveletTransform.inverse()` with:

```python
    def inverse(self, representation: TransformRepresentation) -> np.ndarray:
        coeff_slices = representation.metadata["coeff_slices"]
        channels = []
        for index in range(2):
            coeffs = pywt.array_to_coeffs(
                representation.data[index],
                coeff_slices[index],
                output_format="wavedec",
            )
            channel = pywt.waverec(coeffs, self.wavelet, mode="periodization")
            channels.append(channel[: representation.original_shape[0]])
        return np.column_stack(channels)

    def inverse_components(
        self, components: np.ndarray, original_shape: tuple[int, int]
    ) -> np.ndarray:
        original = np.zeros(original_shape, dtype=np.float64)
        representation = self.forward(original)
        representation = TransformRepresentation(
            data=components,
            original_shape=original_shape,
            metadata=representation.metadata,
        )
        return self.inverse(representation)
```

- [ ] **Step 4: Run focused transform tests**

Run:

```bash
poetry run pytest tests/test_transforms.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chorus/transforms.py tests/test_transforms.py
git commit -m "feat: add invertible research transforms"
```

## Task 4: Select Transform Implementations In The Core

**Files:**
- Modify: `src/chorus/core.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Write selectable transform tests**

Append to `tests/test_core.py`:

```python
@pytest.mark.parametrize("transform_name", ["frft", "wavelet"])
def test_processor_can_run_reconstruction_capable_research_transforms(
    transform_name: str, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    processor = ChorusProcessor(
        ChorusConfig(sample_rate=sample_rate, transform=transform_name, smoothing_alpha=0.0)
    )

    result = processor.process(stereo_identical)

    assert result.metadata["transform"]["transform"] == transform_name
    reconstructed = result.center + result.only + result.surround
    np.testing.assert_allclose(reconstructed, stereo_identical, atol=1e-6, rtol=1e-6)
```

Add `import pytest` near the top of `tests/test_core.py`.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
poetry run pytest tests/test_core.py::test_processor_can_run_reconstruction_capable_research_transforms -q
```

Expected: FAIL with `NotImplementedError` from the core constructor.

- [ ] **Step 3: Select transform classes in the core**

In `src/chorus/core.py`, update imports:

```python
from chorus.transforms import FrFTTransform, STFTConfig, STFTTransform, WaveletTransform
```

Replace the constructor with:

```python
    def __init__(self, config: ChorusConfig) -> None:
        self.config = config
        if config.transform == "stft":
            self.transform = STFTTransform(STFTConfig(config.frame_size, config.hop_size))
        elif config.transform == "frft":
            self.transform = FrFTTransform()
        elif config.transform == "wavelet":
            self.transform = WaveletTransform()
        else:
            raise ValueError(f"unsupported transform {config.transform!r}")
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
poetry run pytest tests/test_core.py tests/test_transforms.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chorus/core.py tests/test_core.py
git commit -m "feat: enable selectable chorus transforms"
```

## Task 5: Generate Spectrogram Artifacts

**Files:**
- Create: `src/chorus/reporting.py`
- Create: `tests/test_reporting.py`

- [ ] **Step 1: Write failing spectrogram tests**

Create `tests/test_reporting.py`:

```python
from __future__ import annotations

import numpy as np

from chorus.core import ChorusConfig, ChorusProcessor
from chorus.reporting import render_spectrograms


def test_render_spectrograms_writes_deterministic_artifacts(
    tmp_path, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    result = ChorusProcessor(ChorusConfig(sample_rate=sample_rate)).process(stereo_identical)

    artifacts = render_spectrograms(tmp_path, sample_rate, stereo_identical, result)

    assert set(artifacts) == {"input_left", "input_right", "Lc", "Rc", "Lo", "Ro", "Ls", "Rs"}
    for name, path in artifacts.items():
        assert path == tmp_path / "spectrograms" / f"{name}.png"
        assert path.exists()
        assert path.stat().st_size > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
poetry run pytest tests/test_reporting.py::test_render_spectrograms_writes_deterministic_artifacts -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'chorus.reporting'`.

- [ ] **Step 3: Implement deterministic spectrogram rendering**

Create `src/chorus/reporting.py`:

```python
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from chorus.core import ChorusResult


def _save_spectrogram(path: Path, sample_rate: int, signal: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    ax.specgram(signal, NFFT=512, Fs=sample_rate, noverlap=256, cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_spectrograms(
    output_dir: str | Path,
    sample_rate: int,
    input_audio: np.ndarray,
    result: ChorusResult,
) -> dict[str, Path]:
    spectrogram_dir = Path(output_dir) / "spectrograms"
    spectrogram_dir.mkdir(parents=True, exist_ok=True)
    signals = {
        "input_left": input_audio[:, 0],
        "input_right": input_audio[:, 1],
        **result.contributions,
    }
    artifacts: dict[str, Path] = {}
    for name, signal in signals.items():
        path = spectrogram_dir / f"{name}.png"
        _save_spectrogram(path, sample_rate, signal)
        artifacts[name] = path
    return artifacts
```

- [ ] **Step 4: Run the test**

Run:

```bash
poetry run pytest tests/test_reporting.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chorus/reporting.py tests/test_reporting.py
git commit -m "feat: render chorus spectrogram artifacts"
```

## Task 6: Add Transform Analyzer Metadata

**Files:**
- Modify: `src/chorus/reporting.py`
- Modify: `tests/test_reporting.py`

- [ ] **Step 1: Write failing analyzer metadata tests**

Append to `tests/test_reporting.py`:

```python
from chorus.reporting import analyze_transforms


def test_analyze_transforms_records_reconstruction_metrics(
    stereo_identical: np.ndarray, sample_rate: int
) -> None:
    analyzers = analyze_transforms(sample_rate, stereo_identical)

    assert set(analyzers) == {"stft", "frft", "wavelet"}
    assert analyzers["stft"]["status"] == "ok"
    assert analyzers["stft"]["experimental"] is False
    assert analyzers["frft"]["status"] == "ok"
    assert analyzers["frft"]["experimental"] is True
    assert analyzers["wavelet"]["status"] == "ok"
    assert analyzers["wavelet"]["experimental"] is True
    for analyzer in analyzers.values():
        assert analyzer["reconstruction"]["max_abs"] < 1e-6
        assert analyzer["reconstruction"]["rms"] < 1e-6
        assert analyzer["duration_seconds"] >= 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
poetry run pytest tests/test_reporting.py::test_analyze_transforms_records_reconstruction_metrics -q
```

Expected: FAIL with `ImportError` because `analyze_transforms` does not exist.

- [ ] **Step 3: Implement analyzer metadata**

Append to `src/chorus/reporting.py`:

```python
import time

from chorus.transforms import FrFTTransform, STFTConfig, STFTTransform, WaveletTransform


def _residual_summary(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    residual = reference - candidate
    return {
        "max_abs": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0,
    }


def analyze_transforms(sample_rate: int, input_audio: np.ndarray) -> dict[str, dict[str, object]]:
    del sample_rate
    transforms = {
        "stft": STFTTransform(STFTConfig()),
        "frft": FrFTTransform(),
        "wavelet": WaveletTransform(),
    }
    analysis: dict[str, dict[str, object]] = {}
    for name, transform in transforms.items():
        started = time.perf_counter()
        try:
            representation = transform.forward(input_audio)
            reconstructed = transform.inverse(representation)
            analysis[name] = {
                "status": "ok",
                "metadata": {
                    key: value
                    for key, value in representation.metadata.items()
                    if key != "coeff_slices"
                },
                "experimental": bool(representation.metadata.get("experimental", False)),
                "shape": list(representation.data.shape),
                "energy": float(np.sum(np.square(np.abs(representation.data)))),
                "reconstruction": _residual_summary(input_audio, reconstructed),
                "duration_seconds": time.perf_counter() - started,
                "warnings": [] if name == "stft" else [f"{name} is reconstruction-capable but experimental in v1"],
            }
        except Exception as exc:
            analysis[name] = {
                "status": "failed",
                "error": str(exc),
                "experimental": name != "stft",
                "duration_seconds": time.perf_counter() - started,
                "warnings": [f"{name} analyzer failed"],
            }
    return analysis
```

- [ ] **Step 4: Run focused reporting tests**

Run:

```bash
poetry run pytest tests/test_reporting.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chorus/reporting.py tests/test_reporting.py
git commit -m "feat: add transform analyzer metadata"
```

## Task 7: Write JSON And Markdown Reports

**Files:**
- Modify: `src/chorus/reporting.py`
- Modify: `src/chorus/io.py`
- Modify: `tests/test_reporting.py`

- [ ] **Step 1: Write failing report assembly test**

Append to `tests/test_reporting.py`:

```python
from chorus.reporting import build_v1_report, write_markdown_report


def test_v1_report_includes_spectrograms_analyzers_and_markdown(
    tmp_path, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    result = ChorusProcessor(ChorusConfig(sample_rate=sample_rate)).process(stereo_identical)
    stem_paths = {
        "center": tmp_path / "center.wav",
        "only": tmp_path / "only.wav",
        "surround": tmp_path / "surround.wav",
    }
    report = build_v1_report(tmp_path / "input.wav", tmp_path, sample_rate, stereo_identical, result, stem_paths)
    markdown_path = write_markdown_report(tmp_path, report)

    assert "spectrograms" in report
    assert "transform_analysis" in report
    assert report["warnings"]
    assert markdown_path == tmp_path / "report.md"
    text = markdown_path.read_text(encoding="utf-8")
    assert "## Transform Comparison" in text
    assert "spectrograms/Lc.png" in text
    assert "FrFT" in text
    assert "Wavelet" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
poetry run pytest tests/test_reporting.py::test_v1_report_includes_spectrograms_analyzers_and_markdown -q
```

Expected: FAIL with missing imports.

- [ ] **Step 3: Move report assembly to `reporting.py`**

Add to `src/chorus/reporting.py`:

```python
def level_metrics(audio: np.ndarray) -> dict[str, float]:
    if not audio.size:
        return {"rms": 0.0, "peak": 0.0}
    return {
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))),
    }


def build_v1_report(
    input_path: str | Path,
    output_dir: str | Path,
    sample_rate: int,
    input_audio: np.ndarray,
    result: ChorusResult,
    stem_paths: dict[str, Path],
) -> dict[str, object]:
    spectrograms = render_spectrograms(output_dir, sample_rate, input_audio, result)
    transform_analysis = analyze_transforms(sample_rate, input_audio)
    reconstructed = result.center + result.only + result.surround
    return {
        "input": {
            "path": str(input_path),
            "sample_rate": sample_rate,
            "duration_seconds": float(input_audio.shape[0] / sample_rate),
            "channels": 2,
        },
        "output": {
            "directory": str(output_dir),
            "stems": {name: str(path) for name, path in stem_paths.items()},
        },
        "settings": result.metadata,
        "levels": {
            "center": level_metrics(result.center),
            "only": level_metrics(result.only),
            "surround": level_metrics(result.surround),
            "contributions": {name: level_metrics(audio) for name, audio in result.contributions.items()},
        },
        "checks": {
            "reconstruction": {
                "passed": bool(_residual_summary(input_audio, reconstructed)["max_abs"] <= 1e-6),
                **_residual_summary(input_audio, reconstructed),
            },
        },
        "spectrograms": {name: str(path) for name, path in spectrograms.items()},
        "transform_analysis": transform_analysis,
        "warnings": [
            "STFT remains the default reference path in v1",
            "FrFT and Wavelet are reconstruction-capable but experimental in v1",
        ],
    }


def write_markdown_report(output_dir: str | Path, report: dict[str, object]) -> Path:
    path = Path(output_dir) / "report.md"
    spectrograms = report["spectrograms"]
    analysis = report["transform_analysis"]
    lines = [
        "# CHORUS Split Report",
        "",
        "## Input",
        f"- Path: `{report['input']['path']}`",
        f"- Sample rate: `{report['input']['sample_rate']}`",
        f"- Duration seconds: `{report['input']['duration_seconds']:.6f}`",
        "",
        "## Output Stems",
    ]
    for name, stem_path in report["output"]["stems"].items():
        lines.append(f"- {name}: `{stem_path}`")
    lines.extend(["", "## Spectrograms"])
    for name, image_path in spectrograms.items():
        lines.append(f"- {name}: ![{name}]({Path(image_path).as_posix()})")
    lines.extend(["", "## Transform Comparison", "| Transform | Status | Experimental | Max abs residual | RMS residual |", "| --- | --- | --- | ---: | ---: |"])
    for name, item in analysis.items():
        residual = item.get("reconstruction", {"max_abs": 0.0, "rms": 0.0})
        lines.append(
            f"| {name.upper()} | {item['status']} | {item['experimental']} | {residual['max_abs']:.6e} | {residual['rms']:.6e} |"
        )
    lines.extend(["", "## Warnings"])
    for warning in report["warnings"]:
        lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
```

Replace `build_report()` in `src/chorus/io.py` with a wrapper:

```python
from chorus.reporting import build_v1_report


def build_report(
    input_path: str | Path,
    output_dir: str | Path,
    sample_rate: int,
    input_audio: np.ndarray,
    result: ChorusResult,
    stem_paths: dict[str, Path],
) -> dict[str, object]:
    return build_v1_report(input_path, output_dir, sample_rate, input_audio, result, stem_paths)
```

- [ ] **Step 4: Run reporting tests**

Run:

```bash
poetry run pytest tests/test_reporting.py tests/test_io_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chorus/reporting.py src/chorus/io.py tests/test_reporting.py
git commit -m "feat: write chorus v1 reports"
```

## Task 8: Integrate Markdown Report In CLI

**Files:**
- Modify: `src/chorus/cli.py`
- Modify: `tests/test_io_cli.py`

- [ ] **Step 1: Extend the CLI integration test**

In `tests/test_io_cli.py`, extend `test_cli_split_writes_expected_outputs`:

```python
    assert (out_dir / "report.md").exists()
    assert (out_dir / "spectrograms" / "input_left.png").exists()
    assert (out_dir / "spectrograms" / "input_right.png").exists()
    assert (out_dir / "spectrograms" / "Lc.png").exists()
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert "transform_analysis" in report
    assert "spectrograms" in report
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
poetry run pytest tests/test_io_cli.py::test_cli_split_writes_expected_outputs -q
```

Expected: FAIL because `report.md` is not written.

- [ ] **Step 3: Write the Markdown report from the CLI**

In `src/chorus/cli.py`, add:

```python
from chorus.reporting import write_markdown_report
```

After writing `report.json`, add:

```python
        write_markdown_report(args.out_dir, report)
```

- [ ] **Step 4: Run the full suite**

Run:

```bash
poetry run pytest -q
poetry run ruff check .
```

Expected: all tests pass and Ruff reports no violations.

- [ ] **Step 5: Commit**

```bash
git add src/chorus/cli.py tests/test_io_cli.py
git commit -m "feat: write markdown split reports"
```

## Self-Review

- Spec coverage: The plan preserves v0 STFT behavior, renders all required spectrograms, writes `report.md` and `report.json`, records transform comparison metadata, keeps the existing `checks.reconstruction.passed` report contract, and tests FrFT/Wavelet reconstruction tolerance.
- Spec tension resolved: The edited v1 goals require FrFT/Wavelet reconstruction support, while older text still calls them research analyzers. The plan makes them selectable and tolerance-gated but still clearly labeled experimental in reports.
- Placeholder scan: No `TBD`, `TODO`, or unspecified edge handling remains.
- Type consistency: `ChorusResult.contributions`, `render_spectrograms()`, `analyze_transforms()`, `build_v1_report()`, and `write_markdown_report()` are defined before use.
