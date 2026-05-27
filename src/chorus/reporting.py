from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from chorus.core import ChorusResult
from chorus.transforms import FrFTTransform, STFTConfig, STFTTransform, WaveletTransform


def _save_spectrogram(path: Path, sample_rate: int, signal: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in log10",
            category=RuntimeWarning,
        )
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
                "warnings": []
                if name == "stft"
                else [f"{name} is reconstruction-capable but experimental in v1"],
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
    residual = _residual_summary(input_audio, reconstructed)
    filter_report = result.metadata.get("filters", {"transparent": True})
    filter_transparent = bool(filter_report.get("transparent", True))
    report_warnings = [
        "STFT remains the default reference path in v1",
        "FrFT and Wavelet are reconstruction-capable but experimental in v1",
    ]
    if not filter_transparent:
        report_warnings.append("Contribution filters intentionally altered the output")
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
        "settings": {
            k: ({ik: iv for ik, iv in v.items() if ik != "coeff_slices"} if k == "transform" and isinstance(v, dict) else v)
            for k, v in result.metadata.items()
            if k != "coeff_slices"
        },
        "filters": filter_report,
        "levels": {
            "center": level_metrics(result.center),
            "only": level_metrics(result.only),
            "surround": level_metrics(result.surround),
            "contributions": {
                name: level_metrics(audio) for name, audio in result.contributions.items()
            },
        },
        "checks": {
            "reconstruction": {
                "passed": bool(residual["max_abs"] <= 1e-6 and filter_transparent),
                **residual,
            },
            "filtered_output": {
                "transparent": filter_transparent,
                "intentionally_altered": not filter_transparent,
            },
        },
        "spectrograms": {name: str(path) for name, path in spectrograms.items()},
        "transform_analysis": transform_analysis,
        "warnings": report_warnings,
    }


def _transform_label(name: str) -> str:
    return {"stft": "STFT", "frft": "FrFT", "wavelet": "Wavelet"}.get(name, name)


def write_markdown_report(output_dir: str | Path, report: dict[str, object]) -> Path:
    path = Path(output_dir) / "report.md"
    spectrograms = report["spectrograms"]
    analysis = report["transform_analysis"]
    filters = report.get("filters", {"transparent": True})
    filter_transparent = bool(filters.get("transparent", True))
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
        rel = Path(image_path).relative_to(Path(output_dir))
        lines.append(f"- {name}: ![{name}]({rel.as_posix()})")
    lines.extend(
        [
            "",
            "## Transform Comparison",
            "| Transform | Status | Experimental | Max abs residual | RMS residual |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for name, item in analysis.items():
        residual = item.get("reconstruction", {"max_abs": 0.0, "rms": 0.0})
        lines.append(
            f"| {_transform_label(name)} | {item['status']} | {item['experimental']} | "
            f"{residual['max_abs']:.6e} | {residual['rms']:.6e} |"
        )
    lines.extend(
        [
            "",
            "## Contribution Filters",
            f"- Transparent: `{filter_transparent}`",
        ]
    )
    if not filter_transparent:
        lines.append("- Filtered output intentionally altered: `True`")
        for name, chain in filters.get("chains", {}).items():
            chain_labels = []
            for item in chain:
                parameters = item.get("parameters", {})
                parameter_text = (
                    ""
                    if not parameters
                    else " "
                    + ", ".join(f"{key}={value}" for key, value in parameters.items())
                )
                chain_labels.append(f"{item['type']}{parameter_text}")
            lines.append(f"- {name}: {' -> '.join(chain_labels)}")
    lines.extend(["", "## Warnings"])
    for warning in report["warnings"]:
        lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
