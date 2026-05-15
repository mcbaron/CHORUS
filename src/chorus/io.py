from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from chorus.core import ChorusResult


def read_stereo_wav(path: str | Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    array = np.asarray(data)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"expected stereo WAV shaped (samples, 2), got {array.shape}")
    if np.issubdtype(array.dtype, np.integer):
        max_value = float(np.iinfo(array.dtype).max)
        array = array.astype(np.float64) / max_value
    else:
        array = array.astype(np.float64)
    return int(sample_rate), array


def write_stems(
    output_dir: str | Path,
    sample_rate: int,
    result: ChorusResult,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stems = {
        "center": out / "center.wav",
        "only": out / "only.wav",
        "surround": out / "surround.wav",
    }
    for name, path in stems.items():
        audio = getattr(result, name)
        wavfile.write(path, sample_rate, np.asarray(audio, dtype=np.float32))
    return stems


def _level_metrics(audio: np.ndarray) -> dict[str, float]:
    if not audio.size:
        return {"rms": 0.0, "peak": 0.0}
    return {
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))),
    }


def _residual_metrics(input_audio: np.ndarray, result: ChorusResult) -> dict[str, object]:
    reconstructed = result.center + result.only + result.surround
    residual = input_audio - reconstructed
    left = residual[:, 0]
    right = residual[:, 1]
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0
    return {
        "passed": bool(max_abs <= 1e-6),
        "max_abs": max_abs,
        "rms": rms,
        "left": {
            "max_abs": float(np.max(np.abs(left))) if left.size else 0.0,
            "rms": float(np.sqrt(np.mean(np.square(left)))) if left.size else 0.0,
        },
        "right": {
            "max_abs": float(np.max(np.abs(right))) if right.size else 0.0,
            "rms": float(np.sqrt(np.mean(np.square(right)))) if right.size else 0.0,
        },
    }


def build_report(
    input_path: str | Path,
    output_dir: str | Path,
    sample_rate: int,
    input_audio: np.ndarray,
    result: ChorusResult,
    stem_paths: dict[str, Path],
) -> dict[str, object]:
    duration_seconds = float(input_audio.shape[0] / sample_rate)
    return {
        "input": {
            "path": str(input_path),
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "channels": 2,
        },
        "output": {
            "directory": str(output_dir),
            "stems": {name: str(path) for name, path in stem_paths.items()},
        },
        "settings": result.metadata,
        "levels": {
            "center": _level_metrics(result.center),
            "only": _level_metrics(result.only),
            "surround": _level_metrics(result.surround),
        },
        "checks": {
            "reconstruction": _residual_metrics(input_audio, result),
        },
    }
