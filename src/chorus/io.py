from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from chorus.core import ChorusResult
from chorus.reporting import build_v1_report


def read_stereo_wav(path: str | Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    array = np.asarray(data)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"expected stereo WAV shaped (samples, 2), got {array.shape}")
    if np.issubdtype(array.dtype, np.unsignedinteger):
        info = np.iinfo(array.dtype)
        midpoint = float(info.max + 1) / 2.0
        array = (array.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(array.dtype, np.signedinteger):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        array = array.astype(np.float64) / scale
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


def build_report(
    input_path: str | Path,
    output_dir: str | Path,
    sample_rate: int,
    input_audio: np.ndarray,
    result: ChorusResult,
    stem_paths: dict[str, Path],
) -> dict[str, object]:
    return build_v1_report(input_path, output_dir, sample_rate, input_audio, result, stem_paths)
