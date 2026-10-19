from __future__ import annotations

import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from chorus.core import ChorusResult


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



# Mixxx needs exactly 4 stems, so "only" splits into its left and right sides.
# Each entry: name, color, (left contribution, right contribution). None is silence.
STEM_FILE_LAYOUT = (
    ("Center", "#3987E5", ("Lc", "Rc")),
    ("Left only", "#D95926", ("Lo", None)),
    ("Right only", "#C98500", (None, "Ro")),
    ("Surround", "#199E70", ("Ls", "Rs")),
)


def write_stem_mp4(
    path: str | Path, sample_rate: int, input_audio: np.ndarray, result: ChorusResult
) -> Path:
    """Write a Native Instruments stem file (the mix, then 4 stems) for Mixxx 2.6+."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("writing a .stem.mp4 file needs ffmpeg on PATH")
    path = Path(path)
    silence = np.zeros(input_audio.shape[0])
    streams = [input_audio] + [
        np.column_stack([result.contributions[c] if c else silence for c in pair])
        for _, _, pair in STEM_FILE_LAYOUT
    ]
    with tempfile.TemporaryDirectory() as tmp:
        args = ["ffmpeg", "-v", "error", "-y"]
        for index, audio in enumerate(streams):
            wav = Path(tmp) / f"{index}.wav"
            wavfile.write(wav, sample_rate, np.asarray(audio, dtype=np.float32))
            args += ["-i", str(wav)]
        for index in range(len(streams)):
            args += ["-map", f"{index}:a"]
        # ALAC is lossless, so the 4 stems that Mixxx sums give back the input exactly.
        args += ["-c:a", "alac", "-sample_fmt", "s32p", "-f", "mp4", str(path)]
        subprocess.run(args, check=True)
    manifest = {
        "version": 1,
        "mastering_dsp": {"compressor": {"enabled": False}, "limiter": {"enabled": False}},
        "stems": [{"name": name, "color": color} for name, color, _ in STEM_FILE_LAYOUT],
    }
    _add_stem_atom(path, json.dumps(manifest).encode())
    return path


def _boxes(data: bytearray, start: int, end: int):
    # ponytail: 32-bit box sizes only. Fine for audio; add 64-bit sizes for files over 4 GB.
    while start < end:
        size, kind = struct.unpack_from(">I4s", data, start)
        if size < 8:
            raise ValueError(f"unsupported MP4 box size {size} at byte {start}")
        yield kind, start, start + size
        start += size


def _add_stem_atom(path: Path, manifest: bytes) -> None:
    """Put the manifest at moov/udta/stem. Mixxx reads only the first udta in moov."""
    data = bytearray(path.read_bytes())
    moov = next(box for box in _boxes(data, 0, len(data)) if box[0] == b"moov")
    if moov[2] != len(data):
        # Growing moov is safe only when nothing follows it (chunk offsets point into mdat).
        raise ValueError("moov must be the last box in the file")
    # The ffmpeg mp4 muxer always writes a udta box, even with -fflags +bitexact.
    udta = next(box for box in _boxes(data, moov[1] + 8, moov[2]) if box[0] == b"udta")
    stem = struct.pack(">I4s", 8 + len(manifest), b"stem") + manifest
    data[udta[2]:udta[2]] = stem
    for _, start, end in (moov, udta):
        struct.pack_into(">I", data, start, end - start + len(stem))
    path.write_bytes(data)
