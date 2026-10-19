from __future__ import annotations

import json
import shutil
import struct
import subprocess

import numpy as np
import pytest
from scipy.io import wavfile

from chorus.cli import main
from chorus.core import ChorusConfig, ChorusProcessor
from chorus.io import read_stereo_wav, write_stems
from chorus.reporting import build_report


def test_wav_read_write_and_report(
    tmp_path, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    input_path = tmp_path / "input.wav"
    out_dir = tmp_path / "out"
    wavfile.write(input_path, sample_rate, stereo_identical.astype(np.float32))

    loaded_rate, loaded = read_stereo_wav(input_path)
    processor = ChorusProcessor(
        ChorusConfig(sample_rate=loaded_rate, smoothing_alpha=0.0)
    )
    result = processor.process(loaded)

    stem_paths = write_stems(out_dir, sample_rate, result)
    report = build_report(
        input_path=input_path,
        output_dir=out_dir,
        sample_rate=sample_rate,
        input_audio=loaded,
        result=result,
        stem_paths=stem_paths,
    )

    assert set(stem_paths) == {"center", "only", "surround"}
    assert (out_dir / "center.wav").exists()
    assert (out_dir / "only.wav").exists()
    assert (out_dir / "surround.wav").exists()
    assert report["checks"]["reconstruction"]["passed"] is True
    assert report["levels"]["center"]["rms"] > report["levels"]["only"]["rms"]

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    loaded_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded_report["input"]["sample_rate"] == sample_rate


def test_read_stereo_wav_normalizes_signed_integer_pcm(tmp_path, sample_rate: int) -> None:
    input_path = tmp_path / "signed.wav"
    samples = np.array(
        [
            [np.iinfo(np.int16).min, np.iinfo(np.int16).max],
            [0, -1],
        ],
        dtype=np.int16,
    )
    wavfile.write(input_path, sample_rate, samples)

    loaded_rate, loaded = read_stereo_wav(input_path)

    assert loaded_rate == sample_rate
    assert loaded.dtype == np.float64
    np.testing.assert_allclose(
        loaded,
        np.array(
            [
                [-1.0, np.iinfo(np.int16).max / 32768.0],
                [0.0, -1.0 / 32768.0],
            ]
        ),
    )
    assert np.max(np.abs(loaded)) <= 1.0


def test_read_stereo_wav_normalizes_unsigned_integer_pcm(tmp_path, sample_rate: int) -> None:
    input_path = tmp_path / "unsigned.wav"
    samples = np.array(
        [
            [0, 128],
            [255, 128],
        ],
        dtype=np.uint8,
    )
    wavfile.write(input_path, sample_rate, samples)

    loaded_rate, loaded = read_stereo_wav(input_path)

    assert loaded_rate == sample_rate
    assert loaded.dtype == np.float64
    np.testing.assert_allclose(
        loaded,
        np.array(
            [
                [-1.0, 0.0],
                [127.0 / 128.0, 0.0],
            ]
        ),
    )
    assert np.max(np.abs(loaded)) <= 1.0


def test_read_stereo_wav_rejects_non_stereo_shapes(tmp_path, sample_rate: int) -> None:
    input_path = tmp_path / "mono.wav"
    wavfile.write(input_path, sample_rate, np.zeros(8, dtype=np.float32))

    with pytest.raises(ValueError, match="expected stereo WAV"):
        read_stereo_wav(input_path)


def test_write_stems_writes_float32_wav(
    tmp_path, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    processor = ChorusProcessor(ChorusConfig(sample_rate=sample_rate, smoothing_alpha=0.0))
    result = processor.process(stereo_identical)
    stem_paths = write_stems(tmp_path, sample_rate, result)

    _loaded_rate, center = wavfile.read(stem_paths["center"])

    assert center.dtype == np.float32


def test_cli_split_writes_expected_outputs(
    tmp_path, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    input_path = tmp_path / "input.wav"
    out_dir = tmp_path / "out"
    wavfile.write(input_path, sample_rate, stereo_identical.astype(np.float32))

    exit_code = main(["split", str(input_path), "--out-dir", str(out_dir), "--transform", "stft"])

    assert exit_code == 0
    assert (out_dir / "center.wav").exists()
    assert (out_dir / "only.wav").exists()
    assert (out_dir / "surround.wav").exists()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "spectrograms" / "input_left.png").exists()
    assert (out_dir / "spectrograms" / "input_right.png").exists()
    assert (out_dir / "spectrograms" / "Lc.png").exists()
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert "transform_analysis" in report
    assert "spectrograms" in report


def test_cli_split_accepts_filter_config(
    tmp_path, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    input_path = tmp_path / "input.wav"
    out_dir = tmp_path / "out"
    filters_path = tmp_path / "filters.json"
    wavfile.write(input_path, sample_rate, stereo_identical.astype(np.float32))
    filters_path.write_text(
        json.dumps({"Lc": [{"type": "gain", "db": -6.0}]}), encoding="utf-8"
    )

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
    filters_path.write_text(
        json.dumps({"bad": [{"type": "gain", "db": 1.0}]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unknown contribution"):
        main(["split", str(input_path), "--out-dir", str(out_dir), "--filters", str(filters_path)])

    assert not out_dir.exists()


def _mp4_child(data: bytes, start: int, end: int, kind: bytes) -> tuple[int, int]:
    # First child box of this kind, as Mixxx searches: returns (payload start, box end).
    while start < end:
        size, box = struct.unpack_from(">I4s", data, start)
        if box == kind:
            return start + 8, start + size
        start += size
    raise AssertionError(f"no {kind!r} box")


def _decode(path, stream: int) -> np.ndarray:
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:a:{stream}", "-f", "f32le", "-"]
    )
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_stem_mp4_loads_like_mixxx(tmp_path, sample_rate: int) -> None:
    rng = np.random.default_rng(3)
    shared = 0.2 * rng.standard_normal(sample_rate)
    left_only = 0.1 * rng.standard_normal(sample_rate)
    audio = np.column_stack([shared + left_only, shared])
    input_path = tmp_path / "song.wav"
    wavfile.write(input_path, sample_rate, audio.astype(np.float32))

    assert main(["split", str(input_path), "--out-dir", str(tmp_path / "out"), "--stem-mp4"]) == 0
    stem_file = tmp_path / "out" / "song.stem.mp4"

    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(stem_file)]
    ))["streams"]
    assert [s["codec_type"] for s in probe] == ["audio"] * 5  # the mix, then 4 stems
    assert {(s["codec_name"], s["sample_rate"], s["channels"]) for s in probe} == {
        ("alac", str(sample_rate), 2)
    }

    data = stem_file.read_bytes()
    moov = _mp4_child(data, 0, len(data), b"moov")
    udta = _mp4_child(data, *moov, b"udta")
    manifest = json.loads(data[slice(*_mp4_child(data, *udta, b"stem"))])
    assert manifest["version"] == 1
    names = [s["name"] for s in manifest["stems"]]
    assert names == ["Center", "Left only", "Right only", "Surround"]

    # Mixxx plays the sum of the 4 stems, not the mix. ALAC makes that sum exact to 24 bits.
    stems = sum(_decode(stem_file, i) for i in range(1, 5))
    assert stems.shape == audio.shape
    error_db = 10 * np.log10(np.mean((stems - audio) ** 2) / np.mean(audio**2))
    assert error_db < -90, f"stems differ from the input by {error_db:.1f} dB"
    assert np.abs(_decode(stem_file, 3)[:, 0]).max() == 0  # "Right only" has a silent left side
