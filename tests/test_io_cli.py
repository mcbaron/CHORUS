from __future__ import annotations

import json

import numpy as np
import pytest
from chorus.cli import main
from chorus.core import ChorusConfig, ChorusProcessor
from chorus.io import build_report, read_stereo_wav, write_stems
from scipy.io import wavfile


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
