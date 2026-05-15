from __future__ import annotations

import json

import numpy as np
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
