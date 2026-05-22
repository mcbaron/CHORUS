from __future__ import annotations

import numpy as np

from chorus.core import ChorusConfig, ChorusProcessor
from chorus.filters import normalize_filter_config
from chorus.reporting import (
    analyze_transforms,
    build_v1_report,
    render_spectrograms,
    write_markdown_report,
)


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


def test_v1_report_includes_spectrograms_analyzers_and_markdown(
    tmp_path, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    result = ChorusProcessor(ChorusConfig(sample_rate=sample_rate)).process(stereo_identical)
    stem_paths = {
        "center": tmp_path / "center.wav",
        "only": tmp_path / "only.wav",
        "surround": tmp_path / "surround.wav",
    }
    report = build_v1_report(
        tmp_path / "input.wav",
        tmp_path,
        sample_rate,
        stereo_identical,
        result,
        stem_paths,
    )
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


def test_report_records_filter_chains_and_filtered_output_status(
    tmp_path, stereo_identical: np.ndarray, sample_rate: int
) -> None:
    result = ChorusProcessor(
        ChorusConfig(
            sample_rate=sample_rate,
            smoothing_alpha=0.0,
            filter_chains=normalize_filter_config({"Lc": [{"type": "gain", "db": -6.0}]}),
        )
    ).process(stereo_identical)
    stem_paths = {
        "center": tmp_path / "center.wav",
        "only": tmp_path / "only.wav",
        "surround": tmp_path / "surround.wav",
    }

    report = build_v1_report(
        tmp_path / "input.wav",
        tmp_path,
        sample_rate,
        stereo_identical,
        result,
        stem_paths,
    )

    assert report["filters"]["transparent"] is False
    assert report["filters"]["chains"]["Lc"][0]["type"] == "gain"
    assert "pre_levels" in report["filters"]
    assert "post_levels" in report["filters"]
    assert report["checks"]["filtered_output"]["transparent"] is False
    assert report["checks"]["filtered_output"]["intentionally_altered"] is True
    assert report["checks"]["reconstruction"]["passed"] is False

    markdown_path = write_markdown_report(tmp_path, report)
    text = markdown_path.read_text(encoding="utf-8")
    assert "## Contribution Filters" in text
    assert "- Transparent: `False`" in text
    assert "- Filtered output intentionally altered: `True`" in text
    assert "- Lc: gain db=-6.0" in text
