from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the worktree's src directory takes precedence over any installed package.
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np

from chorus.core import ChorusConfig, ChorusProcessor
from chorus.filters import normalize_filter_config


def _sine(sample_rate: int, samples: int, frequency: float, amplitude: float) -> np.ndarray:
    time = np.arange(samples, dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency * time)


def _cases(
    sample_rate: int, samples: int
) -> dict[str, tuple[np.ndarray, dict | None, ChorusConfig | None]]:
    tone = _sine(sample_rate, samples, 440.0, 0.25)
    quiet = _sine(sample_rate, samples, 330.0, 1e-8)
    return {
        "unity_bypass": (np.column_stack([tone, tone]), None, None),
        "center_dominant": (np.column_stack([tone, tone * 0.95]), None, None),
        "hard_panned_left": (np.column_stack([tone, np.zeros_like(tone)]), None, None),
        "hard_panned_right": (np.column_stack([np.zeros_like(tone), tone]), None, None),
        "phase_inverted_surround": (np.column_stack([tone, -tone]), None, None),
        "known_eq_preset": (
            np.column_stack([tone, tone]),
            {
                "Lc": [{"type": "gain", "db": -6.0}],
                "Lo": [{"type": "eq", "mode": "highpass", "frequency_hz": 120.0, "q": 0.707}],
                "Rs": [{"type": "polarity"}],
            },
            None,
        ),
        "silence": (np.zeros((samples, 2), dtype=np.float64), None, None),
        "near_silence": (np.column_stack([quiet, -quiet]), None, None),
        "frft_unity_bypass": (
            np.column_stack([tone, tone]),
            None,
            ChorusConfig(
                sample_rate=sample_rate,
                transform="frft",
                frft_order=0.5,
                frame_size=1024,
                smoothing_alpha=0.0,
                filter_chains=normalize_filter_config(None),
            ),
        ),
    }


def build_fixture_set(sample_rate: int = 48_000, samples: int = 4096) -> dict[str, object]:
    arrays: dict[str, np.ndarray] = {}
    manifest = {
        "sample_rate": sample_rate,
        "samples": samples,
        "tolerances": {
            "bypass_max_abs": 1e-6,
            "filtered_max_abs": 5e-5,
        },
        "cases": {},
    }
    for name, (audio, filters, override_config) in _cases(sample_rate, samples).items():
        if override_config is not None:
            config = override_config
        else:
            config = ChorusConfig(
                sample_rate=sample_rate,
                smoothing_alpha=0.0,
                filter_chains=normalize_filter_config(filters),
            )
        processor = ChorusProcessor(config)
        result = processor.process(audio)
        arrays[f"{name}.input"] = audio
        arrays[f"{name}.center"] = result.center
        arrays[f"{name}.only"] = result.only
        arrays[f"{name}.surround"] = result.surround
        manifest["cases"][name] = {
            "filters": filters or {},
            "outputs": {
                "input": f"{name}.input.npy",
                "center": f"{name}.center.npy",
                "only": f"{name}.only.npy",
                "surround": f"{name}.surround.npy",
            },
        }
    return {"manifest": manifest, "arrays": arrays}


def write_fixture_set(output_dir: Path) -> None:
    fixture_set = build_fixture_set()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(fixture_set["manifest"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for name, array in fixture_set["arrays"].items():
        np.save(output_dir / f"{name}.npy", array)


if __name__ == "__main__":
    write_fixture_set(Path("fixtures/v2"))
