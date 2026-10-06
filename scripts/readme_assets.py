"""Build the README figures from one or more audio files (WAV or MP3).

Usage: poetry run python scripts/readme_assets.py TRACK [TRACK ...]

Each track gets one column in the spectrogram figure and one bar in the energy chart.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram

from chorus.core import ChorusConfig, ChorusProcessor

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
STEMS = ("center", "only", "surround")
EXCERPT_AT = 0.4  # hero excerpt starts at 40% of each track
EXCERPT_SECONDS = 8
THEMES = {
    "light": {
        "surface": "#ffffff",
        "ink": "#1f2328",
        "muted": "#59636e",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "ramp": ["#ffffff", "#cde2fb", "#5598e7", "#1c5cab", "#0d366b"],
    },
    "dark": {
        "surface": "#0d1117",
        "ink": "#f0f6fc",
        "muted": "#9198a1",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "ramp": ["#0d1117", "#104281", "#2a78d6", "#86b6ef", "#cde2fb"],
    },
}


def load(path: Path) -> tuple[int, np.ndarray]:
    # ffmpeg decodes WAV and MP3 alike.
    rate = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate", "-of", "csv=p=0", str(path)]
    )
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-"]
    )
    return int(rate), np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)


def split(sample_rate: int, audio: np.ndarray, transform: str = "stft"):
    return ChorusProcessor(ChorusConfig(sample_rate=sample_rate, transform=transform)).process(
        audio
    )


def power_db(sample_rate: int, stereo: np.ndarray):
    freqs, times, power = spectrogram(stereo.T, fs=sample_rate, nperseg=2048, noverlap=1536)
    return freqs, times, 10 * np.log10(power.sum(axis=0) + 1e-12)


def excerpt_spectra(sample_rate: int, audio: np.ndarray, result) -> list:
    start = int(EXCERPT_AT * audio.shape[0])
    window = slice(start, start + EXCERPT_SECONDS * sample_rate)
    return [power_db(sample_rate, audio[window])] + [
        power_db(sample_rate, getattr(result, stem)[window]) for stem in STEMS
    ]


def hero_figure(spectra: dict[str, list]) -> None:
    rows = ["Input", "Center", "Only", "Surround"]
    for mode, theme in THEMES.items():
        cmap = LinearSegmentedColormap.from_list("ramp", theme["ramp"])
        fig, axes = plt.subplots(len(rows), len(spectra), figsize=(2.3 * len(spectra), 7),
                                 sharex=True, sharey=True, dpi=150, squeeze=False)
        fig.patch.set_facecolor(theme["surface"])
        for col, (name, panels) in enumerate(spectra.items()):
            axes[0, col].set_title(label(name), color=theme["ink"], fontsize=10, pad=6)
            peak = panels[0][2].max()  # each column is relative to its own input peak
            for row, (freqs, times, db) in enumerate(panels):
                ax = axes[row, col]
                mesh = ax.pcolormesh(times, freqs / 1000, db - peak, cmap=cmap, vmin=-60,
                                     vmax=0, shading="auto", rasterized=True)
                ax.set_ylim(0, 8)
                ax.tick_params(colors=theme["muted"], labelsize=8)
                for spine in ax.spines.values():
                    spine.set_visible(False)
        for row, title in enumerate(rows):
            axes[row, 0].set_ylabel(f"{title}\nkHz", color=theme["ink"], fontsize=10)
        fig.supxlabel(f"Time (s), {EXCERPT_SECONDS} s excerpt from each track",
                      color=theme["muted"], fontsize=9)
        bar = fig.colorbar(mesh, ax=axes, shrink=0.5, pad=0.015, aspect=30)
        bar.set_label("Power (dB re. input peak)", color=theme["muted"])
        bar.ax.tick_params(colors=theme["muted"], labelsize=8)
        bar.outline.set_visible(False)
        fig.savefig(OUT / f"spectrograms-{mode}.png", facecolor=theme["surface"],
                    bbox_inches="tight")
        plt.close(fig)


def energy_figure(shares: dict[str, list[float]]) -> None:
    names = list(shares)[::-1]
    for mode, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(10, 0.5 + 0.55 * len(names)), dpi=150)
        fig.patch.set_facecolor(theme["surface"])
        ax.set_facecolor(theme["surface"])
        left = np.zeros(len(names))
        for i, stem in enumerate(STEMS):
            widths = np.array([shares[n][i] for n in names])
            ax.barh(names, widths, left=left, height=0.6, color=theme["series"][i],
                    edgecolor=theme["surface"], linewidth=2, label=stem.capitalize())
            for y, (x0, w) in enumerate(zip(left, widths, strict=True)):
                if w >= 0.04:
                    ax.text(x0 + w / 2, y, f"{w:.0%}", ha="center", va="center",
                            color="#ffffff", fontsize=9, fontweight="bold")
            left += widths
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.tick_params(colors=theme["ink"], length=0, labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(ncols=3, loc="lower left", bbox_to_anchor=(0, 1), frameon=False,
                  labelcolor=theme["ink"], fontsize=10, handlelength=1, handleheight=1)
        fig.savefig(OUT / f"energy-{mode}.png", facecolor=theme["surface"], bbox_inches="tight")
        plt.close(fig)


def transform_table(sample_rate: int, audio: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for transform in ("stft", "frft", "wavelet"):
        started = time.perf_counter()
        result = split(sample_rate, audio, transform)
        elapsed = time.perf_counter() - started
        residual = audio - (result.center + result.only + result.surround)
        rows.append({
            "transform": transform,
            "max_abs_residual": float(np.max(np.abs(residual))),
            "realtime_factor": audio.shape[0] / sample_rate / elapsed,
        })
    return rows


def label(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks", nargs="+", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    spectra, shares, durations, transforms = {}, {}, {}, []
    for path in args.tracks:
        sample_rate, audio = load(path)
        result = split(sample_rate, audio)
        spectra[path.stem] = excerpt_spectra(sample_rate, audio, result)
        durations[path.stem] = audio.shape[0] / sample_rate
        energy = [float(np.sum(getattr(result, stem) ** 2)) for stem in STEMS]
        shares[path.stem] = [e / sum(energy) for e in energy]
        start = int(EXCERPT_AT * audio.shape[0])
        transforms.append(transform_table(sample_rate, audio[start:start + 20 * sample_rate]))
    hero_figure(spectra)
    energy_figure({label(name): share for name, share in shares.items()})
    summary = {
        row["transform"]: {
            "worst_max_abs_residual": max(t[i]["max_abs_residual"] for t in transforms),
            "median_realtime_factor": float(np.median([t[i]["realtime_factor"] for t in transforms])),
        }
        for i, row in enumerate(transforms[0])
    }
    print(json.dumps({"energy_shares": shares, "durations": durations, "transforms": summary},
                     indent=2))


if __name__ == "__main__":
    main()
