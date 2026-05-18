from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

CONTRIBUTIONS = frozenset({"Lc", "Rc", "Lo", "Ro", "Ls", "Rs"})
FILTER_TYPES = frozenset({"unity", "gain", "mute", "solo", "polarity", "eq"})


@dataclass(frozen=True)
class FilterSpec:
    type: str
    parameters: dict[str, float | str]


FilterChains = dict[str, list[FilterSpec]]


def _as_spec(target: str, index: int, raw: dict[str, Any], sample_rate: int | None) -> FilterSpec:
    filter_type = str(raw.get("type", ""))
    if filter_type not in FILTER_TYPES:
        raise ValueError(f"unsupported filter type {filter_type!r} for {target}[{index}]")
    parameters = {key: value for key, value in raw.items() if key != "type"}
    spec = FilterSpec(type=filter_type, parameters=parameters)
    _validate_spec(target, index, spec, sample_rate)
    return spec


def _validate_spec(target: str, index: int, spec: FilterSpec, sample_rate: int | None) -> None:
    if spec.type == "gain":
        db = float(spec.parameters.get("db", 0.0))
        if db < -60.0 or db > 24.0:
            raise ValueError(f"gain db for {target}[{index}] must be between -60.0 and 24.0")
    if spec.type == "eq":
        mode = str(spec.parameters.get("mode", "peaking"))
        if mode not in {"peaking", "highpass", "lowpass"}:
            raise ValueError(f"eq mode for {target}[{index}] must be peaking, highpass, or lowpass")
        frequency = float(spec.parameters.get("frequency_hz", 0.0))
        if frequency <= 0.0:
            raise ValueError(f"eq frequency_hz for {target}[{index}] must be positive")
        if sample_rate is not None and frequency >= sample_rate / 2.0:
            raise ValueError(f"eq frequency_hz for {target}[{index}] must be below Nyquist")
        q = float(spec.parameters.get("q", 0.707))
        if q <= 0.0:
            raise ValueError(f"eq q for {target}[{index}] must be positive")


def normalize_filter_config(
    raw: dict[str, list[dict[str, Any]]] | None,
    sample_rate: int | None = None,
) -> FilterChains:
    chains: FilterChains = {
        contribution: [FilterSpec(type="unity", parameters={})]
        for contribution in sorted(CONTRIBUTIONS)
    }
    if raw is None:
        return chains
    for target, filters in raw.items():
        if target not in CONTRIBUTIONS:
            raise ValueError(f"unknown contribution {target!r}")
        chain = [_as_spec(target, index, item, sample_rate) for index, item in enumerate(filters)]
        chains[target] = chain or [FilterSpec(type="unity", parameters={})]
    return chains


def _is_unity_chain(chain: list[FilterSpec]) -> bool:
    return len(chain) == 1 and chain[0].type == "unity"


def _apply_spec(audio: np.ndarray, spec: FilterSpec, sample_rate: int) -> np.ndarray:
    if spec.type == "unity":
        return audio
    if spec.type == "gain":
        return audio * (10.0 ** (float(spec.parameters.get("db", 0.0)) / 20.0))
    if spec.type == "mute":
        return np.zeros_like(audio)
    if spec.type == "solo":
        return audio
    if spec.type == "polarity":
        return -audio
    if spec.type == "eq":
        return _apply_eq(audio, spec, sample_rate)
    raise ValueError(f"unsupported filter type {spec.type!r}")


def _apply_eq(audio: np.ndarray, spec: FilterSpec, sample_rate: int) -> np.ndarray:
    mode = str(spec.parameters.get("mode", "peaking"))
    frequency = float(spec.parameters["frequency_hz"])
    q = float(spec.parameters.get("q", 0.707))
    gain_db = float(spec.parameters.get("gain_db", 0.0))
    if mode == "highpass":
        sos = signal.butter(2, frequency, btype="highpass", fs=sample_rate, output="sos")
    elif mode == "lowpass":
        sos = signal.butter(2, frequency, btype="lowpass", fs=sample_rate, output="sos")
    else:
        b, a = signal.iirpeak(frequency, q, fs=sample_rate)
        filtered = signal.lfilter(b, a, audio)
        return audio + filtered * (10.0 ** (gain_db / 20.0) - 1.0)
    return signal.sosfilt(sos, audio)


def _ensure_finite(name: str, audio: np.ndarray) -> None:
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"filtered output for {name} contains NaN or infinity")


def _level(audio: np.ndarray) -> dict[str, float]:
    magnitude = np.abs(audio)
    return {
        "rms": float(np.sqrt(np.mean(np.square(magnitude)))) if audio.size else 0.0,
        "peak": float(np.max(magnitude)) if audio.size else 0.0,
    }


def apply_filter_chains(
    contributions: dict[str, np.ndarray],
    chains: FilterChains,
    sample_rate: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    soloed = sorted(
        name for name, chain in chains.items() if any(spec.type == "solo" for spec in chain)
    )
    filtered: dict[str, np.ndarray] = {}
    pre_levels = {name: _level(audio) for name, audio in contributions.items()}
    post_levels: dict[str, dict[str, float]] = {}
    for name in sorted(CONTRIBUTIONS):
        source = np.real_if_close(contributions[name], tol=1000)
        if np.iscomplexobj(source):
            raise ValueError(f"contribution {name} has non-negligible imaginary values")
        audio = np.asarray(source, dtype=np.float64).copy()
        for spec in chains[name]:
            audio = _apply_spec(audio, spec, sample_rate)
        if soloed and name not in soloed:
            audio = np.zeros_like(audio)
        _ensure_finite(name, audio)
        filtered[name] = audio
        post_levels[name] = _level(audio)
    transparent = not soloed and all(_is_unity_chain(chain) for chain in chains.values())
    return filtered, {
        "chains": {
            name: [{"type": spec.type, "parameters": spec.parameters} for spec in chain]
            for name, chain in chains.items()
        },
        "transparent": transparent,
        "soloed": soloed,
        "pre_levels": pre_levels,
        "post_levels": post_levels,
    }
