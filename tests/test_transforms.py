from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from chorus.core import ChorusConfig, ChorusProcessor
from chorus.transforms import FrFTTransform, STFTConfig, STFTTransform, WaveletTransform

WAV_FILES = [
    "tests/test_tracks_wav/PinkPanther.wav",
    "tests/test_tracks_wav/TVSong.wav",
]


def _load_wav_stereo(wav_path: str) -> np.ndarray:
    audio, _ = sf.read(wav_path, dtype="float64")
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio


def test_stft_round_trip_pass_through_stereo(stereo_identical: np.ndarray) -> None:
    transform = STFTTransform(STFTConfig(frame_size=1024, hop_size=512))

    representation = transform.forward(stereo_identical)
    reconstructed = transform.inverse(representation)

    assert reconstructed.shape == stereo_identical.shape
    np.testing.assert_allclose(reconstructed, stereo_identical, atol=1e-10, rtol=1e-10)


def test_stft_metadata_records_reference_settings(stereo_identical: np.ndarray) -> None:
    transform = STFTTransform(STFTConfig(frame_size=1024, hop_size=512))

    representation = transform.forward(stereo_identical)

    assert representation.metadata["transform"] == "stft"
    assert representation.metadata["frame_size"] == 1024
    assert representation.metadata["hop_size"] == 512
    assert representation.metadata["window"] == "sqrt_hann"


def test_stft_rejects_non_invertible_hop_size() -> None:
    with pytest.raises(ValueError, match="invertible"):
        STFTTransform(STFTConfig(frame_size=1024, hop_size=1024))


def test_stft_rejects_too_short_stereo_input() -> None:
    transform = STFTTransform()
    too_short = np.zeros((512, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="at least 1024 samples"):
        transform.forward(too_short)


def test_experimental_frft_adapter_smoke(stereo_identical: np.ndarray) -> None:
    transform = FrFTTransform(order=0.75)

    representation = transform.forward(stereo_identical)

    assert representation.data.shape[0] == 2
    assert representation.metadata["experimental"] is True
    assert representation.metadata["transform"] == "frft"


def test_frft_round_trip_pass_through_stereo(stereo_identical: np.ndarray) -> None:
    transform = FrFTTransform(order=0.75)

    representation = transform.forward(stereo_identical)
    reconstructed = transform.inverse(representation)

    assert reconstructed.shape == stereo_identical.shape
    np.testing.assert_allclose(reconstructed, stereo_identical, atol=1e-10, rtol=1e-10)


def test_experimental_wavelet_adapter_smoke(stereo_identical: np.ndarray) -> None:
    transform = WaveletTransform(wavelet="db4", level=3)

    representation = transform.forward(stereo_identical)

    assert representation.data.shape[0] == 2
    assert representation.metadata["experimental"] is True
    assert representation.metadata["transform"] == "wavelet"


def test_wavelet_round_trip_pass_through_stereo(stereo_identical: np.ndarray) -> None:
    transform = WaveletTransform(wavelet="db4", level=3)

    representation = transform.forward(stereo_identical)
    reconstructed = transform.inverse(representation)

    assert reconstructed.shape == stereo_identical.shape
    np.testing.assert_allclose(reconstructed, stereo_identical, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("wav_path", WAV_FILES)
def test_stft_round_trip_wav(wav_path: str) -> None:
    audio = _load_wav_stereo(wav_path)
    transform = STFTTransform(STFTConfig(frame_size=1024, hop_size=512))

    representation = transform.forward(audio)
    reconstructed = transform.inverse(representation)

    np.testing.assert_allclose(reconstructed, audio, atol=1e-10, rtol=1e-10)


def test_frft_ozaktas_round_trip_synthetic() -> None:
    """New Ozaktas-Kutay FrFT must round-trip a synthetic signal within 1e-10."""
    rng = np.random.default_rng(42)
    audio = rng.standard_normal((4096, 2))
    t = FrFTTransform(order=0.5, frame_size=1024)
    rep = t.forward(audio)
    recovered = t.inverse(rep)
    np.testing.assert_allclose(recovered, audio, atol=1e-10)


def test_frft_identity_order_zero() -> None:
    """order=0.0 forward must return input unchanged."""
    rng = np.random.default_rng(7)
    audio = rng.standard_normal((1024, 2))
    t = FrFTTransform(order=0.0, frame_size=1024)
    rep = t.forward(audio)
    recovered = t.inverse(rep)
    np.testing.assert_allclose(recovered, audio, atol=1e-12)


def test_frft_order_one_matches_fft() -> None:
    """order=1.0 forward must match np.fft.fft per frame per channel."""
    rng = np.random.default_rng(13)
    audio = rng.standard_normal((1024, 2))
    t = FrFTTransform(order=1.0, frame_size=1024)
    rep = t.forward(audio)
    expected_left = np.fft.fft(audio[:, 0])
    expected_right = np.fft.fft(audio[:, 1])
    np.testing.assert_allclose(rep.data[0], expected_left, atol=1e-10)
    np.testing.assert_allclose(rep.data[1], expected_right, atol=1e-10)


@pytest.mark.parametrize("wav_path", WAV_FILES)
def test_frft_round_trip_wav(wav_path: str) -> None:
    audio = _load_wav_stereo(wav_path)
    transform = FrFTTransform(order=0.5)

    representation = transform.forward(audio)
    reconstructed = transform.inverse(representation)

    np.testing.assert_allclose(reconstructed, audio, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("wav_path", WAV_FILES)
def test_frft_ozaktas_round_trip_wav(wav_path: str) -> None:
    """Ozaktas-Kutay FrFT must round-trip real WAV files with non-default frame_size=512."""
    audio = _load_wav_stereo(wav_path)
    t = FrFTTransform(order=0.5, frame_size=512)
    rep = t.forward(audio)
    recovered = t.inverse(rep)
    np.testing.assert_allclose(recovered, audio, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("wav_path", WAV_FILES)
def test_wavelet_round_trip_wav(wav_path: str) -> None:
    audio = _load_wav_stereo(wav_path)
    transform = WaveletTransform(wavelet="db4", level=3)

    representation = transform.forward(audio)
    reconstructed = transform.inverse(representation)

    np.testing.assert_allclose(reconstructed, audio, atol=1e-10, rtol=1e-10)


def test_frft_transform_respects_order_from_config() -> None:
    """FrFTTransform constructed via ChorusConfig must use frft_order."""
    config = ChorusConfig(sample_rate=48_000, transform="frft", frft_order=0.25, frame_size=1024)
    processor = ChorusProcessor(config)
    assert processor.transform.order == 0.25
