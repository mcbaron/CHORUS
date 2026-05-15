from __future__ import annotations

import numpy as np
import pytest
from chorus.transforms import FrFTTransform, STFTConfig, STFTTransform, WaveletTransform


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


def test_experimental_wavelet_adapter_smoke(stereo_identical: np.ndarray) -> None:
    transform = WaveletTransform(wavelet="db4", level=3)

    representation = transform.forward(stereo_identical)

    assert representation.data.shape[0] == 2
    assert representation.metadata["experimental"] is True
    assert representation.metadata["transform"] == "wavelet"
