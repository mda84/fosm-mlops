from __future__ import annotations

import numpy as np

from fosm_mlops.features import signal_processing as sp


def test_moving_average_basic():
    values = np.array([1, 2, 3, 4, 5], dtype=float)
    result = sp.moving_average(values, window=3)
    assert result.shape == values.shape
    assert np.isclose(result[2], 3.0)


def test_fft_and_features():
    values = np.sin(np.linspace(0, 2 * np.pi, 128))
    freq, spectrum = sp.fft_spectrum(values, fs=128)
    centroid = sp.spectral_centroid(freq, spectrum)
    entropy = sp.spectral_entropy(spectrum)
    band_energy = sp.band_energy(freq, spectrum, (0, 10))
    assert centroid > 0
    assert entropy >= 0
    assert band_energy >= 0


def test_butterworth_filter():
    values = np.random.rand(512)
    filtered = sp.butterworth_filter(values, cutoff=10, fs=100, order=3, btype="low")
    assert filtered.shape == values.shape
