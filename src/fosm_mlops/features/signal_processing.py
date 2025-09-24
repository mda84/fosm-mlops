"""Signal processing utilities for fiber-optic pipeline monitoring."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import signal


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("Window must be positive")
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def savitzky_golay(
    values: np.ndarray, window_length: int = 5, polyorder: int = 2
) -> np.ndarray:
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    return signal.savgol_filter(
        values, window_length=window_length, polyorder=polyorder
    )


def butterworth_filter(
    values: np.ndarray,
    cutoff: tuple[float, float] | float,
    fs: float,
    order: int = 5,
    btype: str = "low",
) -> np.ndarray:
    nyq = 0.5 * fs
    if isinstance(cutoff, tuple):
        normal_cutoff: float | list[float] = [c / nyq for c in cutoff]
    else:
        normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    return signal.filtfilt(b, a, values)


def stft_spectrogram(
    values: np.ndarray, fs: float, nperseg: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, zxx = signal.stft(values, fs=fs, nperseg=nperseg)
    return f, t, np.abs(zxx)


def fft_spectrum(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    fft_vals = np.fft.rfft(values)
    fft_freq = np.fft.rfftfreq(len(values), 1.0 / fs)
    return fft_freq, np.abs(fft_vals)


def wavelet_transform(values: np.ndarray, widths: Iterable[int]) -> np.ndarray:
    return signal.cwt(values, signal.ricker, widths)


def root_mean_square(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def peak_to_peak(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values))


def spectral_centroid(freqs: np.ndarray, spectrum: np.ndarray) -> float:
    weighted_freq = np.sum(freqs * spectrum)
    magnitude = np.sum(spectrum) + 1e-10
    return float(weighted_freq / magnitude)


def spectral_entropy(spectrum: np.ndarray) -> float:
    prob = spectrum / (np.sum(spectrum) + 1e-10)
    entropy = -np.sum(prob * np.log2(prob + 1e-12))
    return float(entropy)


def band_energy(
    freqs: np.ndarray, spectrum: np.ndarray, band: tuple[float, float]
) -> float:
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.sum(spectrum[mask]))
