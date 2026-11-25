"""Controlled additive stress on full, already preprocessed ECG recordings."""

import hashlib

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ecg.preprocess import SAMPLE_RATE

NOISE_TYPES = ("drift", "band-noise")
BANDPASS = butter(3, (0.5, 40), "bandpass", fs=SAMPLE_RATE, output="sos")


def perturb(signal: np.ndarray, kind: str, relative_rms: float,
            rng: np.random.Generator) -> np.ndarray:
    if kind == "drift":
        time = np.arange(len(signal)) / SAMPLE_RATE
        noise = np.sin(2 * np.pi * 0.25 * time + rng.uniform(0, 2 * np.pi))
    elif kind == "band-noise":
        noise = sosfiltfilt(BANDPASS, rng.standard_normal(len(signal)))
    else:
        raise ValueError(f"Unknown noise type: {kind}")
    if not np.isfinite(relative_rms) or relative_rms <= 0:
        raise ValueError("Relative RMS must be finite and positive")
    noise -= noise.mean()
    scale = relative_rms * np.linalg.norm(signal) / np.linalg.norm(noise)
    return (signal + scale * noise).astype(np.float32)


def query_rng(record_id: str, seed: int) -> np.random.Generator:
    # Same noise for a recording across methods and levels, independent of query order.
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def augment_training(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        return signal
    return perturb(signal, str(rng.choice(NOISE_TYPES)), rng.uniform(0.1, 0.5), rng)
