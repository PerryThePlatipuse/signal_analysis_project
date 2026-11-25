"""Classical whole-record ECG features and pairwise similarity."""

import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly, welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ecg.preprocess import SAMPLE_RATE, load_prepared, r_peaks
from ecg.splits import indices


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Best normalized correlation within 600 ms, on overlapping samples."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lags = correlation_lags(len(a), len(b))
    keep = np.abs(lags) <= round(0.6 * SAMPLE_RATE)
    offsets = lags[keep]
    left_a, left_b = np.maximum(offsets, 0), np.maximum(-offsets, 0)
    length = np.minimum(len(a) - left_a, len(b) - left_b)
    valid = length >= 0.7 * min(len(a), len(b))

    def window_sum(x, left):
        cumsum = np.r_[0, np.cumsum(x)]
        return cumsum[left + length] - cumsum[left]

    sum_a, sum_b = window_sum(a, left_a), window_sum(b, left_b)
    ss_a = window_sum(a * a, left_a) - sum_a ** 2 / length
    ss_b = window_sum(b * b, left_b) - sum_b ** 2 / length
    denominator = np.sqrt(np.maximum(ss_a, 0) * np.maximum(ss_b, 0))
    products = correlate(a, b, method="fft")[keep] - sum_a * sum_b / length
    valid &= denominator > 1e-10
    if not valid.any():
        raise ValueError("No suitable overlap for normalized correlation")
    return float(np.clip(np.max(products[valid] / denominator[valid]), -1, 1))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Banded DTW at 50 Hz for an exploratory comparison of entire records."""
    a, b = resample_poly(a, 1, 5), resample_poly(b, 1, 5)
    band = max(abs(len(a) - len(b)), round(0.10 * max(len(a), len(b))))
    previous = np.full(len(b) + 1, np.inf)
    previous[0] = 0
    for i, value in enumerate(a, 1):
        row = np.full_like(previous, np.inf)
        for j in range(max(1, i - band), min(len(b), i + band) + 1):
            row[j] = (value - b[j - 1]) ** 2 + min(previous[j], row[j - 1], previous[j - 1])
        previous = row
    return float(np.sqrt(previous[-1] / (len(a) + len(b))))


def features(signal: np.ndarray) -> np.ndarray:
    frequencies, power = welch(signal, fs=SAMPLE_RATE, nperseg=min(500, len(signal)))
    bands = [(0.5, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 25), (25, 32), (32, 40)]
    total = power.sum() + 1e-12
    spectrum = [np.log1p(100 * power[(frequencies >= low) & (frequencies < high)].sum() / total)
                for low, high in bands]
    ac = correlate(signal, signal, method="fft")[len(signal) - 1:]
    peaks = r_peaks(signal)
    rr = np.diff(peaks) / SAMPLE_RATE
    rr = rr[(rr >= 0.28) & (rr <= 2.0)]
    statistics = [
        np.median(rr) if rr.size else 0.0,
        np.std(rr) if rr.size else 0.0,
        float(rr.size > 0),
        float(ac[round(0.28 * SAMPLE_RATE):2 * SAMPLE_RATE].max() / ac[0]),
    ]
    return np.r_[spectrum, np.quantile(signal, (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)),
                 np.mean(np.abs(np.diff(signal))), statistics,
                 ac[np.arange(5, 151, 5)] / ac[0]]


def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def project_features(matrix, scale_mean, scale, pca_mean, components):
    projected = np.einsum("ij,kj->ik", (matrix - scale_mean) / scale - pca_mean, components)
    if not np.isfinite(projected).all():
        raise ValueError("Non-finite values after PCA")
    return unit_rows(projected).astype(np.float32)


def encode_pca(path, signals):
    matrix = np.stack([features(signal) for signal in signals])
    with np.load(path, allow_pickle=False) as fit:
        return project_features(matrix, **fit)


def baseline(path: Path, protocol: str, output: Path, dimensions: int) -> None:
    records, signals = load_prepared(path)
    train = indices(path, protocol, records, "train")
    if len(train) < 2 or dimensions < 1:
        raise ValueError("PCA needs at least two training recordings and one requested component")
    matrix = np.stack([features(signal) for signal in signals])
    scale = StandardScaler().fit(matrix[train])
    z = scale.transform(matrix)
    pca = PCA(n_components=min(dimensions, len(train) - 1, z.shape[1]), svd_solver="full").fit(z[train])
    vectors = project_features(matrix, scale.mean_, scale.scale_, pca.mean_, pca.components_)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "embeddings.npz", record_ids=[r["record_id"] for r in records],
                        embeddings=vectors)
    np.savez_compressed(output / "pca.npz", scale_mean=scale.mean_, scale=scale.scale_,
                        pca_mean=pca.mean_, components=pca.components_)
    summary = {"protocol": protocol, "training_records": len(train),
               "training_record_ids": [records[i]["record_id"] for i in train],
               "embeddings": len(vectors), "dimensions": pca.n_components_,
               "explained_variance": float(pca.explained_variance_ratio_.sum())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: val for key, val in summary.items() if key != "training_record_ids"}, indent=2))


def compare(path: Path, left: str, right: str) -> None:
    records, signals = load_prepared(path)
    by_id = {row["record_id"]: signal for row, signal in zip(records, signals)}
    a, b = by_id[left], by_id[right]
    print(json.dumps({"correlation": normalized_correlation(a, b),
                      "dtw_distance": dtw_distance(a, b)}, indent=2))
