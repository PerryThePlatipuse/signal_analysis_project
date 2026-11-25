"""Prepare one full waveform per source recording and save its provenance."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt

from ecg.sources import list_records, ptb_metadata, ptb_rows, read_signal

SAMPLE_RATE = 250


def prepare_signal(signal: np.ndarray, input_rate: int) -> np.ndarray:
    if signal.ndim != 1 or len(signal) < 3 * input_rate or not np.isfinite(signal).all():
        raise ValueError("Expected at least three seconds of finite, single-channel ECG")
    if input_rate != SAMPLE_RATE:
        divisor = gcd(input_rate, SAMPLE_RATE)
        signal = resample_poly(signal, SAMPLE_RATE // divisor, input_rate // divisor)
    filtered = sosfiltfilt(butter(3, (0.5, 40), "bandpass", fs=SAMPLE_RATE, output="sos"), signal)
    centered = filtered - np.median(filtered)
    scale = np.sqrt(np.mean(centered ** 2))
    if scale < 1e-8:
        raise ValueError("Flat waveform after filtering")
    return (centered / scale).astype(np.float32)


def r_peaks(signal: np.ndarray) -> np.ndarray:
    """A small QRS detector used for QC and optional RR statistics."""
    rate = SAMPLE_RATE
    qrs = sosfiltfilt(butter(2, (5, 20), "bandpass", fs=rate, output="sos"), signal)
    gradient = np.diff(qrs, prepend=qrs[0])
    window = max(1, round(0.08 * rate))
    energy = np.convolve(gradient ** 2, np.ones(window) / window, mode="same")
    floor = np.median(energy)
    prominence = max(1e-8, 0.2 * (np.quantile(energy, 0.98) - floor))
    candidates, _ = find_peaks(energy, distance=round(0.28 * rate), prominence=prominence)
    radius = round(0.08 * rate)
    return np.unique([
        peak - radius + np.argmax(np.abs(signal[peak - radius:peak + radius + 1]))
        for peak in candidates if radius <= peak < len(signal) - radius
    ]).astype(int)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def prepare(dataset: str, raw: Path, output: Path) -> None:
    source = list_records(dataset, raw)
    rows, signals, excluded = [], [], []
    waveforms = {}
    source_counts = Counter(record.participant for record in source)
    for record in source:
        if source_counts[record.participant] < 2:
            excluded.append({"record_id": record.record_id, "reason": "no repeat record"})
            continue
        try:
            signal, rate = read_signal(record)
            processed = prepare_signal(signal, rate)
        except (OSError, ValueError) as exc:
            excluded.append({"record_id": record.record_id, "reason": str(exc)})
            continue
        # Hash the exact original waveform, before any processing or resampling.
        key = (rate, hashlib.sha256(np.asarray(signal, dtype="<f8").tobytes()).hexdigest())
        if key in waveforms:
            previous = waveforms[key]
            if record.participant != previous.participant:
                raise ValueError(f"Duplicate waveform across identities: {previous.record_id}, {record.record_id}")
            excluded.append({"record_id": record.record_id, "reason": f"duplicate: {previous.record_id}"})
            continue
        waveforms[key] = record
        rows.append(dict(
            dataset=dataset, participant=record.participant, record_id=record.record_id,
            session=record.session, date=record.date, source_path=record.path.relative_to(raw).as_posix(),
            source_fs=rate, source_samples=len(signal), samples=len(processed),
            detected_r_peaks=len(r_peaks(processed)),
        ))
        signals.append(processed)

    if not rows:
        raise ValueError(f"All {len(source)} source recordings were excluded: {excluded[:3]}")
    participant_counts = Counter(row["participant"] for row in rows)
    sessions = defaultdict(set)
    for row in rows:
        if row["session"]:
            sessions[row["participant"]].add(row["session"])
    inventory = dict(
        dataset=dataset, source_records=len(source), source_participants=len(source_counts),
        prepared_records=len(rows), prepared_participants=len(participant_counts),
        repeat_participants=sum(count >= 2 for count in participant_counts.values()),
        multisession_participants=sum(len(values) >= 2 for values in sessions.values()),
        excluded_records=len(excluded), sample_rate=SAMPLE_RATE,
        sessions=dict(sorted(Counter(row["session"] or "unknown" for row in rows).items())),
    )
    if dataset == "ptb-xl":
        inventory["expected_source_records"] = len(ptb_rows(ptb_metadata(raw)))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "records.csv", rows)
    write_csv(output / "excluded.csv", excluded, ["record_id", "reason"])
    offsets = np.r_[0, np.cumsum([len(x) for x in signals])]
    np.savez_compressed(output / "signals.npz", signals=np.concatenate(signals), offsets=offsets)
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({key: value for key, value in inventory.items() if key != "sessions"}, indent=2))


def load_prepared(path: Path) -> tuple[list[dict], list[np.ndarray]]:
    rows = read_csv(path / "records.csv")
    with np.load(path / "signals.npz", allow_pickle=False) as saved:
        values, offsets = saved["signals"], saved["offsets"]
    if len(offsets) != len(rows) + 1 or any(np.diff(offsets) != [int(row["samples"]) for row in rows]):
        raise ValueError("Prepared signals and metadata disagree. Run prepare again.")
    return rows, [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
