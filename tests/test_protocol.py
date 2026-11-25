import numpy as np
import pytest

from ecg.classical import dtw_distance, normalized_correlation
from ecg.preprocess import SAMPLE_RATE, prepare_signal
from ecg.splits import make_split, validate_split


def recordings():
    return [
        {"participant": str(person), "session": str(session), "record_id": f"{person}/{session}/{n}"}
        for person in range(8) for session in range(2) for n in range(2)
    ] + [{"participant": "no-repeat", "session": "0", "record_id": "only"}]


def test_no_session_leakage_in_known_split():
    rows = make_split(recordings(), "known", seed=9)
    assert {row["session"] for row in rows if row["split"] == "known_query"} == {"1"}
    assert {row["session"] for row in rows if row["split"] == "train"} == {"0"}
    assert all(row["participant"] != "no-repeat" for row in rows)
    train = next(row for row in rows if row["split"] == "train")
    query = next(row for row in rows if row["participant"] == train["participant"] and
                 row["split"] == "known_query")
    with pytest.raises(ValueError, match="session leaked"):
        validate_split([dict(row, session="0") if row is query else row for row in rows], "known")


def test_unseen_groups_and_enrollment_use_original_records():
    rows = make_split(recordings(), "unseen", seed=9)
    people = lambda subset: {r["participant"] for r in rows if r["split"] == subset}
    assert people("test_enroll") == people("test_query")
    assert people("val_enroll") == people("val_query")
    assert not people("train") & (people("val_enroll") | people("test_enroll"))
    assert not people("val_enroll") & people("test_enroll")
    assert {r["session"] for r in rows if r["split"].endswith("enroll")} == {"0"}
    assert len({row["record_id"] for row in rows}) == len(rows) == 32
    assert make_split(recordings(), "unseen", seed=9) == rows


def test_full_duration_and_similarity():
    fs = 500
    time = np.arange(4 * fs) / fs
    waveform = np.sin(2 * np.pi * 7 * time) + 0.3 * np.sin(2 * np.pi * 19 * time)
    full = prepare_signal(waveform, fs)
    assert full.shape == (4 * SAMPLE_RATE,)
    assert full.dtype == np.float32
    shifted = np.r_[np.zeros(15, dtype=full.dtype), full[:-15]]
    assert normalized_correlation(full, shifted) > 0.99
    assert dtw_distance(full, full) == pytest.approx(0)
    assert dtw_distance(full, shifted) > 0


def test_cnn_encoding_independent_of_other_record_length():
    torch = pytest.importorskip("torch")
    from ecg.neural import ECGEncoder, collate

    x = np.random.default_rng(1).normal(size=753).astype(np.float32)
    y = np.random.default_rng(2).normal(size=1420).astype(np.float32)
    model = ECGEncoder(3).eval()
    single, len_single, _ = collate([(x, 0)])
    batch, len_batch, _ = collate([(x, 0), (y, 1)])
    with torch.inference_mode():
        torch.testing.assert_close(model.encode(single, len_single)[0], model.encode(batch, len_batch)[0])
