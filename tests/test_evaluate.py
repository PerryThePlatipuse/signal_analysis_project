import json

import numpy as np
import pytest

from ecg.evaluate import assess, calibrate, evaluate, verification_metrics
from ecg.preprocess import write_csv


def test_rank_and_verification_separate_matching_and_impostor_claims():
    records = [{"record_id": "r0", "participant": "a"}, {"record_id": "r1", "participant": "b"}]
    scores = np.array([[0.8, 0.2, 0.3], [0.7, 0.6, 0.1]])
    metrics, predictions, pairs = assess(records, [0, 1], ["a", "b", "c"], scores, 10, 42)
    assert metrics["top1"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert [p["predicted"] for p in predictions] == ["a", "a"]
    assert [p["claimed"] for p in pairs] == ["a", "b", "c", "b", "a", "c"]
    assert sum(p["genuine"] for p in pairs) == 2
    metrics, decisions = verification_metrics(pairs, threshold=0.75)
    assert metrics["far"] == 0
    assert metrics["frr"] == 0.5
    assert decisions.sum() == 1


def test_eer_interpolates_when_all_scores_tie():
    pairs = [{"genuine": 1, "score": 0.1}, {"genuine": 0, "score": 0.1}]
    threshold = calibrate(pairs)
    metrics, decisions = verification_metrics(pairs, threshold)
    assert threshold > 0.1
    assert metrics == dict(roc_auc=0.5, eer=0.5, far=0, frr=1)
    assert not decisions.any()


def test_unseen_evaluation_uses_validation_threshold_and_rejects_mismatched_model(tmp_path, monkeypatch):
    prepared, run, output = [tmp_path / name for name in ("prepared", "run", "output")]
    prepared.mkdir()
    run.mkdir()
    roles = ("train", "val_enroll", "val_enroll", "val_query", "val_query",
             "test_enroll", "test_enroll", "test_query", "test_query")
    people = ("p0", "p1", "p2", "p1", "p2", "p3", "p4", "p3", "p4")
    rows = [dict(record_id=f"r{i}", participant=person, dataset="test",
                 session="2" if role.endswith("_query") else "1")
            for i, (role, person) in enumerate(zip(roles, people))]
    write_csv(prepared / "records.csv", rows)
    write_csv(prepared / "unseen.csv", [{**row, "split": role} for row, role in zip(rows, roles)])
    meta = run / "summary.json"
    meta.write_text(json.dumps(dict(protocol="known", training_record_ids=["r0"])))
    vectors = np.array([[1, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [2, 1], [1, 2]])
    np.savez(run / "embeddings.npz", record_ids=[r["record_id"] for r in rows][::-1], embeddings=vectors[::-1])
    with pytest.raises(ValueError, match="training records"):
        evaluate(prepared, "unseen", "pca", output, run, negatives=1, seed=42, max_participants=None)
    meta.write_text(json.dumps(dict(protocol="unseen", training_record_ids=["r0"])))
    evaluate(prepared, "unseen", "pca", output, run, negatives=1, seed=42, max_participants=None)
    summary = json.loads((output / "metrics.json").read_text())
    assert summary["threshold_from_validation"] == 1
    assert summary["test"]["top1"] == 1
    assert summary["test"]["eer"] == 0
    assert summary["test"]["far"] == 0
    assert summary["test"]["frr"] == 1

    def mock_perturb(path, method, run, values, metadata, records, queries, kind, noise_rms, seed):
        assert queries == [7, 8]
        modified = values.copy()
        modified[queries] = [[0, 1], [1, 0]]
        return modified

    monkeypatch.setattr("ecg.evaluate.perturbed_values", mock_perturb)
    evaluate(prepared, "unseen", "pca", output, run, negatives=1, seed=42,
             max_participants=None, query_noise="drift", noise_rms=0.25)
    stressed = json.loads((output / "metrics.json").read_text())
    assert stressed["validation"] == summary["validation"]
    assert stressed["threshold_from_validation"] == summary["threshold_from_validation"]
    assert stressed["test"]["top1"] == 0
