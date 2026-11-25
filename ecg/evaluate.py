"""Evaluate identification and verification on the shared recording splits."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from ecg.augment import NOISE_TYPES, perturb, query_rng
from ecg.classical import dtw_distance, encode_pca, normalized_correlation, unit_rows
from ecg.preprocess import load_prepared, read_csv, write_csv
from ecg.splits import indices

METHODS = ("pca", "cnn", "correlation", "dtw")


def load_embeddings(run, method, protocol, records, training):
    metadata = json.loads((run / {"pca": "summary.json", "cnn": "training.json"}[method]).read_text())
    train_ids = sorted(records[i]["record_id"] for i in training)
    if metadata.get("protocol") != protocol or sorted(metadata.get("training_record_ids", ())) != train_ids:
        raise ValueError("Model training records do not match this protocol. Rerun baseline or train.")
    with np.load(run / "embeddings.npz", allow_pickle=False) as data:
        ids, vectors = data["record_ids"].tolist(), data["embeddings"]
    expected = [r["record_id"] for r in records]
    if (vectors.ndim != 2 or vectors.shape[0] != len(ids) or not vectors.shape[1]
            or len(ids) != len(set(ids)) or set(ids) != set(expected) or not np.isfinite(vectors).all()):
        raise ValueError("Embeddings do not match the prepared recordings")
    order = {record_id: i for i, record_id in enumerate(ids)}
    return normalize_embeddings(vectors[[order[record_id] for record_id in expected]]), metadata


def normalize_embeddings(vectors):
    return unit_rows(np.asarray(vectors, dtype=np.float64))


def perturbed_values(path, method, run, values, metadata, records, queries, kind, noise_rms, seed):
    _, waveforms = load_prepared(path)
    signals = [perturb(waveforms[i], kind, noise_rms, query_rng(records[i]["record_id"], seed))
               for i in queries]
    if method == "pca":
        new_embeddings = encode_pca(run / "pca.npz", signals)
    else:
        from ecg.neural import encode_checkpoint

        new_embeddings = encode_checkpoint(run / "model.pt", signals, metadata)
    modified = values.copy()
    modified[queries] = normalize_embeddings(new_embeddings)
    return modified


def score_gallery(values, method, records, gallery, queries):
    grouped = defaultdict(list)
    for i in gallery:
        grouped[records[i]["participant"]].append(i)
    people = sorted(grouped)
    if len(people) < 2:
        raise ValueError("The gallery needs at least two enrolled participants")
    if method in ("pca", "cnn"):
        templates = unit_rows(np.stack([values[grouped[person]].mean(axis=0) for person in people]))
        matrix = np.einsum("ik,jk->ij", values[queries], templates)
    else:
        compare = normalized_correlation if method == "correlation" else dtw_distance
        sign = 1 if method == "correlation" else -1
        matrix = np.empty((len(queries), len(people)))
        for row, i in enumerate(queries):
            for col, person in enumerate(people):
                matrix[row, col] = sign * np.mean([compare(values[i], values[j]) for j in grouped[person]])
            if (row + 1) % 100 == 0:
                print(f"{method}: scored {row + 1}/{len(queries)} queries", flush=True)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite similarities in the gallery")
    return people, matrix


def verification_pairs(truth, people, scores, negatives, seed):
    rng = np.random.default_rng(seed)
    result = []
    person_index = {p: i for i, p in enumerate(people)}
    for key, true, row in zip(truth[0], truth[1], scores):
        pos = person_index[true]
        neg = rng.choice(len(people) - 1, size=min(negatives, len(people) - 1), replace=False)
        ordered = [pos, *sorted(neg + (neg >= pos))]
        result.extend(dict(record_id=key, participant=true, claimed=people[col],
                           genuine=int(col == pos), score=float(row[col])) for col in ordered)
    return result


def verification_metrics(pairs, threshold=None):
    gold = np.asarray([p["genuine"] for p in pairs], dtype=bool)
    scores = np.asarray([p["score"] for p in pairs], dtype=float)
    far, tpr, candidates = roc_curve(gold, scores, drop_intermediate=False)
    frr = 1 - tpr
    delta = far - frr
    right = int(np.searchsorted(delta, 0))
    # Report interpolated EER even when no actual threshold gives equal rates.
    mix = -delta[right - 1] / (delta[right] - delta[right - 1])
    eer = float(far[right - 1] + mix * (far[right] - far[right - 1]))
    if threshold is None:
        return dict(roc_auc=float(roc_auc_score(gold, scores)), eer=eer), None
    accept = scores >= threshold
    return dict(roc_auc=float(roc_auc_score(gold, scores)), eer=eer,
                far=float(accept[~gold].mean()), frr=float((~accept[gold]).mean())), accept


def calibrate(pairs):
    far, tpr, thresholds = roc_curve([p["genuine"] for p in pairs], [p["score"] for p in pairs],
                                    drop_intermediate=False)
    # ROC orders thresholds from strictest to loosest, preferring lower FAR on ties.
    best = int(np.argmin(np.abs(far - (1 - tpr))))
    threshold = thresholds[best]
    if not np.isfinite(threshold):
        threshold = np.nextafter(max(p["score"] for p in pairs), np.inf)
    return float(threshold)


def accuracy_interval(participants, correct, seed):
    groups = defaultdict(list)
    for participant, match in zip(participants, correct):
        groups[participant].append(match)
    totals = np.asarray([sum(items) for items in groups.values()])
    sizes = np.asarray([len(items) for items in groups.values()])
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(groups), size=(1000, len(groups)))
    estimates = totals[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    return np.quantile(estimates, (0.025, 0.975)).tolist()


def assess(records, queries, people, scores, negatives, seed):
    ids = [records[i]["record_id"] for i in queries]
    gold = [records[i]["participant"] for i in queries]
    best = scores.argmax(axis=1)
    predicted = [people[j] for j in best]
    person_index = {p: i for i, p in enumerate(people)}
    actual = np.array([person_index[p] for p in gold])
    correct = actual == best
    actual_counts = np.bincount(actual, minlength=len(people))
    predicted_counts = np.bincount(best, minlength=len(people))
    true_positives = np.bincount(actual[correct], minlength=len(people))
    present = actual_counts > 0
    macro_f1 = (2 * true_positives[present] / (actual_counts + predicted_counts)[present]).mean()
    metrics = dict(gallery_people=len(people), query_people=int(present.sum()), queries=len(queries),
                   top1=float(correct.mean()), macro_f1=float(macro_f1),
                   top1_ci95=accuracy_interval(gold, correct, seed))
    predictions = [dict(record_id=key, participant=true, predicted=prediction, score=float(score))
                   for key, true, prediction, score in zip(ids, gold, predicted, scores[np.arange(len(ids)), best])]
    return metrics, predictions, verification_pairs((ids, gold), people, scores, negatives, seed)


def evaluate(path: Path, protocol: str, method: str, output: Path, run: Path | None,
             negatives: int, seed: int, max_participants: int | None,
             query_noise: str | None = None, noise_rms: float | None = None) -> None:
    if negatives < 1 or (max_participants is not None and max_participants < 2):
        raise ValueError("Use at least one negative and at least two gallery participants")
    if (method in ("pca", "cnn")) != (run is not None):
        raise ValueError("Pass --model-run for pca/cnn only")
    if (query_noise is None) != (noise_rms is None):
        raise ValueError("Pass --query-noise and --noise-rms together")
    if query_noise is not None and (method not in ("pca", "cnn") or query_noise not in NOISE_TYPES
                                    or not np.isfinite(noise_rms) or noise_rms <= 0):
        raise ValueError("Query noise requires pca/cnn, a supported noise type and a positive finite RMS")
    if run is None:
        records, values = load_prepared(path)
    else:
        records = read_csv(path / "records.csv")
        values, metadata = load_embeddings(run, method, protocol, records,
                                           indices(path, protocol, records, "train"))
    roles = {"known": (("train", "known_val"), ("train", "known_query")),
             "unseen": (("val_enroll", "val_query"), ("test_enroll", "test_query"))}[protocol]
    stages, threshold = {}, None
    for stage, (enroll_role, query_role) in zip(("validation", "test"), roles):
        enroll = indices(path, protocol, records, enroll_role)
        queries = indices(path, protocol, records, query_role)
        gallery = sorted({records[i]["participant"] for i in enroll})
        if max_participants is not None and len(gallery) > max_participants:
            chosen = set(np.random.default_rng(seed).choice(gallery, max_participants, replace=False))
            enroll = [i for i in enroll if records[i]["participant"] in chosen]
            queries = [i for i in queries if records[i]["participant"] in chosen]
        if not queries:
            if stage == "test":
                raise ValueError("No queries in test cohort")
            continue
        print(f"{method}/{protocol}/{stage}: {len(queries)} queries, {len(enroll)} enrollment records", flush=True)
        stage_values = values
        if stage == "test" and query_noise is not None:
            stage_values = perturbed_values(path, method, run, values, metadata, records, queries,
                                            query_noise, noise_rms, seed)
        people, scores = score_gallery(stage_values, method, records, enroll, queries)
        stage_seed = seed + (stage == "test")
        metrics, predictions, pairs = assess(records, queries, people, scores, negatives, stage_seed)
        if stage == "validation":
            threshold = calibrate(pairs)
        verification, accept = verification_metrics(pairs, threshold)
        if accept is not None:
            for pair, accepted in zip(pairs, accept):
                pair["accept"] = int(accepted)
        metrics.update(genuine_pairs=len(queries), impostor_pairs=len(pairs) - len(queries), **verification)
        stages[stage] = (metrics, predictions, pairs)

    output.mkdir(parents=True, exist_ok=True)
    summary = dict(dataset=records[0]["dataset"], protocol=protocol, method=method, seed=seed,
                   max_participants=max_participants, max_negatives_per_query=negatives,
                   threshold_from_validation=threshold,
                   **{stage: result[0] for stage, result in stages.items()})
    if query_noise is not None:
        summary.update(query_noise=query_noise, noise_rms=noise_rms)
    for stage, (_, predictions, pairs) in stages.items():
        write_csv(output / f"{stage}_identification.csv", predictions)
        write_csv(output / f"{stage}_pairs.csv", pairs)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
