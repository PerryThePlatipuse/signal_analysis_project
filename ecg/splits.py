"""Whole-record splits for known and previously unseen participants."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ecg.preprocess import read_csv, write_csv

PROTOCOLS = ("known", "unseen")


def make_split(rows: list[dict], protocol: str, seed: int = 42) -> list[dict]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol}")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["session"]:
            grouped[row["participant"]][row["session"]].append(row)
    eligible = sorted(p for p, sessions in grouped.items() if len(sessions) >= 2)
    rng = np.random.default_rng(seed)
    manifest = []
    if len(eligible) < (6 if protocol == "unseen" else 2):
        raise ValueError(f"{protocol} needs at least {6 if protocol == 'unseen' else 2} people with repeated sessions")
    shuffled = rng.permutation(eligible).tolist()
    heldout = max(2, round(0.2 * len(eligible)))
    cohorts = {p: ("test" if i < heldout else "val" if i < 2 * heldout else "train")
               for i, p in enumerate(shuffled)}
    for participant in eligible:
        sessions = grouped[participant]
        first, *later = sorted(sessions)
        cohort = cohorts[participant] if protocol == "unseen" else "known"
        initial = sorted(sessions[first], key=lambda row: row["record_id"])
        if cohort == "train":
            subsets = [(row, "train") for group in sessions.values() for row in group]
        elif cohort == "known":
            # One validation recording from the earliest session, when possible.
            # Later sessions for the same identity remain exclusively queries.
            subsets = [(row, "known_val" if j == len(initial) - 1 and j > 0 else "train")
                       for j, row in enumerate(initial)]
            subsets += [(row, "known_query") for session in later for row in sessions[session]]
        else:
            subsets = [(row, f"{cohort}_enroll") for row in initial]
            subsets += [(row, f"{cohort}_query") for session in later for row in sessions[session]]
        for row, subset in subsets:
            manifest.append({**row, "split": subset})
    validate_split(manifest, protocol)
    return manifest


def validate_split(rows: list[dict], protocol: str) -> None:
    if not rows or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Missing rows or duplicate source records in split")
    roles = defaultdict(lambda: defaultdict(set))
    people = defaultdict(set)
    for row in rows:
        participant = row["participant"]
        if not row["session"]:
            raise ValueError("Split includes a record with unknown session")
        subset = row["split"]
        roles[participant][subset].add(row["session"])
        people[subset].add(participant)
    if protocol == "known":
        if set(people) - {"train", "known_val", "known_query"}:
            raise ValueError("Unexpected role in known split")
        if people["train"] != people["known_query"] or not people["known_val"] <= people["train"]:
            raise ValueError("Known protocol participant coverage mismatch")
        for parts in roles.values():
            if (parts["train"] | parts["known_val"]) & parts["known_query"]:
                raise ValueError("Known query session leaked into development")
    elif protocol == "unseen":
        if set(people) != {"train", "val_enroll", "val_query", "test_enroll", "test_query"}:
            raise ValueError("Unseen split must have complete train, validation and test roles")
        for subset in ("val", "test"):
            if people[f"{subset}_enroll"] != people[f"{subset}_query"]:
                raise ValueError(f"{subset} enrollment and query participants differ")
            for participant in people[f"{subset}_enroll"]:
                parts = roles[participant]
                if parts[f"{subset}_enroll"] & parts[f"{subset}_query"]:
                    raise ValueError(f"{subset} enrollment session overlaps query")
        cohorts = [people["train"], people["val_enroll"], people["test_enroll"]]
        if any(left & right for i, left in enumerate(cohorts) for right in cohorts[i+1:]):
            raise ValueError("Participant leaked across unseen cohorts")
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def write_splits(path: Path, seed: int) -> None:
    source = read_csv(path / "records.csv")
    splits = {name: make_split(source, name, seed) for name in PROTOCOLS}
    for protocol, rows in splits.items():
        write_csv(path / f"{protocol}.csv", rows)
        print(f"{protocol}: {dict(Counter(row['split'] for row in rows))}")
    (path / "split_seed.json").write_text(json.dumps({"seed": seed}) + "\n")


def indices(path: Path, protocol: str, rows: list[dict], subset: str) -> list[int]:
    split = read_csv(path / f"{protocol}.csv")
    validate_split(split, protocol)
    original = {row["record_id"]: (i, row) for i, row in enumerate(rows)}
    picks = []
    for row in split:
        if row["record_id"] not in original:
            raise ValueError("Split references missing records. Recreate it with split.")
        index, prepared = original[row["record_id"]]
        if row != {**prepared, "split": row["split"]}:
            raise ValueError("Prepared metadata changed. Recreate the split.")
        if row["split"] == subset:
            picks.append(index)
    return picks
