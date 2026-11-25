"""Public data acquisition and one-record-at-a-time parsing."""

import csv
import json
import re
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import TextIO
from urllib.request import urlopen

import numpy as np

DATASETS = ("heartprint", "ecg-id", "ptb-xl")
PTB_BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
ECG_ID_BASE = "https://physionet-open.s3.amazonaws.com/ecgiddb/1.0.0/"
HEARTPRINT_API = "https://api.figshare.com/v2/articles/20105354/versions/3"
HEARTPRINT_SESSIONS = {"S1": "01", "S2": "02", "S3R": "03", "S3L": "03"}
PTB_METADATA = "ptbxl_database.csv"


@dataclass(frozen=True)
class Record:
    dataset: str
    participant: str
    record_id: str
    session: str
    path: Path
    date: str = ""


def fetch(url: str, path: Path) -> None:
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(url, timeout=45) as response, temp.open("wb") as out:
            shutil.copyfileobj(response, out)
            length = response.getheader("Content-Length")
            if length and out.tell() != int(length):
                raise OSError(f"Incomplete download: {url}")
        if not temp.stat().st_size:
            raise ValueError(f"Empty download: {url}")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def copy_zip_member(archive: zipfile.ZipFile, item: zipfile.ZipInfo, target: Path,
                    relative: Path) -> None:
    dst = target / relative
    if relative.is_absolute() or ".." in relative.parts or not dst.resolve().is_relative_to(target.resolve()):
        raise ValueError(f"Unexpected archive path: {item.filename}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Do not replace an existing usable file until this member passes the ZIP CRC check.
    temp = dst.with_suffix(dst.suffix + ".part")
    try:
        with archive.open(item) as src, temp.open("wb") as out:
            shutil.copyfileobj(src, out)
        temp.replace(dst)
    finally:
        temp.unlink(missing_ok=True)


def extract_zip(path: Path, target: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir() or "__MACOSX" in Path(item.filename).parts:
                continue
            copy_zip_member(archive, item, target, Path(item.filename))


def ptb_metadata(root: Path) -> Path:
    matches = sorted(root.rglob(PTB_METADATA))
    if len(matches) != 1:
        raise ValueError(f"Expected one {PTB_METADATA} in {root}, found {len(matches)}")
    return matches[0]


def ptb_repeats(fp: TextIO) -> list[dict[str, str]]:
    rows = [row for row in csv.DictReader(fp) if row["patient_id"].strip()]
    ids = {}
    for row in rows:
        ids.setdefault(row["patient_id"], set()).add(row["ecg_id"])
    return [row for row in rows if len(ids[row["patient_id"]]) >= 2]


def ptb_rows(metadata: Path) -> list[dict[str, str]]:
    with metadata.open(newline="", encoding="utf-8-sig") as fp:
        return ptb_repeats(fp)


def extract_ptb_subset(path: Path, target: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        metadata_entries = [item for item in archive.infolist()
                            if Path(item.filename).name == PTB_METADATA
                            and "__MACOSX" not in Path(item.filename).parts]
        if len(metadata_entries) != 1:
            raise ValueError(f"Expected one {PTB_METADATA} in {path}, found {len(metadata_entries)}")
        metadata = metadata_entries[0]
        with archive.open(metadata) as src, TextIOWrapper(src, encoding="utf-8-sig", newline="") as fp:
            rows = ptb_repeats(fp)
        if not rows:
            raise ValueError(f"No repeated participants in {metadata.filename}")
        prefix = Path(metadata.filename).parent
        files = [Path(row["filename_lr"] + suffix) for row in rows for suffix in (".hea", ".dat")]
        if any(file.parts[:1] != ("records100",) or ".." in file.parts for file in files):
            raise ValueError(f"Unexpected filename_lr in {metadata.filename}")
        # Resolve all entries first, so an incomplete archive is rejected before writing.
        entries = [(archive.getinfo((prefix / file).as_posix()), file) for file in files]
        copy_zip_member(archive, metadata, target, Path(PTB_METADATA))
        for item, relative in entries:
            copy_zip_member(archive, item, target, relative)


def import_archive(dataset: str, archive: Path, root: Path) -> None:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    target = root / dataset
    target.mkdir(parents=True, exist_ok=True)
    if dataset == "heartprint":
        extract_heartprint(archive, target)
    elif dataset == "ptb-xl":
        extract_ptb_subset(archive, target)
    elif dataset == "ecg-id":
        extract_zip(archive, target)
    print(f"{dataset}: {len(list_records(dataset, target))} records indexed", flush=True)


def download_records(base: str, target: Path, names: list[str], workers: int) -> None:
    def fetch_record(name):
        for suffix in (".hea", ".dat"):
            fetch(base + name + suffix, target / (name + suffix))

    print(f"Fetching {len(names)} original recordings into {target}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(fetch_record, names), 1):
            if index % 250 == 0:
                print(f"{index}/{len(names)} records", flush=True)


def extract_heartprint(archive: Path, target: Path) -> None:
    if zipfile.is_zipfile(archive):
        extract_zip(archive, target)
    elif archive.suffix.lower() == ".rar":
        executable = shutil.which("bsdtar")
        if executable is None:
            raise OSError(f"Install bsdtar or manually extract {archive} into {target}")
        subprocess.run([executable, "-xf", str(archive), "-C", str(target)], check=True)
    else:
        raise ValueError(f"Expected a Heartprint RAR or ZIP archive: {archive}")


def download(dataset: str, root: Path, workers: int = 8) -> None:
    target = root / dataset
    target.mkdir(parents=True, exist_ok=True)
    if dataset == "ecg-id":
        manifest = target / "RECORDS"
        fetch(ECG_ID_BASE + "RECORDS", manifest)
        names = [line.strip() for line in manifest.read_text().splitlines() if line.startswith("Person_")]
        download_records(ECG_ID_BASE, target, names, workers)
    elif dataset == "heartprint":
        archives = sorted(path for path in target.iterdir() if path.suffix.lower() in {".zip", ".rar"})
        if not archives:
            metadata = target / "figshare-v3.json"
            fetch(HEARTPRINT_API, metadata)
            for item in json.loads(metadata.read_text())["files"]:
                path = target / Path(item["name"]).name
                if path.suffix.lower() in {".zip", ".rar"}:
                    fetch(item["downloadUrl"], path)
                    archives.append(path)
        for archive in archives:
            extract_heartprint(archive, target)
    elif dataset == "ptb-xl":
        fetch(PTB_BASE + "ptbxl_database.csv", target / "ptbxl_database.csv")
        download_records(PTB_BASE, target, [row["filename_lr"] for row in ptb_rows(ptb_metadata(target))], workers)
    print(f"{dataset}: {len(list_records(dataset, target))} records indexed", flush=True)


def ecg_id_date(path: Path) -> str:
    for line in path.read_text(encoding="ascii").splitlines():
        if "date" not in line.lower() and line.startswith("#"):
            continue
        match = re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b", line)
        if match:
            normalized = re.sub(r"[./-]", "/", match[0])
            return datetime.strptime(normalized, "%d/%m/%Y").date().isoformat()
    return ""


def heartprint_date(path: Path) -> str:
    token = path.name.split("_", 1)[0]
    # The release mixes YYYYMMDD and DDMMYYYY in the filename. Session
    # membership always comes from the folder, independently of this field.
    for pattern in ("%Y%m%d", "%d%m%Y"):
        try:
            date = datetime.strptime(token, pattern).date()
        except ValueError:
            continue
        if date.strftime(pattern) == token:
            return date.isoformat()
    return ""


def list_records(dataset: str, root: Path) -> list[Record]:
    records = []
    if dataset == "ecg-id":
        for path in sorted(root.rglob("Person_*/rec_*.hea")):
            if ".old" in path.parts or not path.with_suffix(".dat").is_file():
                continue
            participant = f"{dataset}:{path.parent.name}"
            date = ecg_id_date(path)
            records.append(Record(dataset, participant, f"{participant}/{path.stem}", date, path, date))
    elif dataset == "heartprint":
        for path in sorted(root.rglob("*.txt")):
            relative = path.relative_to(root)
            participant_number = path.parent.name
            if not re.fullmatch(r"\d{3}", participant_number) or not path.name.endswith("_ECG.txt"):
                continue
            session = next((label for part in relative.parts[:-2]
                            if (label := part.upper().replace("SESSION-", "S")) in HEARTPRINT_SESSIONS), None)
            if session is None:
                continue
            participant = f"{dataset}:{participant_number}"
            date = heartprint_date(path)
            records.append(Record(dataset, participant, f"{participant}/{session}/{path.stem}",
                                  HEARTPRINT_SESSIONS[session], path, date))
    elif dataset == "ptb-xl":
        metadata = ptb_metadata(root)
        for row in ptb_rows(metadata):
            path = metadata.parent / (row["filename_lr"] + ".hea")
            if not path.is_file() or not path.with_suffix(".dat").is_file():
                continue
            participant = f'{dataset}:{row["patient_id"]}'
            date = row["recording_date"].split(" ")[0]
            records.append(Record(dataset, participant, f'{dataset}:{row["ecg_id"]}', date, path, date))
    if not records:
        raise ValueError(f"No {dataset} records in {root}. Download or extract the source archive first.")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError(f"Duplicate record IDs in {root}. Keep one copy of the source files.")
    return records


def read_signal(record: Record) -> tuple[np.ndarray, int]:
    if record.dataset == "heartprint":
        # The vendor appends metadata after exactly 3747 one-sample lines.
        signal = np.loadtxt(record.path, max_rows=3747, dtype=np.float64)
        if signal.shape != (3747,):
            raise ValueError(f"Expected 3747 samples in {record.path}, got {signal.shape}")
        return signal, 250

    import wfdb

    stem = str(record.path.with_suffix(""))
    header = wfdb.rdheader(stem)
    names = [name.strip().casefold() for name in header.sig_name]
    lead = {"ecg-id": "ecg i", "ptb-xl": "i"}[record.dataset]
    signal, fields = wfdb.rdsamp(stem, channels=[names.index(lead)])
    return signal[:, 0], int(fields["fs"])
