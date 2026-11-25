from zipfile import ZipFile

import pytest

from ecg.sources import import_archive, list_records


def test_heartprint_import_reads_session_folders_and_two_date_formats(tmp_path):
    archive = tmp_path / "Heartprint.zip"
    with ZipFile(archive, "w") as z:
        for session, date in (
            ("Session-1", "28092013"), ("Session-2", "20131120"),
            ("Session-3L", "20140120"), ("Session-3R", "20140127"),
        ):
            z.writestr(f"Heartprint/{session}/005/{date}_2014_001_ECG.txt", "1\n")
        z.writestr("Heartprint/Session-1/005/notes.txt", "metadata\n")
    root = tmp_path / "raw"
    import_archive("heartprint", archive, root)
    records = list_records("heartprint", root / "heartprint")
    assert {r.participant for r in records} == {"heartprint:005"}
    assert [r.session for r in records] == ["01", "02", "03", "03"]
    assert [r.date for r in records] == ["2013-09-28", "2013-11-20", "2014-01-20", "2014-01-27"]
    assert len({r.record_id for r in records}) == 4


def test_ptb_import_selects_100_hz_records_of_repeat_participants(tmp_path):
    archive = tmp_path / "ptb-xl.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("release/ptbxl_database.csv", "patient_id,ecg_id,recording_date,filename_lr\n"
                   "9,1,1990-01-12 08:00:00,records100/00000/00001_lr\n"
                   "9,2,1991-02-12 08:00:00,records100/00000/00002_lr\n"
                   "10,3,1990-02-10 08:00:00,records100/00000/00003_lr\n")
        for record_id in range(1, 4):
            for suffix in ("dat", "hea"):
                z.writestr(f"release/records100/00000/{record_id:05}_lr.{suffix}", b"test")
                z.writestr(f"release/records500/00000/{record_id:05}_hr.{suffix}", b"test")
    root = tmp_path / "raw"
    import_archive("ptb-xl", archive, root)
    target = root / "ptb-xl"
    records = list_records("ptb-xl", target)
    assert [(r.participant, r.session) for r in records] == [("ptb-xl:9", "1990-01-12"),
                                                              ("ptb-xl:9", "1991-02-12")]
    assert not (target / "records500").exists()
    assert len(list((target / "records100").rglob("*.dat"))) == 2


def test_ptb_import_rejects_missing_archive_entries(tmp_path):
    archive = tmp_path / "partial.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("release/ptbxl_database.csv", "patient_id,ecg_id,recording_date,filename_lr\n"
                   "9,1,1990-01-12,records100/00000/00001_lr\n"
                   "9,2,1991-02-12,records100/00000/00002_lr\n")
    root = tmp_path / "raw"
    with pytest.raises(KeyError, match="00001_lr"):
        import_archive("ptb-xl", archive, root)
    assert not (root / "ptb-xl" / "ptbxl_database.csv").exists()
