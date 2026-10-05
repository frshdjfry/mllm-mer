from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

from schemas import BatchJobRecord

REGISTRY_FIELDS = list(BatchJobRecord.__dataclass_fields__.keys())


def ensure_registry_exists(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()


def append_registry_record(path: Path, record: BatchJobRecord) -> None:
    ensure_registry_exists(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != REGISTRY_FIELDS:
        # Registry from before new columns were added: upgrade the header once
        # so the new row does not land under misaligned columns.
        write_registry(path, read_registry(path) + [record])
        return
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writerow(record.__dict__)


def read_registry(path: Path) -> list[BatchJobRecord]:
    ensure_registry_exists(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [BatchJobRecord(**row) for row in reader]


def write_registry(path: Path, records: list[BatchJobRecord]) -> None:
    # Write to a temporary file and swap it in, so an interrupted write never
    # leaves a truncated registry.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
            writer.writeheader()
            for record in records:
                writer.writerow(record.__dict__)
        if path.exists():
            os.chmod(tmp_name, path.stat().st_mode & 0o777)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def mark_job_collected(path: Path, job_id: str, status: str) -> None:
    records = read_registry(path)
    for record in records:
        if record.job_id == job_id:
            record.result_collected = "true"
            record.status = status
    write_registry(path, records)


def update_job_status(path: Path, job_id: str, status: str) -> None:
    records = read_registry(path)
    for record in records:
        if record.job_id == job_id:
            record.status = status
    write_registry(path, records)


def pending_collection_records(path: Path) -> list[BatchJobRecord]:
    return [record for record in read_registry(path) if record.result_collected.lower() != "true"]
