from __future__ import annotations

import csv
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
    # Rewrite rather than append, so registries created before new columns
    # were added get the current header instead of misaligned rows.
    write_registry(path, read_registry(path) + [record])


def read_registry(path: Path) -> list[BatchJobRecord]:
    ensure_registry_exists(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [BatchJobRecord(**row) for row in reader]


def write_registry(path: Path, records: list[BatchJobRecord]) -> None:
    ensure_registry_exists(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


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
