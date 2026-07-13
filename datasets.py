from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from typing import Iterable
from urllib.parse import urlparse

from schemas import InputManifest, InputRecord, utc_now_iso

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Expected a gs:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def dataset_id_from_gcs_uri(dataset_uri: str) -> str:
    bucket_name, _ = parse_gcs_uri(dataset_uri)
    return bucket_name


def dataset_id_from_csv_path(csv_path: str | Path) -> str:
    return Path(csv_path).stem


def file_id_from_uri(file_uri: str) -> str:
    _, blob_path = parse_gcs_uri(file_uri)
    return Path(blob_path).stem


def is_audio_blob(blob_name: str) -> bool:
    return Path(blob_name).suffix.lower() in _AUDIO_EXTENSIONS


def list_audio_records(dataset_uri: str, storage_client: Any | None = None) -> list[InputRecord]:
    if storage_client is None:
        from google.cloud import storage

        storage_client = storage.Client()
    bucket_name, prefix = parse_gcs_uri(dataset_uri)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    records: list[InputRecord] = []

    for blob in blobs:
        if blob.name.endswith("/") or not is_audio_blob(blob.name):
            continue
        uri = f"gs://{bucket_name}/{blob.name}"
        records.append(
            InputRecord(
                source_type="audio",
                file_id=file_id_from_uri(uri),
                file_uri=uri,
                attributes={"bucket": bucket_name, "blob_path": blob.name},
            )
        )

    records.sort(key=lambda item: item.file_uri)
    return records


def load_text_records(csv_path: str | Path) -> list[InputRecord]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "file_id" not in fieldnames or "transcription" not in fieldnames:
            raise ValueError("Input CSV must contain file_id and transcription columns.")

        records: list[InputRecord] = []
        for row in reader:
            file_id = str(row.get("file_id", "")).strip()
            transcription = str(row.get("transcription", "")).strip()
            if not file_id:
                continue
            attributes = {
                key: str(value)
                for key, value in row.items()
                if key not in {"file_id", "transcription"} and value is not None
            }
            records.append(
                InputRecord(
                    source_type="text",
                    file_id=file_id,
                    transcription=transcription,
                    attributes=attributes,
                )
            )
    return records


def build_input_manifest(input_source: str, records: Iterable[InputRecord], dataset_id: str) -> InputManifest:
    return InputManifest(
        dataset_id=dataset_id,
        input_source=input_source,
        created_at=utc_now_iso(),
        records=list(records),
    )


def save_input_manifest(manifest: InputManifest, path: Path) -> None:
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
