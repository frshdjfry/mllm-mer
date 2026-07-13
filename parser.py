from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from schemas import RequestMetadataRow, utc_now_iso


def parse_output_jsonl_text(
    output_text: str,
    job_id: str,
    request_metadata_index: dict[str, list[RequestMetadataRow]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    created_at = utc_now_iso()

    for line in output_text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        request_id = extract_request_id(payload)
        metadata_rows = request_metadata_index.get(request_id, [])
        response_json = extract_response_json(payload)
        raw_response_json = json.dumps(response_json, sort_keys=True)

        if isinstance(response_json, dict):
            rows.append(
                build_structured_result_row(
                    request_id=request_id,
                    metadata_rows=metadata_rows,
                    response_json=response_json,
                    raw_response_json=raw_response_json,
                    job_id=job_id,
                    created_at=created_at,
                )
            )
        else:
            rows.extend(
                build_result_rows(
                    request_id=request_id,
                    metadata_rows=metadata_rows,
                    response_key="_raw",
                    response_value=response_json,
                    raw_response_json=raw_response_json,
                    job_id=job_id,
                    created_at=created_at,
                )
            )

    return rows


def save_parsed_results_csv(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = collect_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    base_fields = [
        "experiment_id",
        "dataset_id",
        "file_id",
        "file_uri",
        "model",
        "prompt_id",
        "prompt_instance_id",
        "trial_index",
        "raw_response_json",
        "job_id",
        "created_at",
    ]
    dynamic_fields: list[str] = []
    seen = set(base_fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                dynamic_fields.append(key)
                seen.add(key)
    return base_fields + sorted(dynamic_fields)


def load_request_metadata_index(path: Path) -> dict[str, list[RequestMetadataRow]]:
    index: dict[str, list[RequestMetadataRow]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata_row = RequestMetadataRow(
                request_id=row["request_id"],
                experiment_id=row["experiment_id"],
                dataset_id=row["dataset_id"],
                file_id=row["file_id"],
                file_uri=row["file_uri"],
                model=row["model"],
                prompt_id=row["prompt_id"],
                prompt_instance_id=row["prompt_instance_id"],
                trial_index=int(row["trial_index"]),
                variable_name=row["variable_name"],
                variable_value=row["variable_value"],
            )
            index.setdefault(metadata_row.request_id, []).append(metadata_row)
    return index


def extract_request_id(payload: dict[str, Any]) -> str:
    request_id = payload.get("custom_id") or payload.get("request_id")
    if isinstance(request_id, str) and request_id:
        return request_id
    raise ValueError("Could not find custom_id in batch output line.")


def extract_response_json(payload: dict[str, Any]) -> Any:
    candidates = [
        payload.get("response"),
        payload.get("prediction"),
        payload.get("predictions"),
    ]

    for candidate in candidates:
        parsed = maybe_extract_json_from_candidate(candidate)
        if parsed is not None:
            return parsed

    raise ValueError("Could not find a parseable response payload in batch output line.")


def maybe_extract_json_from_candidate(candidate: Any) -> Any | None:
    if candidate is None:
        return None

    if isinstance(candidate, dict) and "candidates" in candidate:
        texts = extract_candidate_texts(candidate)
        for text in texts:
            parsed = parse_json_text(text)
            if parsed is not None:
                return parsed

    if isinstance(candidate, str):
        return parse_json_text(candidate)

    if isinstance(candidate, dict):
        return candidate

    if isinstance(candidate, list) and candidate:
        first = candidate[0]
        if isinstance(first, dict):
            parsed = maybe_extract_json_from_candidate(first)
            if parsed is not None:
                return parsed

    return None


def extract_candidate_texts(response: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    return texts


def parse_json_text(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def build_result_rows(
    request_id: str,
    metadata_rows: list[RequestMetadataRow],
    response_key: str,
    response_value: Any,
    raw_response_json: str,
    job_id: str,
    created_at: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    if not metadata_rows:
        rows.append(
            {
                "experiment_id": "",
                "dataset_id": "",
                "file_id": request_id,
                "file_uri": "",
                "model": "",
                "prompt_id": "",
                "prompt_instance_id": "",
                "trial_index": "0",
                "_response_key": response_key,
                "value": stringify_value(response_value),
                "raw_response_json": raw_response_json,
                "job_id": job_id,
                "created_at": created_at,
            }
        )
        return rows

    for metadata_row in metadata_rows:
        row = {
            "experiment_id": metadata_row.experiment_id,
            "dataset_id": metadata_row.dataset_id,
            "file_id": metadata_row.file_id,
            "file_uri": metadata_row.file_uri,
            "model": metadata_row.model,
            "prompt_id": metadata_row.prompt_id,
            "prompt_instance_id": metadata_row.prompt_instance_id,
            "trial_index": str(metadata_row.trial_index),
            "value": stringify_value(response_value),
            "raw_response_json": raw_response_json,
            "job_id": job_id,
            "created_at": created_at,
        }
        if metadata_row.variable_name:
            row[metadata_row.variable_name] = metadata_row.variable_value
        if response_key != "_raw":
            row["_response_key"] = response_key
        rows.append(row)
    return rows


def build_structured_result_row(
    request_id: str,
    metadata_rows: list[RequestMetadataRow],
    response_json: dict[str, Any],
    raw_response_json: str,
    job_id: str,
    created_at: str,
) -> dict[str, str]:
    if not metadata_rows:
        row = {
            "experiment_id": "",
            "dataset_id": "",
            "file_id": request_id,
            "file_uri": "",
            "model": "",
            "prompt_id": "",
            "prompt_instance_id": "",
            "trial_index": "0",
            "raw_response_json": raw_response_json,
            "job_id": job_id,
            "created_at": created_at,
        }
    else:
        first = metadata_rows[0]
        row = {
            "experiment_id": first.experiment_id,
            "dataset_id": first.dataset_id,
            "file_id": first.file_id,
            "file_uri": first.file_uri,
            "model": first.model,
            "prompt_id": first.prompt_id,
            "prompt_instance_id": first.prompt_instance_id,
            "trial_index": str(first.trial_index),
            "raw_response_json": raw_response_json,
            "job_id": job_id,
            "created_at": created_at,
        }

    for metadata_row in metadata_rows:
        if metadata_row.variable_name:
            row[metadata_row.variable_name] = metadata_row.variable_value

    for key, value in response_json.items():
        row[key] = stringify_value(value)
    return row


def stringify_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)
