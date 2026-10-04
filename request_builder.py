from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from schemas import InputRecord, PromptInstance, RequestItem, RequestMetadataRow


def build_requests(
    experiment_id: str,
    dataset_id: str,
    model: str,
    input_records: list[InputRecord],
    prompt_instances: list[PromptInstance],
    trials: int,
    temperature: float = 1.0,
    base_seed: int | None = None,
) -> list[RequestItem]:
    requests: list[RequestItem] = []

    for record in input_records:
        for prompt_instance in prompt_instances:
            for trial_index in range(trials):
                request_id = (
                    f"{experiment_id}__{record.file_id}__"
                    f"{prompt_instance.prompt_instance_id}__trial-{trial_index}"
                )
                requests.append(
                    RequestItem(
                        request_id=request_id,
                        experiment_id=experiment_id,
                        dataset_id=dataset_id,
                        file_id=record.file_id,
                        file_uri=record.file_uri,
                        source_type=record.source_type,
                        transcription=record.transcription,
                        model=model,
                        prompt_id=prompt_instance.prompt_id,
                        prompt_instance_id=prompt_instance.prompt_instance_id,
                        trial_index=trial_index,
                        variables=prompt_instance.variables,
                        prompt_text=prompt_instance.prompt_text,
                        response_schema=prompt_instance.response_schema,
                        temperature=temperature,
                        seed=None if base_seed is None else base_seed + trial_index,
                    )
                )

    return requests


def save_requests_jsonl(requests: list[RequestItem], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for request in requests:
            payload = build_gemini_batch_request_json(request)
            handle.write(json.dumps(payload) + "\n")


def save_request_metadata_csv(requests: list[RequestItem], path: Path) -> None:
    rows = flatten_request_metadata(requests)
    fieldnames = list(RequestMetadataRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def flatten_request_metadata(requests: list[RequestItem]) -> list[RequestMetadataRow]:
    rows: list[RequestMetadataRow] = []
    for request in requests:
        if request.variables:
            for variable_name, variable_value in request.variables.items():
                rows.append(
                    RequestMetadataRow(
                        request_id=request.request_id,
                        experiment_id=request.experiment_id,
                        dataset_id=request.dataset_id,
                        file_id=request.file_id,
                        file_uri=request.file_uri,
                        model=request.model,
                        prompt_id=request.prompt_id,
                        prompt_instance_id=request.prompt_instance_id,
                        trial_index=request.trial_index,
                        variable_name=variable_name,
                        variable_value=variable_value,
                        **run_settings(request),
                    )
                )
        else:
            rows.append(
                RequestMetadataRow(
                    request_id=request.request_id,
                    experiment_id=request.experiment_id,
                    dataset_id=request.dataset_id,
                    file_id=request.file_id,
                    file_uri=request.file_uri,
                    model=request.model,
                    prompt_id=request.prompt_id,
                    prompt_instance_id=request.prompt_instance_id,
                    trial_index=request.trial_index,
                    variable_name="",
                    variable_value="",
                    **run_settings(request),
                )
            )
    return rows


def run_settings(request: RequestItem) -> dict[str, str]:
    return {
        "temperature": str(request.temperature),
        "seed": "" if request.seed is None else str(request.seed),
        "prompt_text_hash": hashlib.sha256(request.prompt_text.encode("utf-8")).hexdigest(),
    }


def build_gemini_batch_request_json(request: RequestItem) -> dict[str, Any]:
    response_schema = convert_response_schema(request.response_schema)
    generation_config: dict[str, Any] = {
        "responseMimeType": "application/json",
        "responseSchema": response_schema,
        "temperature": request.temperature,
    }
    if request.seed is not None:
        generation_config["seed"] = request.seed
    return {
        "custom_id": request.request_id,
        "method": "generateContent",
        "request": {
            "model": request.model,
            "contents": build_contents(request),
            "generationConfig": generation_config,
        },
    }


def build_contents(request: RequestItem) -> list[dict[str, Any]]:
    if request.source_type == "audio":
        return [
            {
                "role": "user",
                "parts": [
                    {
                        "file_data": {
                            "mime_type": infer_mime_type(request.file_uri),
                            "file_uri": request.file_uri,
                        }
                    },
                    {"text": request.prompt_text},
                ],
            }
        ]

    return [
        {
            "role": "user",
            "parts": [{"text": build_text_prompt(request.transcription, request.prompt_text)}],
        }
    ]


def build_text_prompt(transcription: str, prompt_text: str) -> str:
    return f"{prompt_text}\n\n{transcription}"


def convert_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for key, value in schema.items():
        required.append(key)
        properties[key] = {"type": normalize_schema_type(str(value))}

    return {
        "type": "OBJECT",
        "properties": properties,
        "required": required,
    }


def normalize_schema_type(type_name: str) -> str:
    mapping = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
    }
    normalized = mapping.get(type_name.lower())
    if not normalized:
        raise ValueError(f"Unsupported response schema type: {type_name}")
    return normalized


def infer_mime_type(file_uri: str) -> str:
    lowered = file_uri.lower()
    if lowered.endswith(".wav"):
        return "audio/wav"
    if lowered.endswith(".mp3"):
        return "audio/mpeg"
    if lowered.endswith(".flac"):
        return "audio/flac"
    if lowered.endswith(".m4a"):
        return "audio/mp4"
    if lowered.endswith(".aac"):
        return "audio/aac"
    if lowered.endswith(".ogg"):
        return "audio/ogg"
    return "application/octet-stream"
