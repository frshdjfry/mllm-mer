from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class InputRecord:
    source_type: str
    file_id: str
    file_uri: str = ""
    transcription: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class InputManifest:
    dataset_id: str
    input_source: str
    created_at: str
    records: list[InputRecord]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [asdict(item) for item in self.records]
        return payload


@dataclass
class PromptSpec:
    prompt_id: str
    kind: str
    prompt_text: str
    response_schema: dict[str, Any]
    variables: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class PromptInstance:
    prompt_id: str
    prompt_instance_id: str
    prompt_text: str
    response_schema: dict[str, Any]
    variables: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentSpec:
    experiment_id: str
    dataset_id: str
    input_source: str
    prompt_spec_path: str
    model: str
    trials: int
    output_uri_prefix: str
    created_at: str
    prompt_instances: list[PromptInstance]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prompt_instances"] = [item.to_dict() for item in self.prompt_instances]
        return payload


@dataclass
class RequestItem:
    request_id: str
    experiment_id: str
    dataset_id: str
    file_id: str
    file_uri: str
    source_type: str
    transcription: str
    model: str
    prompt_id: str
    prompt_instance_id: str
    trial_index: int
    variables: dict[str, str]
    prompt_text: str
    response_schema: dict[str, Any]


@dataclass
class RequestMetadataRow:
    request_id: str
    experiment_id: str
    dataset_id: str
    file_id: str
    file_uri: str
    model: str
    prompt_id: str
    prompt_instance_id: str
    trial_index: int
    variable_name: str
    variable_value: str


@dataclass
class BatchJobRecord:
    experiment_id: str
    job_id: str
    submitted_at: str
    status: str
    input_jsonl_uri: str
    output_uri_prefix: str
    result_collected: str
    dataset_id: str
    model: str
    prompt_id: str
    trials: int


@dataclass
class BatchJobSubmission:
    job_id: str
    status: str
    submitted_at: str
    input_jsonl_uri: str
    output_uri_prefix: str


@dataclass
class ParsedResultRow:
    experiment_id: str
    dataset_id: str
    file_id: str
    file_uri: str
    model: str
    prompt_id: str
    prompt_instance_id: str
    trial_index: int
    variable_name: str
    variable_value: str
    response_key: str
    response_value: str
    raw_response_json: str
    job_id: str
    created_at: str
