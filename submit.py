from __future__ import annotations

import json
import time
from pathlib import Path

from datasets import (
    build_input_manifest,
    dataset_id_from_csv_path,
    dataset_id_from_gcs_uri,
    list_audio_records,
    load_text_records,
    save_input_manifest,
)
from prompts import expand_prompt_instances, load_prompt_spec, save_prompt_instances
from registry import append_registry_record
from request_builder import build_requests, save_request_metadata_csv, save_requests_jsonl
from schemas import BatchJobRecord, ExperimentSpec, InputRecord, utc_now_iso
from vertex_batch import VertexBatchClient


def run_submit_experiment(
    dataset_uri: str,
    prompt_spec_path: str,
    model: str,
    trials: int,
    output_uri_prefix: str,
    registry_path: Path,
    local_artifacts_root: Path,
    project: str,
    location: str,
    temperature: float = 1.0,
    base_seed: int | None = None,
) -> str:
    input_records = list_audio_records(dataset_uri)
    dataset_id = dataset_id_from_gcs_uri(dataset_uri)
    return run_submit_common(
        input_source=dataset_uri,
        dataset_id=dataset_id,
        input_records=input_records,
        prompt_spec_path=prompt_spec_path,
        model=model,
        trials=trials,
        output_uri_prefix=output_uri_prefix,
        registry_path=registry_path,
        local_artifacts_root=local_artifacts_root,
        project=project,
        location=location,
        temperature=temperature,
        base_seed=base_seed,
    )


def run_submit_text_experiment(
    input_csv: str,
    prompt_spec_path: str,
    model: str,
    trials: int,
    output_uri_prefix: str,
    registry_path: Path,
    local_artifacts_root: Path,
    project: str,
    location: str,
    temperature: float = 1.0,
    base_seed: int | None = None,
) -> str:
    input_records = load_text_records(input_csv)
    dataset_id = dataset_id_from_csv_path(input_csv)
    return run_submit_common(
        input_source=str(Path(input_csv).resolve()),
        dataset_id=dataset_id,
        input_records=input_records,
        prompt_spec_path=prompt_spec_path,
        model=model,
        trials=trials,
        output_uri_prefix=output_uri_prefix,
        registry_path=registry_path,
        local_artifacts_root=local_artifacts_root,
        project=project,
        location=location,
        temperature=temperature,
        base_seed=base_seed,
    )


def run_submit_common(
    input_source: str,
    dataset_id: str,
    input_records: list[InputRecord],
    prompt_spec_path: str,
    model: str,
    trials: int,
    output_uri_prefix: str,
    registry_path: Path,
    local_artifacts_root: Path,
    project: str,
    location: str,
    temperature: float = 1.0,
    base_seed: int | None = None,
) -> str:
    if trials < 1:
        raise ValueError("Trials must be at least 1.")
    if not input_records:
        raise ValueError(f"No input records found for {input_source}")

    client = VertexBatchClient(project=project, location=location)
    prompt_spec = load_prompt_spec(prompt_spec_path)
    prompt_instances = expand_prompt_instances(prompt_spec)
    experiment_id = build_experiment_id(dataset_id=dataset_id, prompt_id=prompt_spec.prompt_id)
    experiment_dir = local_artifacts_root / experiment_id
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_input_manifest(input_source=input_source, records=input_records, dataset_id=dataset_id)
    save_input_manifest(manifest, experiment_dir / "input_manifest.json")

    experiment_spec = ExperimentSpec(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        input_source=input_source,
        prompt_spec_path=str(Path(prompt_spec_path).resolve()),
        model=model,
        trials=trials,
        output_uri_prefix=output_uri_prefix,
        created_at=utc_now_iso(),
        prompt_instances=prompt_instances,
        temperature=temperature,
        base_seed=base_seed,
    )
    save_json(experiment_spec.to_dict(), experiment_dir / "experiment_spec.json")
    save_prompt_instances(prompt_instances, experiment_dir / "prompt_instances.json")

    requests = build_requests(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        model=model,
        input_records=input_records,
        prompt_instances=prompt_instances,
        trials=trials,
        temperature=temperature,
        base_seed=base_seed,
    )
    local_requests_path = experiment_dir / "requests.jsonl"
    save_requests_jsonl(requests, local_requests_path)
    save_request_metadata_csv(requests, experiment_dir / "request_metadata.csv")

    remote_requests_uri = join_gcs_path(output_uri_prefix, experiment_id, "requests.jsonl")
    client.upload_file(local_requests_path, remote_requests_uri)

    job = client.submit_batch_job(
        display_name=experiment_id,
        model=model,
        input_jsonl_uri=remote_requests_uri,
        output_uri_prefix=join_gcs_path(output_uri_prefix, experiment_id, "batch_output"),
    )

    append_registry_record(
        registry_path,
        BatchJobRecord(
            experiment_id=experiment_id,
            job_id=job.job_id,
            submitted_at=job.submitted_at,
            status=job.status,
            input_jsonl_uri=job.input_jsonl_uri,
            output_uri_prefix=job.output_uri_prefix,
            result_collected="false",
            dataset_id=dataset_id,
            model=model,
            prompt_id=prompt_spec.prompt_id,
            trials=trials,
            temperature=str(temperature),
            base_seed="" if base_seed is None else str(base_seed),
        ),
    )

    return experiment_id


def build_experiment_id(dataset_id: str, prompt_id: str) -> str:
    timestamp = int(time.time())
    return f"{dataset_id}__{prompt_id}__{timestamp}"


def join_gcs_path(*parts: str) -> str:
    clean_parts = [part.strip("/") for part in parts]
    head = clean_parts[0]
    tail = "/".join(clean_parts[1:])
    return f"{head}/{tail}" if tail else head


def save_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
