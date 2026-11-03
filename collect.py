from __future__ import annotations

import json
from pathlib import Path

from parser import NOT_RECORDED, load_request_metadata_index, parse_output_jsonl_text, save_parsed_results_csv
from registry import mark_job_collected, pending_collection_records, read_registry, update_job_status
from schemas import BatchJobRecord
from vertex_batch import VertexBatchClient

RAW_OUTPUT_FILENAME = "raw_output.jsonl"


def run_collect_results(
    registry_path: Path,
    local_artifacts_root: Path,
    project: str | None,
    location: str,
) -> list[str]:
    pending_records = pending_collection_records(registry_path)
    clients: dict[str, object] = {}

    def client_for(backend: str):
        if backend not in clients:
            if backend == "vertex":
                if not project:
                    raise ValueError("--project (or GOOGLE_CLOUD_PROJECT) is needed to collect Vertex jobs.")
                clients[backend] = VertexBatchClient(project=project, location=location)
            else:
                from local_backend import LocalGpuClient

                clients[backend] = LocalGpuClient()
        return clients[backend]

    collected_experiments: list[str] = []

    for record in pending_records:
        client = client_for(record.backend or "vertex")
        if record.backend in ("", "vertex"):
            status = client.get_job_status(record.job_id)
        else:
            status = client.get_job_status(record.job_id, record.output_uri_prefix)
        update_job_status(registry_path, record.job_id, status)

        if not client.job_is_finished(status):
            continue
        if not client.job_succeeded(status):
            mark_job_collected(registry_path, record.job_id, status)
            continue

        batch_job = client.get_batch_job(record.job_id)
        output_text = client.download_job_output_text(
            batch_job,
            output_uri_prefix=record.output_uri_prefix,
        )
        if not output_text.strip():
            continue

        results_dir = local_artifacts_root / record.experiment_id / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / RAW_OUTPUT_FILENAME).write_text(output_text, encoding="utf-8")

        write_parsed_results(record, local_artifacts_root, output_text)
        mark_job_collected(registry_path, record.job_id, status)
        collected_experiments.append(record.experiment_id)

    return collected_experiments


def run_rebuild_results(registry_path: Path, local_artifacts_root: Path) -> list[str]:
    """Rebuild the parsed tables of collected jobs from their saved raw output.

    Reads results/raw_output.jsonl and never writes it, so model output stays
    exactly as returned. Use after the results table format changes.
    """
    rebuilt: list[str] = []
    for record in read_registry(registry_path):
        raw_output_path = local_artifacts_root / record.experiment_id / "results" / RAW_OUTPUT_FILENAME
        if record.result_collected != "true" or not raw_output_path.is_file():
            continue
        write_parsed_results(record, local_artifacts_root, raw_output_path.read_text(encoding="utf-8"))
        rebuilt.append(record.experiment_id)
    return rebuilt


def write_parsed_results(record: BatchJobRecord, local_artifacts_root: Path, output_text: str) -> None:
    experiment_dir = local_artifacts_root / record.experiment_id
    experiment_results_root = local_artifacts_root.parent / "experiment_results"
    experiment_results_root.mkdir(parents=True, exist_ok=True)
    parsed_rows = parse_output_jsonl_text(
        output_text,
        job_id=record.job_id,
        request_metadata_index=load_request_metadata_index(experiment_dir / "request_metadata.csv"),
        submitted_at=record.submitted_at,
        run_columns=run_provenance(record.backend or "vertex", experiment_dir),
        schema_fields=load_schema_fields(experiment_dir / "prompt_instances.json"),
    )
    save_parsed_results_csv(parsed_rows, experiment_dir / "results" / "parsed_long.csv")
    save_parsed_results_csv(parsed_rows, experiment_results_root / f"{record.experiment_id}.csv")


def load_schema_fields(path: Path) -> dict[str, list[str]] | None:
    """response_schema field names per prompt_instance_id, from the saved prompt instances."""
    if not path.is_file():
        return None
    instances = json.loads(path.read_text(encoding="utf-8"))
    return {instance["prompt_instance_id"]: list(instance.get("response_schema") or {}) for instance in instances}


def run_provenance(backend: str, experiment_dir: Path) -> dict[str, str]:
    """Run-level columns for the results table, read from the files each backend writes."""
    spec = read_json_if_exists(experiment_dir / "experiment_spec.json")
    info = read_json_if_exists(experiment_dir / "results" / "run_info.json")
    version = info.get("code_version") or spec.get("code_version") or {}
    commit = version.get("commit") or NOT_RECORDED
    if version.get("uncommitted_changes"):
        commit += "+uncommitted"
    if backend == "vertex":
        return {
            "backend": backend,
            "model_served": "",  # read per line from Gemini's modelVersion
            "settings_source": "requested; unsent settings are Vertex defaults",
            "code_commit": commit,
            "hardware": "google_hosted",
        }
    gpu = info.get("gpu")
    return {
        "backend": backend,
        "model_served": info.get("model") or spec.get("model") or "",
        "settings_source": "effective; see results/run_info.json",
        "code_commit": commit,
        "hardware": "cpu" if gpu == "none" else gpu or NOT_RECORDED,
    }


def read_json_if_exists(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
