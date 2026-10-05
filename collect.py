from __future__ import annotations

from pathlib import Path

from parser import load_request_metadata_index, parse_output_jsonl_text, save_parsed_results_csv
from registry import mark_job_collected, pending_collection_records, update_job_status
from vertex_batch import VertexBatchClient


def run_collect_results(
    registry_path: Path,
    local_artifacts_root: Path,
    project: str,
    location: str,
) -> list[str]:
    client = VertexBatchClient(project=project, location=location)
    pending_records = pending_collection_records(registry_path)
    collected_experiments: list[str] = []
    experiment_results_root = local_artifacts_root.parent / "experiment_results"
    experiment_results_root.mkdir(parents=True, exist_ok=True)

    for record in pending_records:
        status = client.get_job_status(record.job_id)
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

        experiment_dir = local_artifacts_root / record.experiment_id
        results_dir = experiment_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        request_metadata_index = load_request_metadata_index(experiment_dir / "request_metadata.csv")

        raw_output_path = results_dir / "raw_output.jsonl"
        raw_output_path.write_text(output_text, encoding="utf-8")

        parsed_rows = parse_output_jsonl_text(
            output_text,
            job_id=record.job_id,
            request_metadata_index=request_metadata_index,
            submitted_at=record.submitted_at,
        )
        save_parsed_results_csv(parsed_rows, results_dir / "parsed_long.csv")
        save_parsed_results_csv(parsed_rows, experiment_results_root / f"{record.experiment_id}.csv")
        mark_job_collected(registry_path, record.job_id, status)
        collected_experiments.append(record.experiment_id)

    return collected_experiments
