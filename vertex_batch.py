from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import parse_gcs_uri
from schemas import BatchJobSubmission, utc_now_iso


@dataclass(slots=True)
class StubBatchState:
    name: str


@dataclass(slots=True)
class StubBatchDest:
    file_name: str


@dataclass(slots=True)
class StubBatchJob:
    name: str
    state: StubBatchState
    dest: StubBatchDest


class VertexBatchClient:
    def __init__(
        self,
        project: str,
        location: str,
        storage_client: Any | None = None,
        genai_client: Any | None = None,
    ) -> None:
        self.project = project
        self.location = location
        self.storage_client = storage_client or self._build_storage_client(project)
        self.genai_client = genai_client or self._build_genai_client(project, location)

    def upload_file(self, local_path: Path, gcs_uri: str) -> str:
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path), content_type="application/json")
        return gcs_uri

    def download_text(self, gcs_uri: str) -> str:
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_text()

    def list_output_jsonl_uris(self, output_uri_prefix: str) -> list[str]:
        bucket_name, prefix = parse_gcs_uri(output_uri_prefix)
        blobs = self.storage_client.list_blobs(bucket_name, prefix=prefix)
        uris = [
            f"gs://{bucket_name}/{blob.name}"
            for blob in blobs
            if blob.name.endswith(".jsonl") and not blob.name.endswith("requests.jsonl")
        ]
        uris.sort()
        return uris

    def submit_batch_job(
        self,
        display_name: str,
        model: str,
        input_jsonl_uri: str,
        output_uri_prefix: str,
    ) -> BatchJobSubmission:
        if self._stub_enabled():
            fake_job_id = f"stub-job-{uuid.uuid4().hex[:12]}"
            return BatchJobSubmission(
                job_id=fake_job_id,
                status="JOB_STATE_SUCCEEDED",
                submitted_at=utc_now_iso(),
                input_jsonl_uri=input_jsonl_uri,
                output_uri_prefix=output_uri_prefix,
            )

        batch_job = self.genai_client.batches.create(
            model=model,
            src=input_jsonl_uri,
            config={"display_name": display_name},
        )
        return BatchJobSubmission(
            job_id=batch_job.name,
            status=self._state_name(batch_job),
            submitted_at=utc_now_iso(),
            input_jsonl_uri=input_jsonl_uri,
            output_uri_prefix=output_uri_prefix,
        )

    def get_job_status(self, job_id: str) -> str:
        if job_id.startswith("stub-job-"):
            return "JOB_STATE_SUCCEEDED"
        batch_job = self.genai_client.batches.get(name=job_id)
        return self._state_name(batch_job)

    def get_batch_job(self, job_id: str) -> Any:
        if job_id.startswith("stub-job-"):
            return StubBatchJob(
                name=job_id,
                state=StubBatchState(name="JOB_STATE_SUCCEEDED"),
                dest=StubBatchDest(file_name=f"{job_id}.jsonl"),
            )
        return self.genai_client.batches.get(name=job_id)

    def download_job_output_text(self, batch_job: Any, output_uri_prefix: str | None = None) -> str:
        if isinstance(batch_job, StubBatchJob):
            file_name = batch_job.dest.file_name
            if not file_name:
                file_name = f"{batch_job.name}.jsonl"
            return ""

        file_name = self._extract_batch_output_file_name(batch_job)
        if file_name:
            content = self.genai_client.files.download(file=file_name)
            if isinstance(content, bytes):
                return content.decode("utf-8")
            if isinstance(content, bytearray):
                return bytes(content).decode("utf-8")
            if isinstance(content, str):
                return content
            raise TypeError(f"Unexpected batch output content type: {type(content)!r}")

        batch_output_gcs_uri = self._extract_batch_output_gcs_uri(batch_job)
        if batch_output_gcs_uri:
            output_uris = self.list_output_jsonl_uris(batch_output_gcs_uri)
            if output_uris:
                raw_texts = [self.download_text(uri) for uri in output_uris]
                return "\n".join(text.rstrip("\n") for text in raw_texts if text.strip()) + "\n"

        if output_uri_prefix:
            output_uris = self.list_output_jsonl_uris(output_uri_prefix)
            if output_uris:
                raw_texts = [self.download_text(uri) for uri in output_uris]
                return "\n".join(text.rstrip("\n") for text in raw_texts if text.strip()) + "\n"

        raise ValueError("Batch job output is missing both file_name and downloadable GCS outputs.")

    def job_is_finished(self, status: str) -> bool:
        return status in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}

    def job_succeeded(self, status: str) -> bool:
        return status == "JOB_STATE_SUCCEEDED"

    def _build_storage_client(self, project: str) -> Any:
        from google.cloud import storage

        return storage.Client(project=project)

    def _build_genai_client(self, project: str, location: str) -> Any:
        from google import genai

        return genai.Client(vertexai=True, project=project, location=location)

    def _state_name(self, batch_job: Any) -> str:
        state = getattr(batch_job, "state", None)
        if state is None:
            return "JOB_STATE_UNKNOWN"
        name = getattr(state, "name", None)
        if isinstance(name, str) and name:
            return name
        if isinstance(state, str):
            return state
        return "JOB_STATE_UNKNOWN"

    def _stub_enabled(self) -> bool:
        return os.environ.get("GEMINI_BATCH_STUB_SUBMIT", "").lower() == "true"

    def _extract_batch_output_file_name(self, batch_job: Any) -> str | None:
        dest = self._get_value(batch_job, "dest")
        if dest is not None:
            file_name = self._get_value(dest, "file_name")
            if isinstance(file_name, str) and file_name:
                return file_name

        output = self._get_value(batch_job, "output")
        if output is not None:
            file_name = self._get_value(output, "file_name")
            if isinstance(file_name, str) and file_name:
                return file_name

        return None

    def _extract_batch_output_gcs_uri(self, batch_job: Any) -> str | None:
        dest = self._get_value(batch_job, "dest")
        if dest is not None:
            gcs_uri = self._get_value(dest, "gcs_uri")
            if isinstance(gcs_uri, str) and gcs_uri:
                return gcs_uri

        output = self._get_value(batch_job, "output")
        if output is not None:
            gcs_uri = self._get_value(output, "gcs_uri")
            if isinstance(gcs_uri, str) and gcs_uri:
                return gcs_uri

        return None

    def _get_value(self, value: Any, key: str) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)
