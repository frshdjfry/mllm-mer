"""Qwen backend: run requests on Qwen Omni models through Alibaba Cloud Model Studio.

Reads a requests.jsonl written by `submit-experiment --backend qwen` (the same
request format as the Gemini batch backend), sends each request to the
OpenAI-compatible Chat Completions endpoint of Model Studio (DashScope), and
writes raw_output.jsonl in the same shape as a Gemini batch output, so
`collect-results` parses every backend the same way.

Audio is read from local files and sent inline as base64 (no upload to a
bucket). The API key and endpoint come from the environment or a `.env` file
in this folder (gitignored), never from code or the command line:

    DASHSCOPE_API_KEY=...
    QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

Model Studio has no batch API for these models in Singapore, so requests run
one at a time, in this process, when the experiment is submitted. A run that
stops part-way can be resumed: answers already in results/raw_output.jsonl.partial
are kept and not paid for again.

    python -m qwen_api run --requests REQ.jsonl --output OUT.jsonl   # resume a stopped run

The model comes from each request line (`submit-experiment --model`).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from provenance import code_version
from schemas import BatchJobSubmission, utc_now_iso

REPO_DIR = Path(__file__).resolve().parent
ENV_FILE = REPO_DIR / ".env"
OUTPUT_FILENAME = "raw_output.jsonl"
FAILURE_FILENAME = "run_failed.txt"
RUN_INFO_FILENAME = "run_info.json"
BACKEND = "qwen"
STOPPED = "JOB_STATE_STOPPED"  # stopped part-way; resumable
# Model Studio rejects inline (base64) audio whose encoded string is 10 MB or more.
MAX_BASE64_BYTES = 10 * 1024 * 1024
# Retries for rate limits and server errors (Qwen3.5-Omni-Plus allows 60 requests per minute).
MAX_ATTEMPTS = 5
# What is never sent, so the model's own default applies. Recorded with every experiment.
NOT_SENT = (
    "max_tokens, enable_thinking: Qwen defaults (not sent); response_format is json_object, so field names "
    "are only asked for in the prompt; see results/run_info.json"
)

# (chat-completions keyword arguments) -> one streamed response, as a dict (see collect_stream)
CallFn = Callable[[dict[str, Any]], dict[str, Any]]

_AUDIO_FORMATS = {".wav": "wav", ".mp3": "mp3", ".flac": "flac", ".m4a": "m4a", ".aac": "aac", ".ogg": "ogg"}


# --- settings ------------------------------------------------------------------


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """KEY=VALUE lines from a .env file (comments and blank lines skipped)."""
    path = path or ENV_FILE
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return values


def api_settings() -> tuple[str, str]:
    """API key and base URL: environment variables first, then .env."""
    file_values = load_env_file()
    key = os.environ.get("DASHSCOPE_API_KEY") or file_values.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get("QWEN_BASE_URL") or file_values.get("QWEN_BASE_URL", "")
    if not key:
        raise ValueError(f"DASHSCOPE_API_KEY is not set. Put it in {ENV_FILE} (gitignored), not in code.")
    if not base_url:
        raise ValueError(
            f"QWEN_BASE_URL is not set. Put it in {ENV_FILE}, e.g. "
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
    return key, base_url


# --- request translation -----------------------------------------------------


def audio_part(path: str) -> dict[str, Any]:
    """A local audio file as an inline input_audio part."""
    suffix = Path(path).suffix.lower()
    if suffix not in _AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format for Qwen: {path}")
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    if len(encoded) >= MAX_BASE64_BYTES:
        raise ValueError(
            f"{path} is too large to send inline ({len(encoded)} bytes as base64; Model Studio's limit is "
            f"{MAX_BASE64_BYTES}). Shorten or compress it."
        )
    return {"type": "input_audio", "input_audio": {"data": f"data:;base64,{encoded}", "format": _AUDIO_FORMATS[suffix]}}


def request_to_chat_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    """Turn one batch request line into Chat Completions arguments, keeping the order of its parts."""
    body = request["request"]
    content: list[dict[str, Any]] = []
    for message in body["contents"]:
        for part in message["parts"]:
            if "file_data" in part:
                uri = part["file_data"]["file_uri"]
                if "://" in uri:
                    raise ValueError(f"The qwen backend reads local audio files, not URIs: {uri}")
                content.append(audio_part(uri))
            elif "text" in part:
                content.append({"type": "text", "text": part["text"]})
    generation = body.get("generationConfig", {})
    kwargs: dict[str, Any] = {
        "model": body["model"],
        "messages": [{"role": "user", "content": content}],
        "modalities": ["text"],  # text answer only; no speech output
        "stream": True,  # Qwen Omni models only answer in streaming mode
        "stream_options": {"include_usage": True},
        "temperature": float(generation.get("temperature", 1.0)),
    }
    if generation.get("responseMimeType") == "application/json":
        # JSON mode: the answer is bare JSON, not wrapped in a ```json block. Unlike
        # Gemini's responseSchema it does not enforce field names: check schema_keys.
        # (Model Studio's docs list it as unsupported in Singapore; tested working
        # there on 2026-11-24 for qwen3.5-omni-plus and qwen3.8-omni-flash.)
        kwargs["response_format"] = {"type": "json_object"}
    if generation.get("topP") is not None:
        kwargs["top_p"] = generation["topP"]
    if generation.get("seed") is not None:
        kwargs["seed"] = generation["seed"]
    return kwargs


def sent_settings(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The arguments as sent, without the (large) audio data."""
    return {key: value for key, value in kwargs.items() if key != "messages"}


def output_line(custom_id: str, response: dict[str, Any]) -> dict[str, Any]:
    """Same shape as a Gemini batch output line; modelVersion becomes the model_served column."""
    return {
        "custom_id": custom_id,
        "response": {
            "candidates": [{"content": {"parts": [{"text": response["text"]}]}, "finishReason": response["finish_reason"]}],
            "modelVersion": response["model"],
            "usageMetadata": response["usage"],
        },
        "qwen": {
            "request_id": response["id"],
            "reasoning": response["reasoning"],
            "sent": response["sent"],
        },
    }


# --- calling the API -----------------------------------------------------------


def build_openai_call(api_key: str, base_url: str) -> CallFn:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def call(kwargs: dict[str, Any]) -> dict[str, Any]:
        return collect_stream(client.chat.completions.create(**kwargs))

    return call


def collect_stream(chunks) -> dict[str, Any]:
    """Join a streamed answer into one record: text, reasoning (if any), model, usage, finish reason."""
    text: list[str] = []
    reasoning: list[str] = []
    result: dict[str, Any] = {"id": "", "model": "", "finish_reason": "", "usage": {}}
    for chunk in chunks:
        result["id"] = result["id"] or getattr(chunk, "id", "") or ""
        result["model"] = getattr(chunk, "model", "") or result["model"]
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            result["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        for choice in getattr(chunk, "choices", None) or []:
            delta = choice.delta
            if getattr(delta, "content", None):
                text.append(delta.content)
            if getattr(delta, "reasoning_content", None):
                reasoning.append(delta.reasoning_content)
            if choice.finish_reason:
                result["finish_reason"] = choice.finish_reason
    result["text"] = "".join(text)
    result["reasoning"] = "".join(reasoning)
    return result


def is_retryable(error: BaseException) -> bool:
    status = getattr(error, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return type(error).__name__ in {"APIConnectionError", "APITimeoutError"}


def call_with_retries(call: CallFn, kwargs: dict[str, Any], sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call(kwargs)
        except Exception as error:  # noqa: BLE001 - re-raised unless retryable
            if attempt == MAX_ATTEMPTS or not is_retryable(error):
                raise
            sleep(min(60.0, 2.0 ** attempt))
    raise AssertionError("unreachable")


# --- the run -------------------------------------------------------------------


def read_requests(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(
    requests_path: Path,
    output_path: Path,
    call: CallFn | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run every request and write the output atomically. Returns the number of requests.

    Answers are appended to OUT.partial as they arrive; if the run stops, the
    next run skips the requests already answered there.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path = output_path.parent / FAILURE_FILENAME
    failure_path.unlink(missing_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    info: dict[str, Any] = {
        "backend": BACKEND,
        "started_at": utc_now_iso(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "code_version": code_version(),
        "not_sent": NOT_SENT,
    }
    try:
        requests = read_requests(requests_path)
        info["models_requested"] = sorted({request["request"]["model"] for request in requests})
        if call is None:
            import openai

            info["openai_sdk"] = openai.__version__
            api_key, base_url = api_settings()
            info["base_url"] = base_url
            call = build_openai_call(api_key, base_url)
        done = {json.loads(line)["custom_id"] for line in _read_lines(partial_path)}
        info["resumed_answers"] = len(done)
        served: set[str] = set()
        usage_totals: dict[str, int] = {}
        with partial_path.open("a", encoding="utf-8") as handle:
            for request in requests:
                if request["custom_id"] in done:
                    continue
                kwargs = request_to_chat_kwargs(request)
                response = call_with_retries(call, kwargs, sleep=sleep)
                response["sent"] = sent_settings(kwargs)
                handle.write(json.dumps(output_line(request["custom_id"], response)) + "\n")
                handle.flush()
                served.add(response["model"])
                for key, value in response["usage"].items():
                    if isinstance(value, int):
                        usage_totals[key] = usage_totals.get(key, 0) + value
        os.replace(partial_path, output_path)
        info.update({
            "finished_at": utc_now_iso(),
            "requests": len(requests),
            "models_served": sorted(served),
            "usage_totals_this_run": usage_totals,
        })
        return len(requests)
    except BaseException:
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        info["failed_at"] = utc_now_iso()
        raise
    finally:
        (output_path.parent / RUN_INFO_FILENAME).write_text(json.dumps(info, indent=2), encoding="utf-8")


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- launcher: same interface as VertexBatchClient -------------------------------


class QwenApiClient:
    """Runs the requests in this process at submit time. Job ids are `qwen:<output dir>`."""

    def __init__(self, call: CallFn | None = None) -> None:
        self.call = call

    def upload_file(self, local_path: Path, uri: str) -> str:
        return uri  # nothing to upload: audio is sent inline with each request

    def submit_batch_job(self, display_name: str, model: str, input_jsonl_uri: str, output_uri_prefix: str) -> BatchJobSubmission:
        output_dir = Path(output_uri_prefix)
        submitted_at = utc_now_iso()
        try:
            run(Path(input_jsonl_uri), output_dir / OUTPUT_FILENAME, call=self.call)
            status = "JOB_STATE_SUCCEEDED"
        except Exception as error:  # noqa: BLE001 - recorded in the registry and run_failed.txt
            print(
                f"Qwen run stopped: {error}\nDetails: {output_dir / FAILURE_FILENAME}\n"
                f"Answers so far are kept. Resume with:\n  python -m qwen_api run --requests {input_jsonl_uri} "
                f"--output {output_dir / OUTPUT_FILENAME}\nthen run collect-results.",
                file=sys.stderr,
            )
            status = STOPPED
        return BatchJobSubmission(
            job_id=f"{BACKEND}:{output_dir.resolve()}",
            status=status,
            submitted_at=submitted_at,
            input_jsonl_uri=str(input_jsonl_uri),
            output_uri_prefix=str(output_dir),
        )

    def get_job_status(self, job_id: str, output_uri_prefix: str = "") -> str:
        output_dir = Path(job_id.partition(":")[2])
        if (output_dir / OUTPUT_FILENAME).is_file():
            return "JOB_STATE_SUCCEEDED"
        # Runs are synchronous, so no output means it stopped. Not "finished": the
        # job stays pending in the registry until it is resumed and collected.
        return STOPPED

    def get_batch_job(self, job_id: str) -> str:
        return job_id

    def download_job_output_text(self, batch_job: Any, output_uri_prefix: str | None = None) -> str:
        return (Path(output_uri_prefix or "") / OUTPUT_FILENAME).read_text(encoding="utf-8")

    def job_is_finished(self, status: str) -> bool:
        return status == "JOB_STATE_SUCCEEDED"

    def job_succeeded(self, status: str) -> bool:
        return status == "JOB_STATE_SUCCEEDED"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen Omni requests through Model Studio.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run (or resume) a requests.jsonl and write raw_output.jsonl.")
    run_parser.add_argument("--requests", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    count = run(args.requests, args.output)
    print(f"Wrote {count} responses to {args.output}")


if __name__ == "__main__":
    main()
