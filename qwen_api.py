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

A finished run is never re-run (that would re-send and re-bill every request).
A request the API rejects (HTTP 400/422, e.g. content inspection) is logged in
results/rejected_requests.jsonl and the run goes on; it has no row in the results.

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
REJECTED_FILENAME = "rejected_requests.jsonl"
# HTTP statuses that reject one request (its content) rather than the whole run.
REJECTED_STATUSES = {400, 422}
# Stop if this many requests in a row are rejected before any is answered: likely a settings error.
MAX_REJECTED_BEFORE_ANY_ANSWER = 3
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


def check_audio(path: str) -> str:
    """Raise if the file can't be sent inline (remote, missing, unsupported or too large); return its format."""
    if "://" in path:
        raise ValueError(f"The qwen backend reads local audio files, not URIs: {path}")
    suffix = Path(path).suffix.lower()
    if suffix not in _AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format for Qwen: {path}")
    if not Path(path).is_file():
        raise ValueError(f"Audio file not found: {path}")
    encoded_size = 4 * ((Path(path).stat().st_size + 2) // 3)  # base64 length, without reading the file
    if encoded_size >= MAX_BASE64_BYTES:
        raise ValueError(
            f"{path} is too large to send inline ({encoded_size} bytes as base64; Model Studio's limit is "
            f"{MAX_BASE64_BYTES}). Shorten or compress it."
        )
    return _AUDIO_FORMATS[suffix]


def check_all_audio(requests: list[dict[str, Any]]) -> None:
    """Check every audio file before anything is sent, so one bad file can't stop a run half-way (and half-paid)."""
    problems = []
    for request in requests:
        for message in request["request"]["contents"]:
            for part in message["parts"]:
                if "file_data" in part:
                    try:
                        check_audio(part["file_data"]["file_uri"])
                    except ValueError as error:
                        problems.append(str(error))
    if problems:
        raise ValueError("Nothing was sent; fix these audio files first:\n  " + "\n  ".join(sorted(set(problems))))


def audio_part(path: str) -> dict[str, Any]:
    """A local audio file as an inline input_audio part."""
    audio_format = check_audio(path)
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": f"data:;base64,{encoded}", "format": audio_format}}


def request_to_chat_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    """Turn one batch request line into Chat Completions arguments, keeping the order of its parts."""
    body = request["request"]
    content: list[dict[str, Any]] = []
    for message in body["contents"]:
        for part in message["parts"]:
            if "file_data" in part:
                content.append(audio_part(part["file_data"]["file_uri"]))
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
            # Which code produced this answer (a resumed run can mix commits).
            "code_commit": response.get("code_commit", ""),
        },
    }


def commit_label(version: dict[str, Any]) -> str:
    """Commit id, with '+uncommitted' if the checkout had uncommitted changes (as in the results table)."""
    label = version.get("commit") or "not_recorded"
    return label + "+uncommitted" if version.get("uncommitted_changes") else label


def resume_command(requests_path: str | Path, output_path: str | Path) -> str:
    return f"python -m qwen_api run --requests {requests_path} --output {output_path}"


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
    next run skips the requests already answered there. Each attempt is kept
    in run_info.json ("attempts"), so a resumed run's provenance is complete.
    """
    if output_path.is_file():
        raise ValueError(
            f"{output_path} already exists: this run finished. Running it again would re-send and re-bill every "
            "request and replace the answers. To really start over, move that file away first."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path = output_path.parent / FAILURE_FILENAME
    failure_path.unlink(missing_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    rejected_path = output_path.parent / REJECTED_FILENAME
    info_path = output_path.parent / RUN_INFO_FILENAME
    earlier_attempts = read_attempts(info_path)
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
        done, dropped = answered_ids(partial_path)
        info["resumed_answers"] = len(done)
        if dropped:
            info["dropped_incomplete_last_line"] = True  # cut off mid-write; that request is asked again
        pending = [request for request in requests if request["custom_id"] not in done]
        check_all_audio(pending)
        commit = commit_label(info["code_version"])
        served: set[str] = set()
        usage_totals: dict[str, int] = {}
        answered = rejected = 0
        info["answered_this_attempt"] = 0
        with partial_path.open("a", encoding="utf-8") as handle:
            for request in pending:
                kwargs = request_to_chat_kwargs(request)
                try:
                    response = call_with_retries(call, kwargs, sleep=sleep)
                except Exception as error:  # noqa: BLE001 - only per-request rejections are kept going
                    if getattr(error, "status_code", None) not in REJECTED_STATUSES:
                        raise
                    rejected += 1
                    log_rejection(rejected_path, request["custom_id"], error)
                    if answered == 0 and rejected >= MAX_REJECTED_BEFORE_ANY_ANSWER:
                        raise RuntimeError(
                            f"The first {rejected} requests were all rejected; this looks like a settings problem, "
                            f"not one bad song. See {rejected_path}."
                        ) from error
                    continue
                response["sent"] = sent_settings(kwargs)
                response["code_commit"] = commit
                handle.write(json.dumps(output_line(request["custom_id"], response)) + "\n")
                handle.flush()
                answered += 1
                info["answered_this_attempt"] = answered
                served.add(response["model"])
                for key, value in response["usage"].items():
                    if isinstance(value, int):
                        usage_totals[key] = usage_totals.get(key, 0) + value
                info["models_served"] = sorted(served)
                info["usage_totals_this_run"] = usage_totals
        os.replace(partial_path, output_path)
        info.update({"finished_at": utc_now_iso(), "requests": len(requests), "rejected_this_attempt": rejected})
        if rejected:
            print(f"Warning: {rejected} requests were rejected by the API and have no answer; see {rejected_path}",
                  file=sys.stderr)
        return len(requests)
    except BaseException:
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        info["failed_at"] = utc_now_iso()
        raise
    finally:
        # The latest attempt at the top level, and every attempt (including this one) under "attempts".
        info_path.write_text(json.dumps({**info, "attempts": [*earlier_attempts, info]}, indent=2), encoding="utf-8")


def read_attempts(info_path: Path) -> list[dict[str, Any]]:
    """Earlier attempts recorded in run_info.json (a file from before "attempts" counts as one attempt)."""
    if not info_path.is_file():
        return []
    previous = json.loads(info_path.read_text(encoding="utf-8"))
    if "attempts" in previous:
        return previous["attempts"]
    return [previous]


def answered_ids(partial_path: Path) -> tuple[set[str], bool]:
    """custom_ids already answered in OUT.partial. A last line cut off mid-write is removed (and reported)."""
    if not partial_path.is_file():
        return set(), False
    lines = [line for line in partial_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids: set[str] = set()
    for index, line in enumerate(lines):
        try:
            ids.add(json.loads(line)["custom_id"])
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f"{partial_path} line {index + 1} is not valid JSON; it was not the last line, "
                                 "so it was not cut off by a stop. Check the file.") from None
            partial_path.write_text("".join(kept + "\n" for kept in lines[:-1]), encoding="utf-8")
            return ids, True
    return ids, False


def log_rejection(path: Path, custom_id: str, error: BaseException) -> None:
    with path.open("a", encoding="utf-8") as handle:
        record = {"custom_id": custom_id, "status_code": getattr(error, "status_code", None),
                  "error": str(error), "at": utc_now_iso()}
        handle.write(json.dumps(record) + "\n")


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
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - recorded in the registry and run_failed.txt
            # Ctrl-C included: the experiment must still reach the registry, or its paid answers are never collected.
            reason = "interrupted (Ctrl-C)" if isinstance(error, KeyboardInterrupt) else str(error)
            print(
                f"Qwen run stopped: {reason}\nDetails: {output_dir / FAILURE_FILENAME}\n"
                f"Answers so far are kept. Resume with:\n  {resume_command(input_jsonl_uri, output_dir / OUTPUT_FILENAME)}"
                "\nthen run collect-results.",
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
        output_dir = Path(output_uri_prefix or "")
        rejected = output_dir / REJECTED_FILENAME
        if rejected.is_file():
            count = sum(1 for line in rejected.read_text(encoding="utf-8").splitlines() if line.strip())
            print(f"Warning: {count} requests in {output_dir.parent.name} were rejected by the API and have no "
                  f"results row; see {rejected}", file=sys.stderr)
        return (output_dir / OUTPUT_FILENAME).read_text(encoding="utf-8")

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
