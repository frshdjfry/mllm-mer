"""Runner for the music-flamingo backend: run requests on a local GPU.

Reads a requests.jsonl written by `submit-experiment --backend
music-flamingo` (the same request format as the Gemini batch backend),
runs NVIDIA Music Flamingo with Hugging Face transformers on whatever GPU
this process can see, and writes raw_output.jsonl in the same shape as a
Gemini batch output, so `collect-results` parses both the same way.

    python -m music_flamingo run --requests REQ.jsonl --output OUT.jsonl --config CFG.yaml

The launcher (local_backend.py) normally starts this, directly or via Slurm.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from schemas import utc_now_iso

# (conversation, temperature, seed) -> generated text
GenerateFn = Callable[[list[dict[str, Any]], float, "int | None"], str]

FAILURE_FILENAME = "run_failed.txt"
RUN_INFO_FILENAME = "run_info.json"


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Backend config must be a mapping: {path}")
    model = data.get("model") or {}
    if not model.get("id") or not model.get("revision"):
        raise ValueError("Backend config needs model.id and a pinned model.revision.")
    return data


def model_label(config: dict[str, Any]) -> str:
    """Model name recorded in results: id plus the pinned revision."""
    return f"{config['model']['id']}@{config['model']['revision']}"


def read_requests(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def request_to_conversation(request: dict[str, Any]) -> tuple[list[dict[str, Any]], float, int | None]:
    """Turn one batch request line into a Music Flamingo chat conversation."""
    body = request["request"]
    content: list[dict[str, Any]] = []
    # Keep the order of the request's parts, so both backends get the same prompt.
    for message in body["contents"]:
        for part in message["parts"]:
            if "file_data" in part:
                content.append({"type": "audio", "path": part["file_data"]["file_uri"]})
            elif "text" in part:
                content.append({"type": "text", "text": part["text"]})
    generation = body.get("generationConfig", {})
    return [{"role": "user", "content": content}], float(generation.get("temperature", 1.0)), generation.get("seed")


def output_line(custom_id: str, text: str) -> dict[str, Any]:
    """Same shape as a Gemini batch output line, so the existing parser applies."""
    return {"custom_id": custom_id, "response": {"candidates": [{"content": {"parts": [{"text": text}]}}]}}


def run(
    requests_path: Path,
    output_path: Path,
    config: dict[str, Any],
    generate: GenerateFn | None = None,
) -> int:
    """Run every request; write output atomically. Returns the number of requests."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path = output_path.parent / FAILURE_FILENAME
    failure_path.unlink(missing_ok=True)
    info: dict[str, Any] = {
        "model": model_label(config),
        "started_at": utc_now_iso(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    try:
        requests = read_requests(requests_path)
        if generate is None:
            generate, environment = build_hf_generator(config)
            info.update(environment)
        tmp_path = output_path.with_suffix(output_path.suffix + ".partial")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                conversation, temperature, seed = request_to_conversation(request)
                text = generate(conversation, temperature, seed)
                handle.write(json.dumps(output_line(request["custom_id"], text)) + "\n")
                handle.flush()
        os.replace(tmp_path, output_path)
        info.update({"finished_at": utc_now_iso(), "requests": len(requests)})
        return len(requests)
    except BaseException:
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        info["failed_at"] = utc_now_iso()
        raise
    finally:
        (output_path.parent / RUN_INFO_FILENAME).write_text(json.dumps(info, indent=2), encoding="utf-8")


def build_hf_generator(config: dict[str, Any]) -> tuple[GenerateFn, dict[str, Any]]:
    """Load the pinned model once and return a generate function plus environment info."""
    import torch
    import transformers
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    model_cfg = config["model"]
    generation_cfg = config.get("generation") or {}
    dtype = getattr(torch, generation_cfg.get("dtype", "bfloat16"))
    processor = AutoProcessor.from_pretrained(model_cfg["id"], revision=model_cfg["revision"])
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_cfg["id"], revision=model_cfg["revision"], device_map="auto", torch_dtype=dtype
    )
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 1024))

    def generate(conversation: list[dict[str, Any]], temperature: float, seed: int | None) -> str:
        inputs = processor.apply_chat_template(
            [conversation], tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)
        if seed is not None:
            torch.manual_seed(seed)
        outputs = model.generate(**inputs, **kwargs)
        return processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    environment = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda or "",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "dtype": str(dtype),
        "max_new_tokens": max_new_tokens,
    }
    return generate, environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Music Flamingo requests on a local GPU.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run a requests.jsonl and write raw_output.jsonl.")
    run_parser.add_argument("--requests", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    run_parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    count = run(args.requests, args.output, load_config(args.config))
    print(f"Wrote {count} responses to {args.output}")


if __name__ == "__main__":
    main()
