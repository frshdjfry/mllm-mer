"""Follow songs through the mllm-mer pipeline, step by step, printing every intermediate result.

For learning and checking, not for experiments. Nothing is sent to any model,
and nothing is written to outputs/ or the registry. The script replays saved
experiments:

1. It reads one saved experiment's spec: dataset folder, prompt file, model and
   settings.
2. It rebuilds every pipeline step from the audio file with the real
   mllm-mer functions.
3. It checks each rebuilt step against the files saved in that experiment.
4. It shows the model's saved raw answer and parses it again.

For each song it shows what each backend would receive:
- Qwen: the API call.
- Gemini: the batch request line.
- Music Flamingo: the chat conversation.

Only Qwen has saved answers for these songs. Pass --gemini-experiment to also
parse a saved Gemini answer; it is for a different song with the same prompt.

    .venv/bin/python scripts/trace_pipeline.py                       # newest run per Qwen model
    .venv/bin/python scripts/trace_pipeline.py --song web_preview_full_003
    .venv/bin/python scripts/trace_pipeline.py --experiment outputs/experiments/<id>
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

import collect  # noqa: E402
import datasets  # noqa: E402
import music_flamingo  # noqa: E402
import parser as mer_parser  # noqa: E402  (mllm-mer's parser.py, not the old stdlib module)
import prompts  # noqa: E402
import qwen_api  # noqa: E402
import request_builder  # noqa: E402

WIDTH = 100
MAX_STRING = 160  # longer strings (base64 audio, long answers) are shortened when printed


# ----------------------------------------------------------------------------- printing


def where(func) -> str:
    """'module.function (file.py:line)' for any mllm-mer function."""
    path = Path(inspect.getsourcefile(func)).relative_to(REPO_DIR)
    line = inspect.getsourcelines(func)[1]
    return f"{func.__module__}.{func.__qualname__}  ({path}:{line})"


def step(number: str, title: str, *funcs, what: str = "") -> None:
    print()
    print("─" * WIDTH)
    print(f"STEP {number} · {title}")
    for func in funcs:
        print(f"  code: {where(func)}")
    if what:
        for line in textwrap.wrap(what, WIDTH - 4):
            print(f"  {line}")
    print("─" * WIDTH)


def shorten(value: Any) -> Any:
    """Copy of value with long strings cut, so base64 audio doesn't fill the screen."""
    if isinstance(value, str) and len(value) > MAX_STRING:
        return f"{value[:MAX_STRING]}… [{len(value):,} characters in total]"
    if isinstance(value, dict):
        return {key: shorten(item) for key, item in value.items()}
    if isinstance(value, list):
        return [shorten(item) for item in value]
    return value


def compact_json(value: Any, indent: int = 0) -> str:
    """Indented JSON, but any part that fits on one line is kept on one line."""
    one_line = json.dumps(value, ensure_ascii=False, default=str)
    if len(one_line) + indent <= WIDTH - 4 or not isinstance(value, (dict, list)) or not value:
        return one_line
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        items = [f"{pad}{json.dumps(key)}: {compact_json(item, indent + 2)}" for key, item in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [f"{pad}{compact_json(item, indent + 2)}" for item in value]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def show(label: str, value: Any, cut: bool = True) -> None:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        value = {name: getattr(value, name) for name in value.__dataclass_fields__}
    text = compact_json(shorten(value) if cut else value)
    print(f"\n▸ {label}")
    print(textwrap.indent(text, "    "))


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"\n  {'✓' if ok else '✗ MISMATCH'}  {label}" + (f": {detail}" if detail else ""))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def without_experiment_id(payload: Any, experiment_id: str) -> Any:
    """Replace the experiment id (it holds a timestamp) so rebuilt and saved lines can be compared."""
    return json.loads(json.dumps(payload).replace(experiment_id, "<experiment_id>"))


# ----------------------------------------------------------------------------- tracing


def default_experiments() -> list[Path]:
    """Newest saved Qwen experiment for each model (by the time it was created, not by folder name)."""
    newest: dict[str, tuple[str, Path]] = {}
    for spec_path in (REPO_DIR / "outputs" / "experiments").glob("*/experiment_spec.json"):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("backend") != "qwen" or not (spec_path.parent / "results" / "raw_output.jsonl").is_file():
            continue
        created = spec.get("created_at", "")
        if spec["model"] not in newest or created > newest[spec["model"]][0]:
            newest[spec["model"]] = (created, spec_path.parent)
    return [path for _, path in newest.values()]


def shown_path(path: Path) -> str:
    """Path relative to the repository when inside it, else absolute."""
    try:
        return str(path.relative_to(REPO_DIR))
    except ValueError:
        return str(path)


def trace_experiment(experiment_dir: Path, songs: set[str] | None, gemini_dir: Path | None) -> None:
    spec = json.loads((experiment_dir / "experiment_spec.json").read_text(encoding="utf-8"))
    experiment_id = spec["experiment_id"]
    print("═" * WIDTH)
    print(f"REPLAYING SAVED EXPERIMENT  {experiment_id}")
    print(f"  model: {spec['model']}   backend: {spec['backend']}   "
          f"temperature: {spec['temperature']}   base_seed: {spec['base_seed']}   top_p: {spec['top_p']}")
    print(f"  folder: {shown_path(experiment_dir)}")
    print("═" * WIDTH)

    # STEP 1: the dataset folder becomes a list of input records ----------------------------
    step("1", "Find the audio files", datasets.list_local_audio_records, datasets.dataset_id_from_local_dir,
         what="Every audio file in the dataset folder becomes an InputRecord: an id (the file name without "
              "extension), its path, and a SHA-256 checksum that proves later which exact file was used. "
              "(Gemini runs use datasets.list_audio_records instead, which lists a gs:// bucket.)")
    records = datasets.list_local_audio_records(spec["input_source"])
    print(f"\n  dataset_id = {datasets.dataset_id_from_local_dir(spec['input_source'])!r}; "
          f"{len(records)} audio files found: {[r.file_id for r in records]}")
    saved_manifest = json.loads((experiment_dir / "input_manifest.json").read_text(encoding="utf-8"))
    saved_sums = {r["file_id"]: r["attributes"].get("sha256") for r in saved_manifest["records"]}
    check("checksums match the saved input_manifest.json (same audio as the real run)",
          all(saved_sums.get(r.file_id) == r.attributes["sha256"] for r in records))

    # STEP 2: the prompt file becomes prompt instances -------------------------------------
    step("2", "Load and expand the prompt", prompts.load_prompt_spec, prompts.expand_prompt_instances,
         what="The YAML prompt file is read into a PromptSpec. A 'simple' prompt gives one instance; a "
              "'templated' prompt gives one instance per combination of its variables. response_schema "
              "lists the field names and types the answer must have.")
    prompt_spec = prompts.load_prompt_spec(spec["prompt_spec_path"])
    print(f"\n  file: {spec['prompt_spec_path']}")
    print(f"  prompt_id = {prompt_spec.prompt_id!r}, kind = {prompt_spec.kind!r}, "
          f"audio_position = {prompt_spec.audio_position!r} (audio goes {prompt_spec.audio_position} the text)")
    instances = prompts.expand_prompt_instances(prompt_spec)
    for instance in instances:
        print(f"\n▸ prompt instance {instance.prompt_instance_id!r}, text exactly as sent:")
        print(textwrap.indent(instance.prompt_text, "    │ "))
        show("response_schema", instance.response_schema)
    check("prompt instances match the saved prompt_instances.json",
          [i.to_dict() for i in instances]
          == json.loads((experiment_dir / "prompt_instances.json").read_text(encoding="utf-8")))

    # STEP 3: songs x prompts x trials become requests ------------------------------------
    step("3", "Build one request per song × prompt instance × trial", request_builder.build_requests,
         what="Each request gets a unique id (experiment__song__prompt__trial). Trial i gets seed base_seed + i. "
              "These are backend-neutral Python objects (RequestItem).")
    requests = request_builder.build_requests(
        experiment_id=experiment_id, dataset_id=spec["dataset_id"], model=spec["model"],
        input_records=records, prompt_instances=instances, trials=spec["trials"],
        temperature=spec["temperature"], base_seed=spec["base_seed"], top_p=spec["top_p"],
    )
    print(f"\n  {len(records)} songs × {len(instances)} prompt instances × {spec['trials']} trials "
          f"= {len(requests)} requests")
    saved_requests = {line["custom_id"]: line for line in read_jsonl(experiment_dir / "requests.jsonl")}
    saved_answers = {line["custom_id"]: line for line in read_jsonl(experiment_dir / "results" / "raw_output.jsonl")}
    metadata_index = mer_parser.load_request_metadata_index(experiment_dir / "request_metadata.csv")
    saved_parsed = {}
    with (experiment_dir / "results" / "parsed_long.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            saved_parsed[(row["file_id"], row["prompt_instance_id"], row["trial_index"])] = row

    for request in requests:
        if songs and request.file_id not in songs:
            continue
        trace_request(request, spec, experiment_dir, saved_requests, saved_answers, metadata_index, saved_parsed)

    if gemini_dir:
        trace_gemini_answer(gemini_dir)


def trace_request(request, spec, experiment_dir, saved_requests, saved_answers, metadata_index, saved_parsed) -> None:
    print()
    print("═" * WIDTH)
    print(f"SONG {request.file_id}   (trial {request.trial_index}, prompt {request.prompt_instance_id})")
    print(f"  audio: {request.file_uri}")
    print("═" * WIDTH)
    show("RequestItem (backend-neutral)", {**request.__dict__, "prompt_text": request.prompt_text}, cut=True)

    # STEP 4: the shared request line --------------------------------------------------------
    step("4", "Write the shared request line (requests.jsonl)", request_builder.build_gemini_batch_request_json,
         request_builder.convert_response_schema, request_builder.build_contents,
         what="Every backend gets the same file, in Gemini's batch format. The audio is referenced by its "
              "path, not included; response_schema is converted to Gemini's format; settings go in "
              "generationConfig.")
    line = request_builder.build_gemini_batch_request_json(request)
    show("requests.jsonl line", line)
    saved_line = saved_requests.get(request.request_id)
    check("identical to the line saved in requests.jsonl", saved_line == line)

    # STEP 5: what each backend makes of that line ----------------------------------------
    step("5a", "Qwen: translate to an OpenAI-style chat call", qwen_api.request_to_chat_kwargs,
         qwen_api.audio_part, qwen_api.sent_settings,
         what="qwen_api reads the audio file, encodes it as base64 text (audio as letters, so it fits in "
              "JSON), and sends it inside the message. JSON mode is requested; the field names are only in "
              "the prompt text. These arguments go to the OpenAI Python library pointed at Alibaba's server "
              "(qwen_api.build_openai_call), streamed.")
    kwargs = qwen_api.request_to_chat_kwargs(line)
    show("arguments for client.chat.completions.create(...)", kwargs)
    recorded = saved_answers.get(request.request_id, {}).get("qwen", {}).get("sent")
    check("settings match what the real run recorded as sent (raw_output.jsonl → qwen.sent)",
          recorded == qwen_api.sent_settings(kwargs),
          "" if recorded == qwen_api.sent_settings(kwargs) else f"recorded {recorded}")

    step("5b", "Gemini (Vertex batch): the line is sent unchanged", request_builder.build_gemini_batch_request_json,
         what="vertex_batch.VertexBatchClient uploads requests.jsonl to Google Cloud Storage and starts a batch "
              "job; Gemini reads each line as is. One difference in a real Gemini run: file_uri would be a "
              "gs:// address, since Gemini reads audio from Google's storage, not this Mac. Gemini enforces "
              "responseSchema, so it must answer with exactly these fields.")
    gemini_line = copy.deepcopy(line)
    gemini_line["request"]["model"] = "<gemini model, e.g. gemini-3.8-flash>"  # this experiment's model is Qwen
    gemini_line["request"]["contents"][0]["parts"] = [
        {"file_data": {**part["file_data"], "file_uri": f"gs://<bucket>/{Path(part['file_data']['file_uri']).name}"}}
        if "file_data" in part else part
        for part in gemini_line["request"]["contents"][0]["parts"]
    ]
    show("what Gemini would receive", gemini_line)

    step("5c", "Music Flamingo: translate to a chat conversation", music_flamingo.request_to_conversation,
         music_flamingo.build_hf_generator,
         what="music_flamingo turns the line into a Hugging Face chat conversation (audio by path, then text) "
              "and runs the model on a GPU. The processor turns audio into features and text into tokens. "
              "Generation rule (music_flamingo.py, inside build_hf_generator): temperature > 0 → sample at "
              "that temperature (plus top_p if given); temperature 0 → greedy, no sampling. responseSchema is "
              "not used; the field names are only in the prompt text.")
    conversation, temperature, seed, top_p = music_flamingo.request_to_conversation(line)
    show("conversation given to processor.apply_chat_template", conversation)
    print(f"\n  generation settings → temperature={temperature}, seed={seed}, top_p={top_p}")

    # STEP 6: raw answer ---------------------------------------------------------------------
    step("6", "The model's answer, as saved (raw_output.jsonl)", qwen_api.collect_stream, qwen_api.output_line,
         what="Qwen streams its answer in small pieces; collect_stream joins them, and output_line stores the "
              "result in Gemini's output shape, so one parser serves every backend. Extra Qwen details "
              "(request id, Flash's thinking text, settings sent) sit under 'qwen'. The unprocessed stream is "
              "not saved.")
    answer = saved_answers.get(request.request_id)
    if answer is None:
        print("\n  (no saved answer for this request)")
        return
    show("raw_output.jsonl line", answer)
    reasoning = answer.get("qwen", {}).get("reasoning", "")
    if reasoning:
        print(f"\n  Flash thought for {len(reasoning):,} characters before answering (shown shortened above).")

    # STEP 7: parse --------------------------------------------------------------------------
    step("7", "Parse the answer into a results row (parsed_long.csv)", mer_parser.parse_output_jsonl_text,
         mer_parser.extract_response_json, mer_parser.check_schema_keys, collect.run_provenance,
         what="The parser finds the answer text, reads it as JSON, and makes one row: the request's details, "
              "one answer_<field> column per field in the answer (the prefix marks what the model wrote, so it can "
              "never overwrite our columns), the audio's SHA-256, the settings, provenance (which model answered, which "
              "code commit), and schema_keys: 'exact' if the answer used exactly the schema's field names. "
              "If the text is not valid JSON, the whole response is stored unparsed instead.")
    print(f"\n  answer text the parser reads: "
          f"{mer_parser.extract_candidate_texts(answer['response'])[0]!r}"[: WIDTH * 3])
    rows = mer_parser.parse_output_jsonl_text(
        json.dumps(answer), job_id=f"qwen:{(experiment_dir / 'results').resolve()}",
        request_metadata_index=metadata_index, submitted_at="",
        run_columns=collect.run_provenance("qwen", experiment_dir),
        schema_fields=collect.load_schema_fields(experiment_dir / "prompt_instances.json"),
        audio_sha256=mer_parser.load_audio_sha256(experiment_dir / "input_manifest.json"),
    )
    row = rows[0]
    show("parsed row", {key: value for key, value in row.items() if key not in ("raw_response_json",)})
    saved = saved_parsed.get((row["file_id"], row["prompt_instance_id"], row["trial_index"]), {})
    columns = [mer_parser.ANSWER_PREFIX + field for field in request.response_schema]
    # Tables built before 2026-11-25 have no answer_ prefix; compare with either.
    check("emotion values match the saved parsed_long.csv",
          all(saved.get(column, saved.get(column.removeprefix(mer_parser.ANSWER_PREFIX))) == row.get(column)
              for column in columns),
          ", ".join(f"{column}={row.get(column)}" for column in columns))


def trace_gemini_answer(gemini_dir: Path) -> None:
    step("6–7 (Gemini)", "A saved Gemini answer, parsed the same way", mer_parser.parse_output_jsonl_text,
         what=f"No Gemini answers are saved for these songs, so this is the first answer of {gemini_dir.name}: "
              "a different song, same parser. Note Gemini's own fields (modelVersion, usageMetadata) in the "
              "same shape as the Qwen lines above.")
    answer = read_jsonl(gemini_dir / "results" / "raw_output.jsonl")[0]
    show("raw_output.jsonl line (Gemini)", answer)
    rows = mer_parser.parse_output_jsonl_text(
        json.dumps(answer), job_id="(gemini example)",
        request_metadata_index=mer_parser.load_request_metadata_index(gemini_dir / "request_metadata.csv"),
        run_columns=collect.run_provenance("vertex", gemini_dir),
        schema_fields=collect.load_schema_fields(gemini_dir / "prompt_instances.json"),
        audio_sha256=mer_parser.load_audio_sha256(gemini_dir / "input_manifest.json"),
    )
    show("parsed row (Gemini)", {key: value for key, value in rows[0].items() if key != "raw_response_json"})


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    cli.add_argument("--experiment", action="append", type=Path,
                     help="Saved Qwen experiment folder to replay (repeatable). Default: the newest one per Qwen model.")
    cli.add_argument("--song", action="append", help="Only this file_id (repeatable), e.g. web_preview_full_003.")
    cli.add_argument("--gemini-experiment", type=Path,
                     help="Saved Gemini experiment folder whose first answer is also parsed (any song).")
    args = cli.parse_args()

    experiments = [path.resolve() for path in args.experiment] if args.experiment else default_experiments()
    if not experiments:
        cli.error("No saved Qwen experiments found in outputs/experiments; pass --experiment.")
    for index, experiment_dir in enumerate(experiments):
        gemini = args.gemini_experiment if index == len(experiments) - 1 else None  # show once, at the end
        trace_experiment(experiment_dir, set(args.song) if args.song else None, gemini)


if __name__ == "__main__":
    main()
