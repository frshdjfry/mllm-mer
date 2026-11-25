# Gemini Batch Audio CLI

Small Python CLI for running Gemini batch experiments on audio files stored in a GCS bucket or prefix.
It also supports text-only experiments from a CSV containing `file_id` and `transcription`.

Batch submission uses the `google.genai` batch API style:

- upload local request JSONL to GCS
- call `client.batches.create(model=..., src=gs://...)`
- poll with `client.batches.get(...)`
- download the finished JSONL with `client.files.download(...)`

The project keeps the workflow intentionally simple:

- `submit-experiment`
  - lists audio files from a dataset `gs://` URI
  - saves local experiment artifacts
  - expands prompt specs into prompt instances
  - builds request JSONL
  - uploads the JSONL to GCS
  - submits a Vertex batch job
  - appends job metadata to a CSV registry

- `submit-text-experiment`
  - reads a local CSV with `file_id` and `transcription`
  - saves local experiment artifacts
  - expands prompt specs into prompt instances
  - builds request JSONL
  - uploads the JSONL to GCS
  - submits a Vertex batch job
  - appends job metadata to a CSV registry

- `collect-results`
  - reads the CSV registry
  - checks uncollected jobs
  - downloads finished output JSONL files
  - parses model responses into a normalized long-form CSV
  - marks the job as collected

## Files

- `cli.py`
- `schemas.py`
- `datasets.py`
- `prompts.py`
- `request_builder.py`
- `submit.py`
- `collect.py`
- `registry.py`
- `parser.py`
- `vertex_batch.py`

## Setup

Before running the CLI, initialize and authorize the `gcloud` CLI:

- Install the Google Cloud SDK and follow the auth steps here:
  [Initialize and authorize the gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk)
- Run:

```bash
gcloud auth application-default login
```

- Visit Vertex AI in Google Cloud Console and enable it for your project.
- Clone the project:

```bash
git clone https://github.com/frshdjfry/mllm-mer.git
cd mllm-mer
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install google-genai google-cloud-storage PyYAML
```

Typical environment:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="global"
```

## Prompt Specs

Simple prompt example: [examples/simple_prompt.yaml](/Users/frshd/Documents/Codex/2026-06-12/build-a-small-python-cli-project/examples/simple_prompt.yaml)

Templated prompt example: [examples/templated_emotion_prompt.yaml](/Users/frshd/Documents/Codex/2026-06-12/build-a-small-python-cli-project/examples/templated_emotion_prompt.yaml)

Text input CSV example: [examples/transcriptions.csv](/Users/frshd/Documents/Codex/2026-06-12/build-a-small-python-cli-project/examples/transcriptions.csv)

## Usage

Submit an experiment:

```bash
python cli.py submit-experiment \
  --dataset-uri gs://your-dataset-bucket/audio-prefix \
  --prompt-spec examples/templated_emotion_prompt.yaml \
  --model models/gemini-2.5-flash \
  --trials 3 \
  --output-uri-prefix gs://your-output-bucket/gemini-batch-runs
```

Generation settings (both submit commands):

- `--temperature` (default `1.0`, the Gemini 3 default; Google advises not
  lowering it for Gemini 3 models) is sent with every request.
- `--top-p P` (optional, 0 < P <= 1) samples only from the most likely
  tokens whose probabilities add up to P (nucleus sampling). Omitted, it is
  not sent and the model's own default applies. Not allowed with
  `--temperature 0`.
- `--seed N` (optional) makes sampling best-effort reproducible: trial `i`
  uses seed `N + i`, so trials still differ from each other but a re-run
  with the same `N` should repeat them. Google does not guarantee identical
  output.

Each parsed result row records `submitted_at`, `temperature`, `top_p`, `seed` and
`prompt_text_hash` (SHA-256 of the rendered prompt text from the prompt
spec, as 64 hex characters; any edit to the prompt changes it). For text
experiments it covers the prompt only, not the transcription appended to
it. Model answers cannot clash with these columns (see `answer_` below), and
prompt variables may not use their names.

Every row also has the same provenance columns for both backends, so
Gemini and Music Flamingo results can be compared in one table:

| Column | Gemini (`vertex`) | Music Flamingo |
|---|---|---|
| `backend` | `vertex` | `music-flamingo` |
| `model_served` | `modelVersion` from that response line | pinned `id@revision` |
| `settings_source` | `as sent; Vertex does not report the settings used` | `effective; see results/run_info.json` |
| `code_commit` | commit at submit time | commit the runner ran |
| `hardware` | `google_hosted` | GPU name (`cpu` without one) |

Unknown values are named, never blank: `not_recorded` (e.g. runs from
before code commits were recorded), `vertex_default` for a temperature that
was never sent to Gemini, `not_set` for a seed that was never sent, and
`vertex_default` / `model_default` for a top_p that was never sent.
`code_commit` ends in `+uncommitted` if the code had uncommitted changes.

`schema_keys` says whether the answer used the prompt's `response_schema`
field names. Gemini is forced to use them; Music Flamingo is only asked to,
and may write e.g. `"joyful activation"` for `joyful_activation`. Keys are
never respelled (only prefixed with `answer_`): prevent such answers in the
prompt (one spelling per field name), and check every row that is not
`exact` before analysis. `schema_keys` compares the model's own key names,
without the prefix.

| `schema_keys` | Meaning |
|---|---|
| `exact` | every key is a schema name |
| `missing: …` | schema fields not answered (also when the answer could not be parsed) |
| `extra: …` | keys not in the schema, kept as returned |
| `not_checked` | no saved schema, or the answer is not a JSON object |

### Results table columns (data dictionary)

What each column of `results/parsed_long.csv` holds and where it comes from.
One row per request: one song × one prompt instance × one trial.

**What we sent to the model.** The model hears the audio and reads the
prompt text; it never sees file names, paths or ids.

| Column | Meaning |
|---|---|
| `file_uri` | Where the audio was read from: a `gs://` address (Gemini), or the full local path on the machine that ran the job (Qwen, Music Flamingo). |
| `model` | The model requested. |
| prompt variables, e.g. `item` | The value put into the prompt template for this request (templated prompts only). |
| `temperature`, `top_p`, `seed` | Decoding settings, or a marker saying none was sent (see above). |

The prompt text itself is not a column: `prompt_instance_id` and
`prompt_text_hash` identify it in `prompt_instances.json`.

**Identifiers (ours; never sent to the model).**

| Column | Meaning |
|---|---|
| `experiment_id` | One submitted run. |
| `dataset_id` | The audio folder or bucket the songs came from. |
| `file_id` | The song's id within the dataset: the file name without extension (local datasets) or the name in the bucket (Gemini). It can differ between backends for the same song. |
| `audio_sha256` | SHA-256 checksum of the audio file's bytes, from `input_manifest.json`. The same audio has the same checksum on every backend and machine, so use it to join results across backends. `not_recorded` where the manifest has none (Cloud Storage datasets so far). |
| `prompt_id`, `prompt_instance_id` | The prompt spec, and the expanded prompt (one per combination of variable values). |
| `trial_index` | Repeat number of the same request (0, 1, …). |
| `job_id` | Backend job that ran the request. |
| `submitted_at`, `created_at` | When the run was submitted; when this table was built. |
| `prompt_text_hash` | SHA-256 of the rendered prompt text. |

**Run provenance (facts we record about the run):** `backend`,
`model_served`, `settings_source`, `code_commit`, `hardware` (table above).

**From the model.**

| Column | Meaning |
|---|---|
| `raw_response_json` | The model's answer exactly as returned (the parsed JSON, or the whole response if it could not be read as JSON). |
| `answer_<key>` | One column per key in the answer, e.g. `answer_wonder`. The prefix marks every column the model wrote, so an answer can never overwrite one of ours: a model's `"file_id"` goes to `answer_file_id`. Values are as returned; key names are never respelled. |
| `answer_candidates` | Present when the answer was not valid JSON: the unparsed response. |
| `answer_value` | The answer, when it was valid JSON but not an object (e.g. a bare list). |
| `schema_keys` | Our check of the answer's key names against the prompt's `response_schema` (table above). |

Tables built before 2026-11-25 have no `answer_` prefix and no
`audio_sha256`, and an answer key could overwrite a column of the same name:
in Gemini runs whose prompt asked for `"file_id"`, the `file_id` column holds
the model's guess. `rebuild-results` (below) rebuilds them from the saved raw
output.

The registry gains `temperature` and `base_seed` columns. An older registry
is upgraded on the first submit with this version; after that, older
versions of this CLI cannot read it, so keep using this version.

Collect finished results:

```bash
python cli.py collect-results
```

After the results table format changes, rebuild the tables of already
collected jobs from their saved `results/raw_output.jsonl` (the raw output is
only read, never changed; `created_at` becomes the rebuild time):

```bash
python cli.py rebuild-results
```

Submit a text-only experiment:

```bash
python cli.py submit-text-experiment \
  --input-csv examples/transcriptions.csv \
  --prompt-spec examples/templated_emotion_prompt.yaml \
  --model models/gemini-2.5-flash \
  --trials 2 \
  --output-uri-prefix gs://your-output-bucket/gemini-batch-runs
```

## Music Flamingo backend (local GPU or Slurm)

`--backend music-flamingo` runs [NVIDIA Music Flamingo](https://huggingface.co/nvidia/music-flamingo-hf)
with Hugging Face `transformers` on a GPU you control, using the same prompt
specs, registry, artifacts and results columns as the Gemini backend. Audio
is read from a local folder; nothing is uploaded. The model is
non-commercial-research only (NVIDIA OneWay Noncommercial License).

Requirements where the model runs: a GPU with roughly 40 GB of memory (the
weights are 16.6 GB) and the pinned environment from
`requirements-music-flamingo.txt` (`python -m pip install -r requirements-music-flamingo.txt`;
`transformers` is a development commit, as Music Flamingo needs unreleased code). The machine that
only submits and collects does not need PyTorch.

1. Copy `config/music_flamingo.cambridge-hpc.example.yaml`, fill in your
   paths, and keep the pinned `model.revision`. Copies of the example are
   gitignored.
2. Submit, as a Slurm job (default) or directly on this machine's GPU:

```bash
python cli.py submit-experiment --backend music-flamingo \
  --dataset-dir /path/to/anonymised/audio \
  --prompt-spec examples/simple_prompt.yaml \
  --backend-config config/my_music_flamingo.yaml \
  --trials 3 --seed 0 \
  --launcher slurm        # or: --launcher local
```

3. Collect as usual with `python cli.py collect-results`. For Slurm jobs, run
   it on the cluster, where `sacct` can see the job.

What the model actually receives (checked with the pinned processor):
- Its chat template always places the audio **before** the text, whatever
  order the request uses, so `audio_position` has no effect for Music
  Flamingo. Use `audio_position: before` (default) when comparing with Gemini.
- The template adds a system message, "You are a helpful assistant.".
- Audio becomes 25 tokens per second (750 tokens per 30 s window).
- Prompt text must not contain `<sound>`: it is the model's audio placeholder
  token and duplicates the audio slot. The runner rejects such prompts.

How it works: `submit-experiment` writes `requests.jsonl` in the same format
as for Gemini, then the launcher (`local_backend.py`) starts the runner
(`music_flamingo.py`) directly or writes and `sbatch`-submits
`slurm_job.sh` from the config. The runner writes `results/raw_output.jsonl`
in Gemini's batch output shape, plus `run_info.json` (model revision, package
versions, GPU, timings) or `run_failed.txt` with the error. A request without
a temperature uses the model's own published generation settings
(`generation_config.json`, which sets no sampling: greedy decoding); the
settings actually used are recorded in `run_info.json`
(`effective_generation_config`). `--temperature 0` also means greedy
decoding; above 0 samples at that temperature, with `--top-p` if given. The
model card's sampling example uses temperature 0.7 and top_p 0.9.

The registry gains a `backend` column (`vertex` for existing rows); as with
the earlier columns, an older registry is upgraded once and older versions of
this CLI cannot read it afterwards.

Offline tests (no GPU, Slurm or network): `python -m unittest discover -s tests`.
`tests/test_formats.py` locks every file format against golden files in
`tests/golden/`; after an intentional format change, regenerate them with
`UPDATE_GOLDEN=1 python -m unittest tests.test_formats` and review the diff.
Prompt specs may set `audio_position: before` (default) or `after`, and both
backends send the parts in that order.

## Qwen backend (Alibaba Cloud Model Studio API)

`--backend qwen` sends each request to a Qwen Omni model through Model
Studio's OpenAI-compatible Chat Completions endpoint, using the same prompt
specs, registry, artifacts and results columns as the other backends. Tested
model IDs: `qwen3.5-omni-plus` and `qwen3.8-omni-flash`.

Setup: `pip install "openai>=1.52" PyYAML`, then copy `.env.example` to `.env`
(gitignored) and fill in `DASHSCOPE_API_KEY` and `QWEN_BASE_URL` (Singapore
and Beijing keys and URLs differ). The key is read only from the environment
or `.env`, never from code or the command line.

```bash
python cli.py submit-experiment --backend qwen \
  --dataset-dir /path/to/audio \
  --prompt-spec examples/simple_prompt.yaml \
  --model qwen3.5-omni-plus \
  --trials 1 --seed 0
python cli.py collect-results
```

How it differs from Gemini:
- Audio is read from a local folder and sent inline as base64 (under 10 MB
  encoded per file); nothing is uploaded to a bucket.
- Requests run one by one at submit time (Model Studio has no batch API for
  these models in Singapore). If a run stops, its registry status is
  `JOB_STATE_STOPPED`, answers so far stay in `raw_output.jsonl.partial`, and
  the printed `python -m qwen_api run ...` command resumes it without paying
  for answered requests again; then run `collect-results`. This also covers
  Ctrl-C. `collect-results` prints the resume command for stopped runs. A
  finished run is never run again, because that would re-bill everything.
  Each attempt is kept in `run_info.json` (`attempts`). If attempts used
  different code, the `code_commit` column says `mixed: ...`, and each answer
  records its own commit in `raw_output.jsonl` (`qwen.code_commit`).
- All audio files are checked (format, size) before anything is sent. A request
  the API rejects (HTTP 400/422, e.g. content inspection) is logged in
  `results/rejected_requests.jsonl` and has no results row. The run goes on,
  and `collect-results` warns about it. If the first 3 requests are all
  rejected, the run stops, since that points to a settings problem.
- `response_format: json_object` is sent, so answers are bare JSON (without
  it, both models wrap JSON in a ```` ```json ```` block, which the parser does
  not read). Model Studio's docs list it as unsupported in Singapore, but it
  worked there on 2026-11-24. Unlike Gemini's `responseSchema`, it does not
  enforce the field names: check `schema_keys`.
- `max_tokens` and `enable_thinking` are not sent. `qwen3.8-omni-flash` then
  thinks before answering (its reasoning text is kept in `raw_output.jsonl`
  under `qwen.reasoning` and costs output tokens); `qwen3.5-omni-plus` does not.
- `model_served` is the model name Model Studio returned for each answer;
  `raw_output.jsonl` also keeps token usage, the request id, any reasoning
  text, and the settings sent (without the audio). `hardware` is
  `alibaba_hosted`; `results/run_info.json` holds SDK version and token totals.

## Artifacts

Each experiment gets a local directory under `outputs/experiments/<experiment_id>/` with:

- `input_manifest.json`
- `experiment_spec.json`
- `prompt_instances.json`
- `requests.jsonl`
- `request_metadata.csv`
- `results/raw_output.jsonl`
- `results/parsed_long.csv`

The registry lives at:

- `outputs/registry/experiments_registry.csv`

## Notes

- Provider-specific Vertex/Gemini batch logic is isolated in `vertex_batch.py`.
- Prompt loading and template expansion are isolated in `prompts.py`.
- CSV registry handling is isolated in `registry.py`.
- Result normalization is isolated in `parser.py`.
- The request JSONL is structured in the `google.genai` batch format with `custom_id`, `method=generateContent`, and a `request` payload.
- Request metadata used for normalized CSV output is kept locally in `request_metadata.csv` and joined back during result collection using `custom_id`.
- Text-only requests are sent as a single text prompt with the transcription prepended:
  `Transcription: ...` followed by `Task: ...`
- For local workflow demos without calling Vertex, set `GEMINI_BATCH_STUB_SUBMIT=true` before running `submit-experiment`.
