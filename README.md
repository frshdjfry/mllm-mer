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

Collect finished results:

```bash
python cli.py collect-results
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
