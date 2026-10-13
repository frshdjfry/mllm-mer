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
- `--seed N` (optional) makes sampling best-effort reproducible: trial `i`
  uses seed `N + i`, so trials still differ from each other but a re-run
  with the same `N` should repeat them. Google does not guarantee identical
  output.

Each parsed result row records `submitted_at`, `temperature`, `seed` and
`prompt_text_hash` (SHA-256 of the rendered prompt text from the prompt
spec, as 64 hex characters; any edit to the prompt changes it). For text
experiments it covers the prompt only, not the transcription appended to
it. If a model answer or prompt variable uses one of these names, the
model's value is kept and the run setting is left out of that column.

The registry gains `temperature` and `base_seed` columns. An older registry
is upgraded on the first submit with this version; after that, older
versions of this CLI cannot read it, so keep using this version.

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
versions, GPU, timings) or `run_failed.txt` with the error. `--temperature 0`
means greedy decoding; above 0 samples at that temperature. The model card's
own example samples at 0.7.

The registry gains a `backend` column (`vertex` for existing rows); as with
the earlier columns, an older registry is upgraded once and older versions of
this CLI cannot read it afterwards.

Offline tests (no GPU, Slurm or network): `python -m unittest discover -s tests`.
`tests/test_formats.py` locks every file format against golden files in
`tests/golden/`; after an intentional format change, regenerate them with
`UPDATE_GOLDEN=1 python -m unittest tests.test_formats` and review the diff.
Prompt specs may set `audio_position: before` (default) or `after`, and both
backends send the parts in that order.

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
