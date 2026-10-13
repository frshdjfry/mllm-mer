# Reproducibility: what each run records

Maps the items that reproducibility checklists (e.g. NeurIPS, ACL Responsible
NLP) ask for to where this CLI records them. Paths are relative to
`outputs/experiments/<experiment_id>/` unless stated.

| Checklist item | Where it is recorded |
|---|---|
| Exact model and version | `experiment_spec.json` → `model`. Music Flamingo: `id@revision` (pinned Hugging Face commit). Gemini: model resource name, plus `modelVersion` on every line of `results/raw_output.jsonl` (the version Google actually served). |
| Full prompts | `prompt_instances.json` (every rendered prompt, incl. `audio_position`); `requests.jsonl` (exactly what was sent); `prompt_text_hash` in every results row. |
| Decoding settings | `temperature` and `seed` in `requests.jsonl`, `request_metadata.csv` and every results row. Music Flamingo: `results/run_info.json` → `effective_generation_config` (everything `generate()` used, including library defaults such as `top_k`). Gemini: settings not sent are Vertex defaults, stated in `experiment_spec.json` → `unset_generation_settings`. |
| Random seeds | `--seed N`: trial *i* uses `N + i`; recorded per request. Seeds make sampling repeatable only on the same model version, hardware and software (best effort). |
| Number of runs and variability | `--trials`; every trial is a separate results row, so spread (e.g. SD across trials) can be reported. |
| Dates and run times | `submitted_at` (registry and results rows). Music Flamingo: `started_at` / `finished_at` in `results/run_info.json`. |
| Compute | Music Flamingo: GPU name, CUDA, host, Slurm job id in `results/run_info.json`. Gemini: token counts (`usageMetadata`) in `results/raw_output.jsonl`. |
| Software versions | Music Flamingo: `results/environment.txt` (full `pip freeze` of the model's Python) and torch/transformers versions in `run_info.json`. |
| Which code produced the result | `experiment_spec.json` and `run_info.json` → `code_version` (git commit, branch, uncommitted-changes flag). A run with uncommitted changes is not reproducible from the commit alone. |
| Exactly which data | `input_manifest.json`: every file with its path and checksum (`sha256` for local files; GCS `md5_base64` and object `generation` for `gs://` files). |
| Licences | Music Flamingo: NVIDIA OneWay Noncommercial (research only), see README. Dataset licences belong to the project using this CLI. |
| Raw outputs kept unaltered | `results/raw_output.jsonl` is stored as returned; parsing never changes model output (see README). |

Known limits: hosted API models (Gemini) can change or be retired, so exact
re-runs are not possible in the long term; report `modelVersion` and run
dates, and variability across trials. Locally run models (Music Flamingo)
can be re-run from the pinned revision, `environment.txt` and the seed.
