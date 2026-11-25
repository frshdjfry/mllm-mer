# Decisions log

Dated record of analysis decisions and the reasons for them. Record each decision
before looking at the results it affects. Mark any later change as a deviation
and give the reason.

## 2026-11-24 — Closed (API) models run at their vendor's default decoding settings

**Status:** agreed, not yet in the code.

**Decision.** For closed models served through a vendor API (Gemini, Qwen Omni,
and any added later), mllm-mer sends no decoding settings. It omits
`temperature`, `top_p`, `top_k`, presence and repetition penalties, and `seed`,
so each model runs as its vendor serves it by default.

**Why.**
- We can't fully control decoding on a closed model. Vendors apply settings we
  never send. Examples from Alibaba's docs:
  - Qwen Omni applies a presence penalty (1.5 for Qwen3.5-Omni).
  - qwen3.8-omni-flash thinks by default.
  - The docs suggest further defaults (top_k 20, repetition penalty 1.05); these
    are inferred, not stated for these models.
  - Setting only `temperature` creates a mix nobody designed or tested: our value
    plus the vendor's hidden defaults. For example, our CLI default of 1.0 is
    above Qwen3.5-Omni's documented default of 0.7.
- The research question becomes clear and realistic: how does each model rate
  music emotion as its vendor serves it?
- One rule covers every closed model.
- Fixing settings doesn't buy repeatability. On 2026-11-24, temperature 0 on
  Alibaba's API gave a different answer every time it was repeated:
  - EMMA: 3 clips, 2 runs each, on qwen3.5-omni-plus and qwen3.8-omni-flash.
  - A text-only test on qwen3.5-omni-plus: 4 different answers in 5 repeats.
- A seed doesn't buy repeatability either. Both EMMA temperature-0 runs sent
  `seed: 0` and still differed on every clip. Alibaba promises only results that
  match "as much as possible" (from a summary of the docs, not a direct quote).
  So `seed` is left out like the other settings.

**What each run must record**, because defaults are invisible and can change
without notice:
- the run date;
- the model version the API reports serving;
- the vendor's documented defaults on that date, with the URL of the page they
  came from. Mark any value that is inferred rather than stated for the model.
  Sources for Qwen:
  - https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions
  - https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope

  So far these values come from a research agent's quotes. Check them against
  the live pages before citing them.

**Limitations to state in write-ups.**
- Models are compared at different decoding settings, each at its own vendor's
  default. The exact values are not reported by the APIs: Alibaba's
  `system_fingerprint` is always null. So the methods section can only cite the
  documented defaults on the run date.

**Not covered by this entry:**
- open-weight models run on our own hardware (e.g. Music Flamingo), which have no
  vendor default;
- non-decoding request options, such as Flash's thinking mode and JSON mode.

**Code changes needed:**
- `--temperature` must stop defaulting to 1.0 and must not be sent unless given.
- `--temperature` and `--top-p` may stay as explicit overrides for experiments.
  Any run that uses them deviates from this decision and must be logged here.
- These are shared CLI and backend files. Make the change on top of `master`
  after the pending backend PRs are merged, not only on `qwen-backend`.

## 2026-11-24 — Open-weight models run at their own published generation settings

**Status:** agreed. The runner supports it now (requests without a
temperature); the CLI still always sends one until the vendor-defaults change
above is made.

**Decision.** For open-weight models that we run ourselves (Music Flamingo),
mllm-mer chooses no decoding settings. When a request has no temperature,
the model uses the generation settings its developers publish with the
weights (`generation_config.json`).

**Why.** This is the same principle as for closed models: each model runs as
its developers intend, not at a value we picked. An open model has no vendor
API to apply defaults, so its published settings file plays that role.

**What it means for Music Flamingo.** At the pinned revision
(`nvidia/music-flamingo-hf@e560c33247f20e2f735c87ce9adebb8e53befe69`), the
file sets no sampling options, so the Hugging Face software uses greedy
decoding (checked 2026-11-24). Its `max_new_tokens` (2048) is overridden by
our backend config (1024). The settings actually used are recorded per run in
`results/run_info.json` (`effective_generation_config`, key `model_default`).

**Note.** The model card's own example (as quoted in our README, not rechecked) samples at temperature 0.7 and top_p
0.9. That example is not the published default, so we don't follow it.

## 2026-11-25 — Preliminary EMMA comparison: Qwen vs September Gemini (written before results)

**Status:** preliminary, for the 2026-11-25 supervision. Not the benchmark run.

**Songs.** The 168 EMMA songs in both (a) the 2026-09-27 Gemini runs
(audio-manip-mer `emma-youtube-clips__gems_felt_simple_v1__1790523914`,
gemini-3.8-flash, and `__1790525039`, gemini-3.1-pro-preview; 176 files, of
which 173 match the EMMA table) and (b) the lalm-mer-bench EMMA set with audio
(266 songs). Matched via OSF `song_overview.csv` (`code` → `song_code`).
106 pop, 62 hip-hop, no classical (the September set had none).

**Audio.** Qwen hears the lalm-mer-bench WAVs (rr-pc02,
`/Data/lyl31/lalm-mer-bench-data/emma/audio`), transcoded to 192 kbps MP3
(ffmpeg n7.1.3, libmp3lame) because the 32-bit float WAVs exceed Model
Studio's 10 MB inline limit. Copies with SHA256SUMS:
`/Data/lyl31/lalm-mer-bench-data/emma/prelim_mp3_192k`. Gemini heard the
September MP3 clips, which may not be the same excerpt; so the two model
families did not hear identical audio.

**Prompt.** The exact September text and schema
(`prelim/gems_felt_simple_v1_sept2026.yaml`), one prompt with all nine
dimensions (not templated), one trial per song.

**Models.** `qwen3.5-omni-plus-2026-03-15`, `qwen3.8-omni-flash`.

**Deviation from the vendor-defaults decision (2026-11-24).** Qwen requests
send temperature 1.0 (the CLI still always sends it). The September Gemini
requests sent no temperature (vendor default).

**Metrics.** Per GEMS-9 dimension: Spearman correlation across songs between
model rating and the mean listener rating (lalm-mer-bench `ratings_long.csv`,
scored to GEMS-9), 95% bootstrap interval over songs. Human reference: noise
ceiling = sqrt of the Spearman–Brown-corrected split-half reliability of the
listener mean (random rater splits per song, fixed seed). Summary across
dimensions via Fisher z.

**Addition (2026-11-25, before the final analysis, after seeing preview numbers):** the noise ceiling also
gets a 95% interval, bootstrapping the same song resamples as the models, with one fresh random listener
split per draw. Asked for so the ceiling's uncertainty is shown like the models'; it does not change any
point estimate.

**Addition (2026-11-25, after the runs, before the final analysis):** 7 qwen3.8-omni-flash answers (H032,
H041, H105, H117, P060, P087, P123) are the requested JSON object wrapped in a one-element list
(`[{...}]`), which the parser does not read. The analysis unwraps such answers (shape only; no value is
changed) and reports which songs were recovered. P004's answer is an empty list and counts as no answer.
Proposed parser change for mllm-mer: accept a one-element list holding the object.

**Addition (2026-11-25, after the final analysis):** each failed (song, model) request was retried once
(`prelim/retry/`). qwen3.8-omni-flash answered H047 and P004 on the retry; H022 and H119 (both Qwen
models) and P098 (Flash) failed again. Retry answers are reported on the dashboard but **not** used in
the correlations, which keep one attempt per song as planned.
