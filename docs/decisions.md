# Decisions log

Dated record of analysis decisions and the reasons for them. Record each decision
before looking at the results it affects. Mark any later change as a deviation
and give the reason.

## 2026-11-25 — Answer columns get an `answer_` prefix; results gain `audio_sha256`

**Status:** in the code (parser.py).

**Decision.** Every key in a model's JSON answer becomes a column named
`answer_<key>` (e.g. `answer_wonder`). Our own columns keep their names.
Prompt variables may not use a results column name or the `answer_` prefix.
Each row also gets `audio_sha256`, the audio file's checksum from
`input_manifest.json`.

**Why.**
- The parser used to copy answer keys into the row under their own names, so
  an answer could overwrite our columns. The old prompts asked Gemini for a
  `"file_id"`; its guesses (e.g. `"audio.mp3"`, `"00:00"`) replaced the real
  `file_id` in every row of 26 Gemini experiments (9,086 of 10,330 rows
  checked on 2026-11-25; scan of `raw_output.jsonl` against
  `request_metadata.csv`). An answer key named like a run setting
  (`temperature`, `seed`, `backend`, …) would instead have hidden the setting.
- One fixed, documented prefix makes clashes impossible and shows at a glance
  which columns the model wrote. Prefixing only clashing keys would have
  kept old column names but made the rule depend on the answer.
- `file_id` and `file_uri` differ between backends and machines for the same
  song; the checksum identifies the exact audio everywhere.

**Unchanged.** `raw_response_json` keeps the answer exactly as returned; key
names are never respelled; `schema_keys` still compares the model's own key
names.

**Existing tables** are not rebuilt yet. Until they are, match on `file_uri`
(as decided in audio-manip-mer on 2026-10-04), not `file_id`.

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
