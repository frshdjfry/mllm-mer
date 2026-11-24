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
