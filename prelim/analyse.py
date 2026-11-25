"""Preliminary EMMA comparison (2026-11-25): model vs listener correlations and the human noise ceiling.

Inputs (paths below):
  - lalm-mer-bench EMMA tables copied from rr-pc02: ratings_long.csv (per-listener GEMS-9, scale gems9_max),
    labels_mean.csv (song means)
  - overlap.csv: September Gemini file -> EMMA code -> lalm-mer-bench song_code (168 songs)
  - the two September Gemini parsed_long.csv files, and the two Qwen runs in prelim/runs

Method (written down in docs/decisions.md before results):
  - per GEMS-9 dimension, Spearman correlation across songs between model rating and listener mean
  - 95% bootstrap interval over songs (2,000 draws, seed below)
  - noise ceiling = sqrt(Spearman-Brown(split-half Spearman of listener half-means)), 500 random splits;
    its 95% interval resamples the same songs as the models, with a fresh listener split per draw
  - summaries across dimensions via Fisher z
  - only songs that every model answered are used, so all models are compared on the same songs

    python3 prelim/analyse.py
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
DATA = Path.home() / "mllm-mer-prelim-data"
GEMINI_DIR = Path.home() / "audio-manip-mer/results/gemini/raw/experiments"
GEMINI_RUNS = {
    "gemini-3.8-flash": "emma-youtube-clips__gems_felt_simple_v1__1790523914",
    "gemini-3.1-pro-preview": "emma-youtube-clips__gems_felt_simple_v1__1790525039",
}
DIMS = ["wonder", "transcendence", "tenderness", "nostalgia", "peacefulness",
        "joyful_activation", "power", "sadness", "tension"]
BOOT_DRAWS, SPLIT_DRAWS, SEED = 2000, 500, 20261125
OUT_TABLE = ROOT / "emma_prelim_correlations.csv"
OUT_FIG = ROOT.parent / "docs/figures/emma_prelim_correlations.png"
COLOURS = {"gemini-3.8-flash": "#2a78d6", "gemini-3.1-pro-preview": "#eb6834",
           "qwen3.5-omni-plus-2026-03-15": "#1baf7a", "qwen3.8-omni-flash": "#eda100"}


def spearman_matrix(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def fisher_mean(values) -> float:
    return float(np.tanh(np.nanmean(np.arctanh(np.clip(values, -0.999999, 0.999999)))))


def load_models(overlap: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, song_code) with the nine ratings."""
    to_code = dict(zip(overlap["gemini_file"], overlap["song_code"]))
    frames = []
    for model, run in GEMINI_RUNS.items():
        df = pd.read_csv(GEMINI_DIR / run / "results/parsed_long.csv")
        # file_id was overwritten by the model's own "file_id" answer (parser bug, PR #4): use the audio path.
        df["song_code"] = df["file_uri"].str.extract(r"/([^/]+)\.mp3$")[0].map(to_code)
        df["model"] = model
        frames.append(df)
    for path in glob.glob(str(ROOT / "runs/*/results/parsed_long.csv")):
        df = pd.read_csv(path)
        df["song_code"] = df["file_uri"].str.extract(r"([^/]+)\.mp3$")[0]
        df["model"] = df["model_served"]
        frames.append(df)
    models = pd.concat(frames, ignore_index=True)
    models = models[models["song_code"].isin(overlap["song_code"])].copy()
    for dim in DIMS:
        models[dim] = pd.to_numeric(models[dim], errors="coerce")
    # Some answers are the expected JSON object wrapped in a one-element list ([{...}]), which the parser does
    # not read. Recover those (shape only; values untouched). Anything else stays missing.
    models["recovered_from_list"] = False
    for i in models.index[models[DIMS].isna().any(axis=1)]:
        try:
            answer = json.loads(models.at[i, "raw_response_json"])
        except (TypeError, ValueError):
            continue
        if isinstance(answer, list) and len(answer) == 1 and isinstance(answer[0], dict):
            for dim in DIMS:
                models.at[i, dim] = pd.to_numeric(answer[0].get(dim), errors="coerce")
            models.at[i, "recovered_from_list"] = True
    return models[["model", "song_code", "recovered_from_list", *DIMS]]


def noise_ceiling(ratings: pd.DataFrame, songs: list[str], rng: np.random.Generator) -> dict[str, float]:
    ceilings = {}
    for dim in DIMS:
        per_song = [ratings[(ratings["label"] == dim) & (ratings["item_id"] == s)]["value"].to_numpy()
                    for s in songs]
        halves = []
        for _ in range(SPLIT_DRAWS):
            a, b = [], []
            for values in per_song:
                shuffled = rng.permutation(values)
                middle = len(shuffled) // 2
                a.append(shuffled[:middle].mean())
                b.append(shuffled[middle:].mean())
            halves.append(spearman_matrix(np.array(a), np.array(b)))
        r_half = fisher_mean(halves)
        r_full = 2 * r_half / (1 + r_half)  # Spearman-Brown: reliability of the full listener mean
        ceilings[dim] = float(np.sqrt(max(r_full, 0.0)))
    return ceilings


def ceiling_bootstrap(ratings: pd.DataFrame, songs: list[str], boot_index: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """Ceiling per bootstrap draw: the same song resamples as the models, one fresh listener split per draw.

    The interval therefore covers both which songs were sampled and which way the listeners were split.
    """
    per_dim = []
    for dim in DIMS:
        sub = ratings[(ratings["label"] == dim) & ratings["item_id"].isin(songs)]
        groups = sub.groupby("item_id")["value"].apply(lambda v: v.to_numpy())
        per_dim.append([groups[s] for s in songs])
    out = np.empty((len(boot_index), len(DIMS)))
    for b, idx in enumerate(boot_index):
        for j, per_song in enumerate(per_dim):
            a, c = [], []
            for i in idx:
                shuffled = rng.permutation(per_song[i])
                middle = len(shuffled) // 2
                a.append(shuffled[:middle].mean())
                c.append(shuffled[middle:].mean())
            r_half = spearman_matrix(np.array(a), np.array(c))
            r_full = 2 * r_half / (1 + r_half)
            out[b, j] = np.sqrt(max(r_full, 0.0))
    return out


def main() -> None:
    overlap = pd.read_csv(DATA / "overlap.csv")
    ratings = pd.read_csv(DATA / "human/ratings_long.csv")
    ratings = ratings[ratings["scale"] == "gems9_max"]
    means = pd.read_csv(DATA / "human/labels_mean.csv")
    means = means[means["scale"] == "gems9_max"].pivot(index="item_id", columns="label", values="mean")

    models = load_models(overlap)
    answered = models.dropna(subset=DIMS).drop_duplicates(["model", "song_code"])
    counts = answered.groupby("model")["song_code"].nunique().to_dict()
    model_names = list(COLOURS)
    common = sorted(set.intersection(*[set(answered[answered["model"] == m]["song_code"]) for m in model_names]))
    n_raters = ratings[ratings["item_id"].isin(common)].groupby("item_id")["rater_id"].nunique()
    print(f"Songs answered per model: {counts}")
    print(f"Songs answered by all four models: {len(common)}; listeners per song: "
          f"median {n_raters.median():.0f}, range {n_raters.min()}-{n_raters.max()}")

    rng = np.random.default_rng(SEED)
    human = means.loc[common, DIMS].to_numpy()
    boot_index = rng.integers(0, len(common), size=(BOOT_DRAWS, len(common)))
    rows = []
    for model in model_names:
        table = answered[answered["model"] == model].set_index("song_code").loc[common, DIMS].to_numpy()
        point = [spearman_matrix(table[:, j], human[:, j]) for j in range(len(DIMS))]
        boots = np.array([[spearman_matrix(table[idx, j], human[idx, j]) for j in range(len(DIMS))]
                          for idx in boot_index])
        boot_summary = [fisher_mean(b) for b in boots]
        for j, dim in enumerate(DIMS):
            low, high = np.nanpercentile(boots[:, j], [2.5, 97.5])
            rows.append(dict(model=model, dimension=dim, spearman=point[j], ci_low=low, ci_high=high))
        low, high = np.nanpercentile(boot_summary, [2.5, 97.5])
        rows.append(dict(model=model, dimension="mean (Fisher z)", spearman=fisher_mean(point), ci_low=low, ci_high=high))

    ceilings = noise_ceiling(ratings, common, rng)
    ceiling_boots = ceiling_bootstrap(ratings, common, boot_index, rng)
    for j, (dim, value) in enumerate(ceilings.items()):
        low, high = np.nanpercentile(ceiling_boots[:, j], [2.5, 97.5])
        rows.append(dict(model="human noise ceiling", dimension=dim, spearman=value, ci_low=low, ci_high=high))
    low, high = np.nanpercentile([fisher_mean(b) for b in ceiling_boots], [2.5, 97.5])
    rows.append(dict(model="human noise ceiling", dimension="mean (Fisher z)",
                     spearman=fisher_mean(list(ceilings.values())), ci_low=low, ci_high=high))
    result = pd.DataFrame(rows)
    result["n_songs"] = len(common)
    result.to_csv(OUT_TABLE, index=False, float_format="%.3f")
    all_songs = sorted(overlap["song_code"])
    meta = {
        "n_overlap": len(all_songs), "n_common": len(common),
        "excluded": {s: sorted(m for m in model_names if s not in set(answered[answered["model"] == m]["song_code"]))
                     for s in all_songs if s not in common},
        "answered_per_model": {m: int(counts.get(m, 0)) for m in model_names},
        "recovered_from_list": {m: sorted(answered[(answered["model"] == m) & answered["recovered_from_list"]]["song_code"])
                                for m in model_names},
        "listeners_per_song": {"median": float(n_raters.median()), "min": int(n_raters.min()), "max": int(n_raters.max())},
        "boot_draws": BOOT_DRAWS, "split_draws": SPLIT_DRAWS, "seed": SEED,
    }
    OUT_TABLE.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(result.pivot(index="dimension", columns="model", values="spearman")
          .reindex([*DIMS, "mean (Fisher z)"]).round(2).to_string())

    # --- figure ---------------------------------------------------------------
    labels = [*DIMS, "mean (Fisher z)"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=200)
    ceiling = result[result["model"] == "human noise ceiling"].set_index("dimension").loc[labels, "spearman"]
    ax.bar(x, ceiling, width=0.8, color="#e4e3dd", label="Human noise ceiling (split-half)", zorder=1)
    offsets = np.linspace(-0.27, 0.27, len(model_names))
    for offset, model in zip(offsets, model_names):
        part = result[result["model"] == model].set_index("dimension").loc[labels]
        ax.errorbar(x + offset, part["spearman"], yerr=[part["spearman"] - part["ci_low"], part["ci_high"] - part["spearman"]],
                    fmt="o", ms=5, lw=1.2, capsize=0, color=COLOURS[model], label=model, zorder=3)
    ax.axhline(0, color="#8a8980", lw=0.8)
    ax.axvline(len(DIMS) - 0.5, color="#c3c2b7", lw=0.8, ls=":")
    ax.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=25, ha="right")
    ax.set_ylabel("Spearman ρ with listener mean")
    ax.set_ylim(-0.4, 1.0)
    ax.grid(axis="y", color="#efeee9", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"EMMA, {len(common)} songs (pop + hip-hop): model vs listeners, 95% bootstrap intervals over songs",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, -0.28))
    fig.text(0.01, 0.01, "Preliminary. Gemini: Sept 2026 runs, vendor-default decoding. Qwen: 2026-11-25, temperature 1.0. "
             "Same prompt (gems_felt_simple_v1); Gemini heard Sept MP3 clips, Qwen the lalm-mer-bench excerpts (may differ).", fontsize=7, color="#6b6a62")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    print(f"Wrote {OUT_TABLE} and {OUT_FIG}")


if __name__ == "__main__":
    main()
