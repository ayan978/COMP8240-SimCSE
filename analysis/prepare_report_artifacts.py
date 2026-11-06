from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final_report"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. Headline new-data results table
# --------------------------------------------------
zhao = pd.read_csv(FINAL / "zhao_hu_summary.csv")
constructed_overall = pd.read_csv(FINAL / "constructed_overall.csv")

new_results = pd.DataFrame([
    {
        "dataset": "Zhao & Hu cleaned dev",
        "n": int(zhao.loc[0, "n"]),
        "reference": "human similarity score",
        "spearman": float(zhao.loc[0, "spearman"]),
        "pearson": float(zhao.loc[0, "pearson"]),
    },
    {
        "dataset": "Controlled diagnostic dataset",
        "n": int(constructed_overall.loc[0, "n"]),
        "reference": "Gemini 3.5 Flash-Lite score",
        "spearman": float(constructed_overall.loc[0, "spearman"]),
        "pearson": float(constructed_overall.loc[0, "pearson"]),
    },
])

new_results.to_csv(
    FINAL / "new_data_results_table.csv",
    index=False
)

# --------------------------------------------------
# 2. Zhao & Hu representative data snippets
#    Deterministically select high / medium / low scores.
# --------------------------------------------------
zhao_clean = pd.read_csv(
    ROOT / "data" / "zhao_hu" / "local" / "cw2_dev_clean.csv"
)

targets = [
    ("High similarity", 1.0),
    ("Medium similarity", 0.5),
    ("Low similarity", 0.0),
]

rows = []

for label, target in targets:
    candidates = zhao_clean.copy()
    candidates["distance"] = (candidates["SimScore"] - target).abs()

    # Stable deterministic selection.
    row = (
        candidates.sort_values(
            ["distance", "SimScore", "Sent1", "Sent2"],
            ascending=[True, False, True, True]
        )
        .iloc[0]
    )

    rows.append({
        "example_type": label,
        "sentence1": row["Sent1"],
        "sentence2": row["Sent2"],
        "human_score": row["SimScore"],
    })

zhao_snippets = pd.DataFrame(rows)

zhao_snippets.to_csv(
    FINAL / "zhao_hu_data_snippets.csv",
    index=False
)

# --------------------------------------------------
# 3. Constructed-data representative snippets
# --------------------------------------------------
constructed = pd.read_csv(
    ROOT / "results" / "constructed" /
    "constructed_final_400_predictions.csv"
)

snippet_categories = [
    "paraphrase",
    "synonym_substitution",
    "negation",
    "antonym_substitution",
    "causal_relation_reversal",
    "high_lexical_overlap_different_meaning",
]

constructed_rows = []

for category in snippet_categories:
    subset = (
        constructed[constructed["category"] == category]
        .sort_values("pair_id")
    )

    # First deterministic example from each diagnostic category.
    row = subset.iloc[0]

    constructed_rows.append({
        "pair_id": int(row["pair_id"]),
        "category": category,
        "sentence1": row["sentence1"],
        "sentence2": row["sentence2"],
        "gemini_score": row["gemini_score"],
        "simcse_cosine": row["simcse_cosine"],
    })

constructed_snippets = pd.DataFrame(constructed_rows)

constructed_snippets.to_csv(
    FINAL / "constructed_data_snippets.csv",
    index=False
)

# --------------------------------------------------
# 4. Meaning-preserved vs meaning-changed summary
# --------------------------------------------------
relation_summary = (
    constructed.groupby("expected_relation")
    .agg(
        n=("pair_id", "count"),
        gemini_mean=("gemini_score", "mean"),
        gemini_std=("gemini_score", "std"),
        simcse_cosine_mean=("simcse_cosine", "mean"),
        simcse_cosine_std=("simcse_cosine", "std"),
    )
    .reset_index()
)

relation_summary.to_csv(
    FINAL / "constructed_relation_summary.csv",
    index=False
)

# --------------------------------------------------
# 5. Figure: paper vs fresh reproduction averages
# --------------------------------------------------
rep = pd.read_csv(FINAL / "replication_comparison.csv")
avg = rep[rep["dataset"] == "Average"].copy()

x = np.arange(len(avg))
width = 0.35

fig, ax = plt.subplots(figsize=(6.5, 4.0))

ax.bar(
    x - width / 2,
    avg["paper"],
    width,
    label="Gao et al. (2021)"
)

ax.bar(
    x + width / 2,
    avg["replicated"],
    width,
    label="Fresh reproduction"
)

ax.set_ylabel("Average STS Spearman × 100")
ax.set_xticks(x)
ax.set_xticklabels(avg["setting"])
ax.set_ylim(65, 85)
ax.legend()
ax.grid(axis="y", alpha=0.25)

for xpos, value in zip(x - width / 2, avg["paper"]):
    ax.text(
        xpos, value + 0.25,
        f"{value:.2f}",
        ha="center", va="bottom", fontsize=9
    )

for xpos, value in zip(x + width / 2, avg["replicated"]):
    ax.text(
        xpos, value + 0.25,
        f"{value:.2f}",
        ha="center", va="bottom", fontsize=9
    )

fig.tight_layout()

fig.savefig(
    FIG / "replication_average_comparison.pdf",
    bbox_inches="tight"
)

fig.savefig(
    FIG / "replication_average_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# --------------------------------------------------
# 6. Figure: constructed category comparison
#    Gemini divided by 5 ONLY for common 0-1 visual scale.
# --------------------------------------------------
cat = pd.read_csv(FINAL / "constructed_category_summary.csv")

labels = {
    "paraphrase": "Paraphrase",
    "synonym_substitution": "Synonym",
    "negation": "Negation",
    "named_entity_substitution": "Entity",
    "number_date_substitution": "Number/date",
    "antonym_substitution": "Antonym",
    "causal_relation_reversal": "Causal reversal",
    "high_lexical_overlap_different_meaning": "High-overlap\nrole reversal",
}

order = [
    "paraphrase",
    "synonym_substitution",
    "named_entity_substitution",
    "number_date_substitution",
    "negation",
    "antonym_substitution",
    "causal_relation_reversal",
    "high_lexical_overlap_different_meaning",
]

cat = (
    cat.set_index("category")
       .loc[order]
       .reset_index()
)

cat["gemini_normalized"] = cat["gemini_mean"] / 5.0

x = np.arange(len(cat))
width = 0.38

fig, ax = plt.subplots(figsize=(9.0, 4.6))

ax.bar(
    x - width / 2,
    cat["gemini_normalized"],
    width,
    label="Gemini reference score / 5"
)

ax.bar(
    x + width / 2,
    cat["simcse_cosine_mean"],
    width,
    label="SimCSE cosine similarity"
)

ax.set_ylabel("Mean score on 0–1 scale")
ax.set_ylim(0, 1.08)
ax.set_xticks(x)
ax.set_xticklabels(
    [labels[c] for c in cat["category"]],
    rotation=25,
    ha="right"
)
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    FIG / "constructed_category_comparison.pdf",
    bbox_inches="tight"
)

fig.savefig(
    FIG / "constructed_category_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("=== REPORT ARTIFACTS CREATED ===")

print("\nHeadline new-data results:")
print(new_results.round(4).to_string(index=False))

print("\nZhao & Hu snippets:")
print(zhao_snippets.to_string(index=False))

print("\nConstructed snippets:")
print(
    constructed_snippets[
        [
            "pair_id",
            "category",
            "gemini_score",
            "simcse_cosine"
        ]
    ].round(4).to_string(index=False)
)

print("\nRelation summary:")
print(relation_summary.round(4).to_string(index=False))

print("\nFigures:")
print(FIG / "replication_average_comparison.pdf")
print(FIG / "constructed_category_comparison.pdf")
