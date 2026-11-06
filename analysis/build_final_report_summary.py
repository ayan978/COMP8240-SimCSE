import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_report"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Original-paper replication comparison
# --------------------------------------------------
replication = pd.DataFrame([
    ["Supervised", "STS12", 75.30, 77.67],
    ["Supervised", "STS13", 84.67, 81.55],
    ["Supervised", "STS14", 80.19, 79.48],
    ["Supervised", "STS15", 85.40, 85.30],
    ["Supervised", "STS16", 80.82, 82.17],
    ["Supervised", "STS-B", 84.25, 82.64],
    ["Supervised", "SICK-R", 80.39, 79.60],
    ["Supervised", "Average", 81.57, 81.20],

    ["Unsupervised", "STS12", 68.40, 67.06],
    ["Unsupervised", "STS13", 82.41, 80.35],
    ["Unsupervised", "STS14", 74.38, 71.87],
    ["Unsupervised", "STS15", 80.91, 80.53],
    ["Unsupervised", "STS16", 78.56, 77.28],
    ["Unsupervised", "STS-B", 76.85, 75.88],
    ["Unsupervised", "SICK-R", 72.23, 70.94],
    ["Unsupervised", "Average", 76.25, 74.84],
], columns=["setting", "dataset", "paper", "replicated"])

replication["difference"] = (
    replication["replicated"] - replication["paper"]
)

replication.to_csv(
    OUT / "replication_comparison.csv",
    index=False
)

# --------------------------------------------------
# 2. Zhao & Hu external-dataset result
# --------------------------------------------------
zhao_hu = pd.DataFrame([{
    "dataset": "Zhao & Hu cleaned dev set",
    "n": 2998,
    "spearman": 0.6008,
    "pearson": 0.6759,
}])

zhao_hu.to_csv(
    OUT / "zhao_hu_summary.csv",
    index=False
)

# --------------------------------------------------
# 2. Constructed dataset results
# --------------------------------------------------
constructed = pd.read_csv(
    ROOT / "results" / "constructed" /
    "constructed_final_400_predictions.csv"
)

overall = pd.DataFrame([{
    "dataset": "Constructed controlled dataset",
    "n": len(constructed),
    "spearman": spearmanr(
        constructed["gemini_score"],
        constructed["simcse_cosine"]
    ).statistic,
    "pearson": pearsonr(
        constructed["gemini_score"],
        constructed["simcse_cosine"]
    ).statistic,
    "gemini_mean": constructed["gemini_score"].mean(),
    "simcse_cosine_mean": constructed["simcse_cosine"].mean(),
}])

overall.to_csv(
    OUT / "constructed_overall.csv",
    index=False
)

category = (
    constructed.groupby("category")
    .agg(
        n=("pair_id", "count"),
        gemini_mean=("gemini_score", "mean"),
        gemini_std=("gemini_score", "std"),
        simcse_cosine_mean=("simcse_cosine", "mean"),
        simcse_cosine_std=("simcse_cosine", "std"),
    )
    .reset_index()
)

category.to_csv(
    OUT / "constructed_category_summary.csv",
    index=False
)

# --------------------------------------------------
# 3. Gemini vs manual pilot
# --------------------------------------------------
pilot = constructed[
    constructed["source"] == "pilot_manual"
].copy()

pilot["human_similarity_score"] = pd.to_numeric(
    pilot["human_similarity_score"]
)

agreement = pd.DataFrame([{
    "n": len(pilot),
    "spearman": spearmanr(
        pilot["human_similarity_score"],
        pilot["gemini_score"]
    ).statistic,
    "pearson": pearsonr(
        pilot["human_similarity_score"],
        pilot["gemini_score"]
    ).statistic,
}])

agreement.to_csv(
    OUT / "gemini_manual_agreement.csv",
    index=False
)

# --------------------------------------------------
# 4. Failure cases
# --------------------------------------------------
constructed["gemini_normalized"] = (
    constructed["gemini_score"] / 5.0
)
constructed["diagnostic_gap"] = (
    constructed["simcse_cosine"]
    - constructed["gemini_normalized"]
)

failures = constructed.sort_values(
    "diagnostic_gap",
    ascending=False
).head(15)

failures[
    [
        "pair_id",
        "category",
        "sentence1",
        "sentence2",
        "gemini_score",
        "simcse_cosine",
        "diagnostic_gap",
    ]
].to_csv(
    OUT / "top_failure_cases.csv",
    index=False
)

print("=== FINAL REPORT SUMMARY CREATED ===")
print("\nReplication averages:")
print(
    replication[
        replication["dataset"] == "Average"
    ].to_string(index=False)
)

print("\nConstructed dataset:")
print(overall.round(4).to_string(index=False))

print("\nGemini vs manual pilot:")
print(agreement.round(4).to_string(index=False))

print("\nFiles written to:")
print(OUT)
