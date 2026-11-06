import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "constructed" / "constructed_pilot_predictions.csv"

df = pd.read_csv(INPUT)

# Normalize manual 0–5 score to 0–1 for comparison with cosine similarity.
df["human_similarity_norm"] = df["human_similarity_score"] / 5.0

# Positive gap = SimCSE thinks pair is more similar than the manual annotation.
df["overestimate_gap"] = (
    df["simcse_cosine"] - df["human_similarity_norm"]
)

# Category summary
summary = (
    df.groupby("category")
      .agg(
          pairs=("pair_id", "count"),
          mean_human_score=("human_similarity_score", "mean"),
          mean_simcse_cosine=("simcse_cosine", "mean"),
          mean_overestimate_gap=("overestimate_gap", "mean")
      )
      .round(4)
      .sort_values("mean_simcse_cosine", ascending=False)
)

summary.to_csv(
    ROOT / "results" / "constructed" / "constructed_category_summary.csv"
)

# Strongest cases where SimCSE overestimates similarity
failures = (
    df.sort_values("overestimate_gap", ascending=False)
      [["pair_id", "sentence1", "sentence2", "category",
        "human_similarity_score", "simcse_cosine",
        "overestimate_gap"]]
      .head(10)
)

failures.to_csv(
    ROOT / "results" / "constructed" / "constructed_top_failures.csv",
    index=False
)

print("\n===== Category Summary =====")
print(summary)

print("\n===== Top 10 SimCSE Overestimation Examples =====")
print(failures.to_string(index=False))