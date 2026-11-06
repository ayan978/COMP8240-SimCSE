import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr, pearsonr

MODEL = "princeton-nlp/sup-simcse-bert-base-uncased"

ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / "results" / "constructed" /
    "gemini_35_flash_lite_judgments_400.csv"
)
OUTPUT = (
    ROOT / "results" / "constructed" /
    "constructed_final_400_predictions.csv"
)
SUMMARY = (
    ROOT / "results" / "constructed" /
    "constructed_final_category_summary.csv"
)

df = pd.read_csv(INPUT)

assert len(df) == 400
assert df["gemini_score"].notna().all()
assert df["pair_id"].is_unique

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print("Model:", MODEL)
print("Device:", device)
print("Pairs:", len(df))

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)
model.eval()

sentences = (
    df["sentence1"].tolist() +
    df["sentence2"].tolist()
)

encoded = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    embeddings = model(
        **encoded,
        return_dict=True
    ).pooler_output

n = len(df)
e1 = embeddings[:n]
e2 = embeddings[n:]

df["simcse_cosine"] = (
    F.cosine_similarity(e1, e2)
    .cpu()
    .numpy()
)

# Main evaluation against Gemini reference labels.
spearman = spearmanr(
    df["gemini_score"],
    df["simcse_cosine"]
)

pearson = pearsonr(
    df["gemini_score"],
    df["simcse_cosine"]
)

print("\n===== FINAL CONSTRUCTED DATASET =====")
print("Pairs:", len(df))
print(f"Spearman: {spearman.statistic:.4f}")
print(f"Pearson:  {pearson.statistic:.4f}")

print("\n===== CATEGORY RESULTS =====")

rows = []

for category, g in df.groupby("category"):
    if (
        g["gemini_score"].nunique() > 1
        and g["simcse_cosine"].nunique() > 1
    ):
        sp = spearmanr(
            g["gemini_score"],
            g["simcse_cosine"]
        ).statistic

        pr = pearsonr(
            g["gemini_score"],
            g["simcse_cosine"]
        ).statistic
    else:
        sp = float("nan")
        pr = float("nan")

    rows.append({
        "category": category,
        "n": len(g),
        "gemini_mean": g["gemini_score"].mean(),
        "simcse_cosine_mean": g["simcse_cosine"].mean(),
        "spearman": sp,
        "pearson": pr,
    })

summary = pd.DataFrame(rows).sort_values(
    "simcse_cosine_mean",
    ascending=False
)

print(summary.round(4).to_string(index=False))

print("\n===== EXPECTED RELATION =====")
print(
    df.groupby("expected_relation")
      .agg(
          n=("pair_id", "count"),
          gemini_mean=("gemini_score", "mean"),
          simcse_cosine_mean=("simcse_cosine", "mean")
      )
      .round(4)
)

# Diagnostic only: not treated as a calibrated error metric.
df["gemini_normalized"] = df["gemini_score"] / 5.0
df["diagnostic_gap"] = (
    df["simcse_cosine"] - df["gemini_normalized"]
)

print("\n===== TOP 10 HIGH-COSINE / LOW-JUDGE FAILURES =====")

failures = (
    df.sort_values("diagnostic_gap", ascending=False)
      .head(10)
)

for _, r in failures.iterrows():
    print(
        f"\nPair {int(r['pair_id'])} | {r['category']}"
        f"\nGemini: {r['gemini_score']:.1f}/5"
        f" | SimCSE cosine: {r['simcse_cosine']:.4f}"
        f"\nS1: {r['sentence1']}"
        f"\nS2: {r['sentence2']}"
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT, index=False)
summary.to_csv(SUMMARY, index=False)

print("\nSaved predictions:", OUTPUT)
print("Saved category summary:", SUMMARY)
