import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr, pearsonr

MODEL = "princeton-nlp/sup-simcse-bert-base-uncased"
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "constructed" / "constructed_pilot_annotated.csv"
OUTPUT = ROOT / "results" / "constructed" / "constructed_pilot_predictions.csv"

df = pd.read_csv(INPUT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Pairs:", len(df))

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)
model.eval()

sentences = df["sentence1"].tolist() + df["sentence2"].tolist()

encoded = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    embeddings = model(**encoded, return_dict=True).pooler_output

n = len(df)
e1 = embeddings[:n]
e2 = embeddings[n:]

df["simcse_cosine"] = (
    F.cosine_similarity(e1, e2)
    .cpu()
    .numpy()
)

spearman = spearmanr(
    df["human_similarity_score"],
    df["simcse_cosine"]
)

pearson = pearsonr(
    df["human_similarity_score"],
    df["simcse_cosine"]
)

print("\n===== Constructed Pilot Results =====")
print(f"Pairs: {len(df)}")
print(f"Spearman: {spearman.statistic:.4f}")
print(f"Pearson:  {pearson.statistic:.4f}")

print("\nAverage SimCSE cosine by category:")
print(
    df.groupby("category")["simcse_cosine"]
      .mean()
      .sort_values(ascending=False)
      .round(4)
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)
print("\nSaved predictions to:", OUTPUT)