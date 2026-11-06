import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr, pearsonr

MODEL = "princeton-nlp/sup-simcse-bert-base-uncased"
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "zhao_hu" / "local" / "cw2_dev_clean.csv"
OUTPUT = ROOT / "data" / "zhao_hu" / "local" / "zhao_hu_dev_predictions.csv"

df = pd.read_csv(INPUT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Pairs:", len(df))

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)
model.eval()

scores = []
batch_size = 64

for start in range(0, len(df), batch_size):
    batch = df.iloc[start:start + batch_size]

    s1 = batch["Sent1"].tolist()
    s2 = batch["Sent2"].tolist()
    sentences = s1 + s2

    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        embeddings = model(**encoded, return_dict=True).pooler_output

    n = len(s1)
    e1 = embeddings[:n]
    e2 = embeddings[n:]

    similarities = F.cosine_similarity(e1, e2).cpu().numpy()
    scores.extend(similarities)

    print(f"Processed {min(start + batch_size, len(df))}/{len(df)}")

df["SimCSE_Cosine"] = scores

spearman = spearmanr(df["SimScore"], df["SimCSE_Cosine"])
pearson = pearsonr(df["SimScore"], df["SimCSE_Cosine"])

print("\n===== Zhao & Hu Dev Results =====")
print(f"Pairs: {len(df)}")
print(f"Spearman: {spearman.statistic:.4f}")
print(f"Pearson:  {pearson.statistic:.4f}")

df.to_csv(OUTPUT, index=False)
print(f"Saved predictions to: {OUTPUT}")