from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "zhao_hu" / "local"

REQUIRED_COLUMNS = ["Sent1", "Sent2", "SimScore"]

EXPECTED = {
    "train": {"raw": 11498, "duplicates": 40, "clean": 11458},
    "dev": {"raw": 3000, "duplicates": 2, "clean": 2998},
}

def prepare(split):
    input_path = DATA_DIR / f"cw2_{split}.csv"
    output_path = DATA_DIR / f"cw2_{split}_clean.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing: {input_path}")

    df = pd.read_csv(input_path)
    raw_rows = len(df)

    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"{split}: expected columns {REQUIRED_COLUMNS}, "
            f"found {df.columns.tolist()}"
        )

    missing_values = int(df.isna().sum().sum())
    if missing_values != 0:
        raise ValueError(f"{split}: found {missing_values} missing values")

    duplicate_rows = int(df.duplicated().sum())

    clean = df.drop_duplicates().reset_index(drop=True)

    expected = EXPECTED[split]

    assert raw_rows == expected["raw"]
    assert duplicate_rows == expected["duplicates"]
    assert len(clean) == expected["clean"]

    clean.to_csv(output_path, index=False)

    print(f"{split}:")
    print(f"  raw rows:       {raw_rows}")
    print(f"  duplicates:     {duplicate_rows}")
    print(f"  clean rows:     {len(clean)}")
    print(f"  missing values: {missing_values}")
    print(f"  saved:          {output_path}")

for split in ["train", "dev"]:
    prepare(split)
