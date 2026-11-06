import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Literal

import pandas as pd
from google import genai
from pydantic import BaseModel

MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0
BATCH_SIZE = 10
SHUFFLE_SEED = 8240
MAX_RETRIES = 6

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "constructed" / "constructed_final_400.csv"
OUT_DIR = ROOT / "results" / "constructed"
OUTPUT = OUT_DIR / "gemini_35_flash_lite_judgments_400.csv"
RAW_OUTPUT = OUT_DIR / "gemini_35_flash_lite_judgments_raw.jsonl"
CONFIG_OUTPUT = OUT_DIR / "gemini_35_flash_lite_judge_config.json"

Score = Literal[
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
    3.0, 3.5, 4.0, 4.5, 5.0
]

class ItemJudgment(BaseModel):
    pair_id: int
    score: Score
    rationale: str

class BatchJudgments(BaseModel):
    judgments: list[ItemJudgment]

INSTRUCTIONS = """You are an independent judge of semantic textual similarity.

For EACH sentence pair, rate how similar the MEANING of the two sentences is.

Use exactly one of these scores:
0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0

Scoring anchors:
5.0 = fully equivalent meaning
4.0-4.5 = mostly equivalent, with only minor non-essential differences
3.0-3.5 = substantial shared meaning, but an important detail differs
2.0-2.5 = clearly related or lexically similar, but the core proposition,
          semantic role, relation, entity, quantity, time, or polarity differs
1.0-1.5 = only weakly related in meaning
0.0-0.5 = essentially unrelated or strongly incompatible in meaning

Important:
- Judge semantic meaning, not surface lexical overlap.
- Pay attention to negation, who-did-what-to-whom, causal direction,
  named entities, numbers, dates and antonyms.
- Judge every pair independently.
- Do not compare one pair with another.
- Do not infer hidden dataset labels or categories.
- Give a concise rationale for each score.
"""

def make_prompt(batch):
    parts = [INSTRUCTIONS, "\nSentence pairs:\n"]
    for row in batch:
        parts.append(
            f"\nPair ID: {row['pair_id']}\n"
            f"Sentence 1: {row['sentence1']}\n"
            f"Sentence 2: {row['sentence2']}\n"
        )
    parts.append(
        "\nReturn exactly one judgment for every Pair ID supplied."
    )
    return "".join(parts)

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            'GEMINI_API_KEY is not set. Run: '
            'export GEMINI_API_KEY="PASTE_YOUR_GEMINI_API_KEY_HERE"'
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original = pd.read_csv(INPUT)

    assert len(original) == 400
    assert original["pair_id"].is_unique
    assert original[["sentence1", "sentence2"]].isna().sum().sum() == 0

    if OUTPUT.exists():
        df = pd.read_csv(OUTPUT)
        print(f"Resuming existing output: {OUTPUT}")
    else:
        df = original.copy()
        df["gemini_score"] = pd.NA
        df["gemini_rationale"] = ""
        df["gemini_model"] = ""
        df.to_csv(OUTPUT, index=False)

    # Ensure output still corresponds exactly to the source dataset.
    assert len(df) == 400
    assert df["pair_id"].tolist() == original["pair_id"].tolist()

    config = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "batch_size": BATCH_SIZE,
        "shuffle_seed": SHUFFLE_SEED,
        "score_scale": [
            0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
            3.0, 3.5, 4.0, 4.5, 5.0
        ],
        "category_shown_to_judge": False,
        "expected_relation_shown_to_judge": False,
        "instructions": INSTRUCTIONS,
    }

    with open(CONFIG_OUTPUT, "w") as f:
        json.dump(config, f, indent=2)

    completed = set(
        df.loc[df["gemini_score"].notna(), "pair_id"]
        .astype(int)
        .tolist()
    )

    order = original["pair_id"].astype(int).tolist()
    random.Random(SHUFFLE_SEED).shuffle(order)
    remaining = [pid for pid in order if pid not in completed]

    print(f"Model: {MODEL}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Expected API calls this run: {(len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE}")

    if not remaining:
        print("All 400 pairs are already judged.")
        return

    client = genai.Client(api_key=api_key)

    row_index = {
        int(pid): idx
        for idx, pid in enumerate(df["pair_id"].tolist())
    }

    for start in range(0, len(remaining), BATCH_SIZE):
        batch_ids = remaining[start:start + BATCH_SIZE]

        batch = (
            original[original["pair_id"].isin(batch_ids)]
            .set_index("pair_id")
            .loc[batch_ids]
            .reset_index()
            [["pair_id", "sentence1", "sentence2"]]
            .to_dict("records")
        )

        prompt = make_prompt(batch)

        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config={
                        "temperature": TEMPERATURE,
                        "response_mime_type": "application/json",
                        "response_json_schema":
                            BatchJudgments.model_json_schema(),
                    },
                )

                parsed = BatchJudgments.model_validate_json(
                    response.text
                )

                returned = {
                    int(j.pair_id): j
                    for j in parsed.judgments
                }

                if set(returned) != set(batch_ids):
                    raise ValueError(
                        f"Returned Pair IDs {sorted(returned)} "
                        f"do not match requested IDs {sorted(batch_ids)}"
                    )

                # Save raw API response for auditability.
                with open(RAW_OUTPUT, "a") as f:
                    f.write(json.dumps({
                        "model": MODEL,
                        "pair_ids": batch_ids,
                        "response": json.loads(response.text),
                    }) + "\n")

                # Store individual judgments.
                for pid in batch_ids:
                    judgment = returned[pid]
                    idx = row_index[pid]

                    df.at[idx, "gemini_score"] = float(
                        judgment.score
                    )
                    df.at[idx, "gemini_rationale"] = (
                        judgment.rationale.strip()
                    )
                    df.at[idx, "gemini_model"] = MODEL

                # Checkpoint after every successful API call.
                df.to_csv(OUTPUT, index=False)

                completed_now = int(
                    df["gemini_score"].notna().sum()
                )

                print(
                    f"Completed {completed_now}/400 "
                    f"(batch IDs {batch_ids[0]}...{batch_ids[-1]})"
                )

                success = True
                time.sleep(2)
                break

            except KeyboardInterrupt:
                print(
                    "\nInterrupted safely. Existing judgments "
                    "have already been checkpointed."
                )
                raise

            except Exception as exc:
                wait = min(120, 5 * (2 ** (attempt - 1)))

                print(
                    f"Attempt {attempt}/{MAX_RETRIES} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                if attempt < MAX_RETRIES:
                    print(f"Retrying in {wait} seconds...")
                    time.sleep(wait)

        if not success:
            print(
                "\nStopped after repeated API failures."
                "\nNo completed judgments were lost."
                "\nRun the same command again later to resume."
            )
            sys.exit(1)

    df["gemini_score"] = pd.to_numeric(
        df["gemini_score"], errors="coerce"
    )

    assert df["gemini_score"].notna().sum() == 400
    assert df["gemini_score"].between(0, 5).all()

    df = df.sort_values("pair_id")
    df.to_csv(OUTPUT, index=False)

    print("\n=== GEMINI JUDGING COMPLETE ===")
    print("Rows judged:", df["gemini_score"].notna().sum())
    print("Model:", MODEL)
    print("Output:", OUTPUT)
    print("Raw audit output:", RAW_OUTPUT)
    print("Configuration:", CONFIG_OUTPUT)

    print("\nScore distribution:")
    print(
        df["gemini_score"]
        .value_counts()
        .sort_index()
        .to_string()
    )

if __name__ == "__main__":
    main()
