#!/usr/bin/env bash
set -euo pipefail

SIMCSE_DIR="${SIMCSE_DIR:-$HOME/SimCSE}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXPECTED_COMMIT="49fa580a853752ede55b8c76d9debf748e214d3f"

cd "$SIMCSE_DIR"

CURRENT_COMMIT="$(git rev-parse HEAD)"

if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "ERROR: Expected SimCSE commit:"
    echo "$EXPECTED_COMMIT"
    echo "Current commit:"
    echo "$CURRENT_COMMIT"
    exit 1
fi

if [[ ! -f data/nli_for_simcse.csv ]]; then
    echo "ERROR: data/nli_for_simcse.csv not found."
    exit 1
fi

echo "=== SUPERVISED SIMCSE REPRODUCTION ==="
echo "Repository: $SIMCSE_DIR"
echo "Commit: $CURRENT_COMMIT"

"$PYTHON_BIN" train.py \
  --model_name_or_path bert-base-uncased \
  --train_file data/nli_for_simcse.csv \
  --output_dir result/sup-simcse-bert-base-uncased-reproduction \
  --num_train_epochs 3 \
  --per_device_train_batch_size 512 \
  --learning_rate 5e-5 \
  --max_seq_length 32 \
  --evaluation_strategy steps \
  --eval_steps 250 \
  --save_steps 250 \
  --metric_for_best_model stsb_spearman \
  --load_best_model_at_end \
  --pooler_type cls \
  --temp 0.05 \
  --do_train \
  --do_eval \
  --fp16 \
  --seed 42 \
  --overwrite_output_dir
