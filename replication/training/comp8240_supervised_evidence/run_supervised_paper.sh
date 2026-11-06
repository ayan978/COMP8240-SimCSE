#!/bin/bash
cd /workspace/SimCSE
START=$(date +%s)

echo "=== SUPERVISED SIMCSE REPRODUCTION ==="
date
echo "Commit: $(git rev-parse HEAD)"

 /venv/simcse38/bin/python train.py \
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

STATUS=$?
END=$(date +%s)

echo "=== EXIT: $STATUS ==="
echo "Elapsed seconds: $((END-START))"
date
exit $STATUS
