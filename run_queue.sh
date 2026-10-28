#!/bin/bash
# Runs V5 to 300 epochs, then the SSR-skeleton+MamConvB baseline. Each run gets its own WeChat monitor.
cd /home/mlp4/data/ZHOU/SSR
export SCT_KEY="$SCT_KEY" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
for run in "Train_v5.py training_v5.log" "Train_base.py training_base.log"; do
  set -- $run; script=$1; log=$2
  touch "$log"
  TRAIN_LOG="$PWD/$log" TRAIN_PROC="$script" TOTAL_EPOCHS=300 .venv/bin/python monitor_train.py > "monitor_${log%.log}.log" 2>&1 &
  mon=$!
  stdbuf -oL -eL .venv/bin/python -u "$script" 2>&1 | tee -a "$log"
  echo "TRAIN_EXIT:${PIPESTATUS[0]} $script $(date)" | tee -a "$log"
  sleep 400   # let the monitor send its 训练完成 push before it is stopped
  kill $mon 2>/dev/null
done
