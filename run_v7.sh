#!/bin/bash
# Resume/continue V7 to 300 epochs with its WeChat monitor. Train_v7.py auto-resumes from the newest
# Checkpoint_v7/model_XXX.pkl. Launch:  tmux new -s train_v7 -d 'bash run_v7.sh'
cd /home/mlp4/data/ZHOU/SSR
export SCT_KEY="$SCT_KEY" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
script=Train_v7.py; log=training_v7.log
touch "$log"
echo "RESUME $(date) (after reboot; last checkpoint $(ls Checkpoint_v7 | sort | tail -1))" | tee -a "$log"
TRAIN_LOG="$PWD/$log" TRAIN_PROC="$script" TOTAL_EPOCHS=300 .venv/bin/python monitor_train.py >> "monitor_v7.log" 2>&1 &
mon=$!
stdbuf -oL -eL .venv/bin/python -u "$script" 2>&1 | tee -a "$log"
echo "TRAIN_EXIT:${PIPESTATUS[0]} $script $(date)" | tee -a "$log"
sleep 400
kill $mon 2>/dev/null
