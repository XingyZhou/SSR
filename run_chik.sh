#!/bin/bash
# 128-band (Chikusei) spectral-scan line. Runs the spectral-scan arm, then the spatial-scan control,
# each with its own WeChat monitor. Launch inside tmux once the GPU is free (V7 uses ~11 GB; each run
# below needs ~11 GB at batch 2 / 128x128, or ~6 GB at batch 1):
#   tmux new -s train_chik -d 'bash run_chik.sh'
cd /home/mlp4/data/ZHOU/SSR
export SCT_KEY="$SCT_KEY" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
EPOCHS=${EPOCHS:-100}   # 128-band epochs take ~40-50 min; 100 epochs ~= 3 days. EPOCHS=300 bash run_chik.sh for the CAVE schedule
COMMON="--data_path ./Chikusei_128/ --bands 128 --size 128 --mask_path ./mask_128_128.mat --batch_size 2 --max_epochs $EPOCHS --sched_epochs $EPOCHS"
for run in "HyMCNetL3_V7NB Checkpoint_chik_v7nb training_chik_v7nb.log" "HyMCNetL3_NB Checkpoint_chik_nb training_chik_nb.log"; do
  set -- $run; model=$1; ckpt=$2; log=$3
  touch "$log"
  TRAIN_LOG="$PWD/$log" TRAIN_PROC="Train_nb.py --model $model" TOTAL_EPOCHS=$EPOCHS .venv/bin/python monitor_train.py > "monitor_${log%.log}.log" 2>&1 &
  mon=$!
  stdbuf -oL -eL .venv/bin/python -u Train_nb.py --model $model --ckpt_dir ./$ckpt $COMMON 2>&1 | tee -a "$log"
  echo "TRAIN_EXIT:${PIPESTATUS[0]} $model $(date)" | tee -a "$log"
  sleep 400
  kill $mon 2>/dev/null
done
# test afterwards (256x256 tiles, 21 scenes):
#   .venv/bin/python Test_nb.py --model HyMCNetL3_V7NB --data_path ./Chikusei_128/ --bands 128 --size 256 \
#       --mask_path ./mask_256_128.mat --ckpt_dir ./Checkpoint_chik_v7nb --out_dir ./Result_chik_v7nb --epochs 291-300
