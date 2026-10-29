#!/bin/bash
# V7S ablation queue: trains the FixedSigmaMamba arms one after another on one GPU.
#
#   bash run_v7s.sh                                  # selective -> s4 -> const, sigma 4, GPU 0
#   ARMS="s4 const" bash run_v7s.sh                  # a subset
#   GPU=1 ARMS=const bash run_v7s.sh                 # another card (one arm per card is the way to run in parallel)
#   SIGMA=8 ARMS=s4 bash run_v7s.sh                  # sigma sweep
#   tmux new -s v7s -d 'bash run_v7s.sh'             # the usual way: detached, survives logout
#
# Each arm: checkpoints -> Checkpoint_v7s_<mode>_sigma<sigma>/  (auto-resume, Train_v7s.py picks the newest pkl)
#           log         -> training_v7s_<mode>_sigma<sigma>.log
#           WeChat/Bark monitor only when SCT_KEY or BARK_KEY is set (monitor_train.py)
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU:-0}"
PY=${PY:-.venv/bin/python}
ARMS=${ARMS:-"selective s4 const"}
SIGMA=${SIGMA:-4}
EPOCHS=${EPOCHS:-300}

for mode in $ARMS; do
  tag="v7s_${mode}_sigma${SIGMA}"; log="training_${tag}.log"; touch "$log"
  echo "START $(date) $tag gpu=$CUDA_VISIBLE_DEVICES epochs=$EPOCHS" | tee -a "$log"
  mon=
  if [ -n "$SCT_KEY" ] || [ -n "$BARK_KEY" ]; then
    TRAIN_LOG="$PWD/$log" TRAIN_PROC="Train_v7s.py --bc_mode $mode" TOTAL_EPOCHS=$EPOCHS \
      $PY monitor_train.py > "monitor_${tag}.log" 2>&1 &
    mon=$!
  fi
  stdbuf -oL -eL $PY -u Train_v7s.py --bc_mode "$mode" --sigma "$SIGMA" --max_epochs "$EPOCHS" 2>&1 | tee -a "$log"
  echo "TRAIN_EXIT:${PIPESTATUS[0]} $tag $(date)" | tee -a "$log"
  if [ -n "$mon" ]; then sleep 400; kill "$mon" 2>/dev/null; fi   # let the monitor push its final report
done
