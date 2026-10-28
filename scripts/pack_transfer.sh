#!/bin/bash
# Copy the non-git part of this project (data, masks, best checkpoints, result cubes) to another server.
#
#   bash scripts/pack_transfer.sh                       # show what would be copied and its size
#   bash scripts/pack_transfer.sh user@host:/path/SSR   # rsync it there (resumable, re-run to continue)
#   FULL=1 bash scripts/pack_transfer.sh user@host:/path/SSR   # also every per-epoch checkpoint (~65 GB)
#
# Code and docs are NOT copied here: clone the GitHub repo on the new server first, then run this
# against the clone directory. See docs/SETUP.md.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${1:-}"
FULL="${FULL:-0}"

# Essentials: ~23 GB (CAVE 9.8 GB, Chikusei 1.3 GB, V7 checkpoints 7.7 GB, test set, masks, final
# checkpoints of the other lines, SSR/Mamba reconstructions, training logs).
INCLUDE=(
  CAVE_1024_28/
  Test_data/
  Chikusei_128/
  mask_256_28.mat mask_128_128.mat mask_256_128.mat
  Checkpoint_pretrain/
  Checkpoint_dwt/model_300.pkl        # HyMCNetL3 (wavelet sampling + spatial Mamba), reference model
  Checkpoint_base/model_300.pkl       # HyMCNetL3_Base (SSR sampling + MamConvB)
  Checkpoint_v4/model_300.pkl
  Checkpoint_v5/model_300.pkl
  Checkpoint_v6/model_300.pkl
  Checkpoint_4stage/model_326.pkl
  Checkpoint_parallel/model_300.pkl
  Checkpoint_v7/                      # spectral-scan line, still training; keep every epoch to resume
  Result/ Label/                      # SSR-S/M/L and early Mamba reconstructions used by make_figure.py
  Result_v3/ Result_v4/ Result_v5/ Result_v6/ Result_base/ Result_freq/ Result_freq2/
  training_*.log monitor*.log
)
if [ "$FULL" = "1" ]; then
  INCLUDE+=(Checkpoint_dwt/ Checkpoint_base/ Checkpoint_v3/ Checkpoint_v4/ Checkpoint_v5/ Checkpoint_v6/
            Checkpoint_4stage/ Checkpoint_parallel/ Result_parallel/ Label_parallel/ HSI_raw/)
fi

LIST=$(mktemp)
for p in "${INCLUDE[@]}"; do
  p="${p%%#*}"; p="${p// /}"          # strip trailing comments / spaces
  [ -z "$p" ] && continue
  for f in $p; do [ -e "$f" ] && echo "$f" >> "$LIST"; done
done

echo "Files/dirs to copy:"; sed 's/^/  /' "$LIST"
echo; echo "Total size:"; du -shc $(cat "$LIST") | tail -1

if [ -n "$DEST" ]; then
  echo; echo "rsync -> $DEST"
  rsync -avh --partial --progress --files-from="$LIST" --relative . "$DEST"
fi
rm -f "$LIST"
