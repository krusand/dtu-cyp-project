#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cyp_tune
#BSUB -u s253471@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -o outputs/cyp_%J.out
#BSUB -e outputs/cyp_%J.err
set -euo pipefail
pwd

CACHE_ROOT="${BLACKHOLE:-$PWD}"
export UV_CACHE_DIR="$CACHE_ROOT/.cache/uv"
export HF_HOME="$CACHE_ROOT/.hf_cache"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

uv sync --extra cu124

MODEL="${MODEL:-atom_jepa}"
TASK="${TASK:-regression}"
OVERRIDES="${OVERRIDES:-}"

[[ -f "conf/model/$MODEL.yaml" ]] || { echo "Unknown MODEL=$MODEL" >&2; exit 2; }
case "$TASK" in regression|classification) ;; *) echo "Invalid TASK=$TASK" >&2; exit 2;; esac
nvidia-smi
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA-enabled PyTorch/GPU required"; print(torch.__version__, torch.version.cuda)'

for f in train val test; do
    [[ -f "data/$TASK/$f.csv" ]] || { echo "Missing data/$TASK/$f.csv; run prepare_data.sh first" >&2; exit 2; }
done

if [[ "$TASK" == "classification" ]]; then
    SWEEP_ARGS="model.encoder_lr=1e-5,5e-6 model.readout_lr=1e-3,5e-4 train.batch_size=16 readout.dropout=0.1,0.3"
else
    SWEEP_ARGS="model.encoder_lr=1e-5,5e-6 model.readout_lr=1e-3,5e-4 train.batch_size=16 readout.dropout=0.1,0.2"
fi

uv run --no-sync python -u finetune.py -m \
    model="$MODEL" \
    readout=mlp \
    task="$TASK" \
    seed=42 \
    device=cuda \
    model.precision=fp32 \
    model.freeze_encoder=false \
    train.epochs=20 \
    $SWEEP_ARGS \
    $OVERRIDES