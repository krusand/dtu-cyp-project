#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cyp_finetune
#BSUB -u s253844@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -o outputs/cyp_%J.out
#BSUB -e outputs/cyp_%J.err
set -euo pipefail
pwd

# Setup UV environment
export UV_CACHE_DIR="${UV_CACHE_DIR:-$BLACKHOLE/dtu-cyp-project/.uv_cache}"
uv sync --extra cu124

# Choose the model and task at submission (LSF only passes variables given with -env):
#   bsub -env "MODEL=morgan_mlp, TASK=classification" < finetune.sh
# MODEL: atom_jepa | morgan_mlp | rdkit_mlp | morgan_rdkit_mlp | morgan_rdkit_lgbm (see conf/model/)
# TASK: regression | classification
# OVERRIDES: extra Hydra overrides separated by spaces, e.g. OVERRIDES="train.epochs=50"
MODEL="${MODEL:-atom_jepa}"
TASK="${TASK:-regression}"
OVERRIDES="${OVERRIDES:-}"

echo "$MODEL"
echo "$TASK"
echo "$OVERRIDES"

# Submit from the starter directory. Set up this venv BEFORE submitting.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Keep Hugging Face cache consistent with the pre-download command in README.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
[[ -f "conf/model/$MODEL.yaml" ]] || { echo "Unknown MODEL=$MODEL" >&2; exit 2; }
case "$TASK" in regression|classification) ;; *) echo "Invalid TASK=$TASK" >&2; exit 2;; esac
nvidia-smi
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA-enabled PyTorch/GPU required"; print(torch.__version__, torch.version.cuda)'

# Data must be prepared first with: bsub < prepare_data.sh
for f in train val test; do
    [[ -f "data/$TASK/$f.csv" ]] || { echo "Missing data/$TASK/$f.csv; run prepare_data.sh first" >&2; exit 2; }
done
# Results go to runs/<task>/<model>/<timestamp>/ (set in conf/config.yaml).
uv run --no-sync python -u finetune.py model="$MODEL" task="$TASK" seed=42 device=cuda $OVERRIDES
