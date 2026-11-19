from itertools import product
from subprocess import run
import textwrap


config_atom_tier1 = {
    "model": ["atom_jepa"],
    "model.readout_lr": [1e-3, 1e-4, 1e-5, 1e-6],
    "train.batch_size": [16],
    "readout.dropout": [0.1, 0.3, 0.5],
    "model.encoder_lr": [1e-4, 1e-5, 1e-6],
    "task": ["classification", "regression"],
}


best_atom_tier1_regression = {'model.encoder_lr': 0.0001, 'model.readout_lr': 0.0001, 'train.batch_size': 16, 'readout.dropout': 0.1}
# {'model.encoder_lr': 1e-05,  'model.readout_lr': 1e-06,  'train.batch_size': 16, 'readout.dropout': 0.3}
# {'model.encoder_lr': 1e-05,  'model.readout_lr': 1e-05,  'train.batch_size': 16, 'readout.dropout': 0.3}


config_atom_tier2 = {
    "model": ["atom_jepa"],
    "model.readout_lr": [best_atom_tier1_regression["model.readout_lr"]],
    "train.batch_size": [best_atom_tier1_regression["train.batch_size"]],
    "readout.dropout": [best_atom_tier1_regression["readout.dropout"]],
    "model.encoder_lr": [best_atom_tier1_regression["model.encoder_lr"]],
    "model.freeze_encoder": [True, False],
    "train.weight_decay": [1.0e-3, 1.0e-4, 1.0e-5],
    "model.num_conformers": [1, 2, 5],
    "readout": ["linear", "mlp", "mlpbn"],
    "task": ["regression"],
}

config_chemberta_tier_1 = {
    "model": ["chemberta"],
    "model.readout_lr": [1e-3, 1e-4, 1e-5, 1e-6],
    "train.batch_size": [16],
    "readout.dropout": [0.1, 0.3, 0.5],
    "model.encoder_lr": [1e-3, 1e-4, 1e-5, 1e-6],
    "task": ["regression", "classification"],
}

# Testing tier 2 with the found best hyperparameters from tier 1.
config_chemberta_tier_2 = {
    "model": ["chemberta"],
    "model.readout_lr": [1e-3],
    "readout.dropout": [0.1],
    "model.encoder_lr": [1e-4],
    "train.batch_size": [8, 16, 32],
    "model.freeze_encoder": [True, False],
    "train.weight_decay": [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6],
    "readout": ["linear", "mlp"],
    "task": ["classification"],
}

config = config_atom_tier2

keys, values = zip(*config.items())
result = [dict(zip(keys, p)) for p in product(*values)]

print(f"Len of the result {len(result)}")

for i, d in enumerate(result):

    overrides_arg = " ".join([f"{key}={d[key]}" for key in d.keys()])

    num_conformers_to_wall_clock = {
        1: '3:00',
        2: '6:00',
        5: '8:00'
    }

    bsub_script = textwrap.dedent(f"""\
    #!/bin/bash
    #BSUB -q gpuv100
    #BSUB -J cyp_finetune_{i}
    #BSUB -n 4
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=4GB]"
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W {num_conformers_to_wall_clock[d["model.num_conformers"]]}
    #BSUB -o outputs/cyp_{i}_%J.out
    #BSUB -e outputs/cyp_{i}_%J.err
    set -euo pipefail
    pwd

    # Setup UV environment
    export UV_CACHE_DIR=".uv_cache"
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv
    uv sync --extra cu124


    # Submit from the starter directory. Set up this venv BEFORE submitting.
    flock .venv.lock uv sync --extra cu124 --frozen

    # Keep Hugging Face cache consistent with the pre-download command in README.
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export HF_HOME=".hf_cache"

    uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA-enabled PyTorch/GPU required"; print(torch.__version__, torch.version.cuda)'

    # Results go to runs/<task>/<model>/<timestamp>/ (set in conf/config.yaml).
    uv run --no-sync python -u finetune.py {overrides_arg} device=cuda frac_data=1
    # uv run --no-sync python -u finetune.py {overrides_arg} device=cuda frac_data=1 hydra.run.dir=runs/chemberta_hparam/tier2_bs16/{d['task']}/job_{i}
    """)

    print(bsub_script)

    
    r = run(
        ["bsub"], 
        input=bsub_script,
        text=True,
        capture_output=True,
        check=True
    )
    print(r.stdout.strip())
    print("Submitted job")

