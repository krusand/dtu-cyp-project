from itertools import product
from subprocess import call, run

config = {
    "readout_lr": [1e-3, 1e-4],
    "batch_size": [16],
    "dropout": [0.1, 0.3, 0.5],
    "encoder_lr": [1e-5, 5e-06],
    "freeze_encoder": [True, False],
    "weight_decay": [1.0e-3, 1.0e-4, 1.0e-5],
    "num_conformers": [1, 2, 5],
    "task": ["regression", "classfication"],
    "readout": ["linear", "mlp", "mlpbn"]
}


keys, values = zip(*config.items())
result = [dict(zip(keys, p)) for p in product(*values)]

for d in result[:5]:
    
    overrides_arg = " ".join([f"{key}={d[key]}" for key in d.keys()])

    bsub_script = f"""
    #!/bin/bash
    #BSUB -q gpuv100
    #BSUB -J cyp_finetune
    #BSUB -n 4
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=1GB]"
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 11:00
    #BSUB -o outputs/cyp_%J.out
    #BSUB -e outputs/cyp_%J.err
    set -euo pipefail
    pwd

    # Setup UV environment
    export UV_CACHE_DIR=".uv_cache"
    uv sync --extra cu124


    # Submit from the starter directory. Set up this venv BEFORE submitting.

    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    # Keep Hugging Face cache consistent with the pre-download command in README.
    export HF_HOME=".hf_cache"

    uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA-enabled PyTorch/GPU required"; print(torch.__version__, torch.version.cuda)'

    # Data must be prepared first with: bsub < prepare_data.sh

    # Results go to runs/<task>/<model>/<timestamp>/ (set in conf/config.yaml).
    uv run --no-sync python -u finetune.py {overrides_arg}
    """

    print(bsub_script)
    run(
        ["bsub"], 
        input=bsub_script,
        text=True,
        capture_output=True,
        check=True
    )
    print("Submitted job")



