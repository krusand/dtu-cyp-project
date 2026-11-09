from itertools import product
from subprocess import call

config = {
    "readout_lr": [1e-3, 1e-4],
    "batch_size": [16, 32, 64]
}


keys, values = zip(*config.items())
result = [dict(zip(keys, p)) for p in product(*values)]

for d in result:
    cmd = f"uv run python finetune.py +readout_lr={d['readout_lr']} +batch_size={d['batch_size']}"
    cmd_list = cmd.split(" ")
    call(cmd_list)
    exit()


