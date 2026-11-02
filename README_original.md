# CYP Inhibition 🧬 - Starter code

**[Leaderboard](https://khelverskovp.github.io/dtu-cyp-leaderboard/)** · [Atom-JEPA](https://github.com/khelverskovp/atom-jepa) · [Dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)

This repository contains the starter code to get started on the DTU Deep Learning project **Predicting CYP Inhibition with Molecular Foundation Models**.

The data comes from the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge). The challenge ends on November 3, 2026, before this project starts, so we repurpose its training data for this project: we split it into our own train/validation/test sets and run our own leaderboard.

The repo gives you a simple, working fine-tuning setup to build on:

- **Atom-JEPA**: code for loading the pretrained Atom-JEPA model and fine-tuning it with a simple readout head.
- **Four baselines** to compare against: an MLP on Morgan fingerprints, on RDKit descriptors, and on both, plus LightGBM on both.
- **A simple fine-tuning protocol**: fixed train/validation/test splits, one training loop for all models, and the challenge metrics.

Your job is to extend and change it to complete the tasks in the project description.

Along the way you can submit your test predictions to the [project leaderboard](https://khelverskovp.github.io/dtu-cyp-leaderboard/) to see how you rank.

## What is in the repo

| Path | What it does |
|---|---|
| `prepare_data.py` | Downloads the two datasets, generates 3D conformers and splits each into 70/15/15 train/val/test |
| `finetune.py` | Trains one model, picks the best epoch on validation, then evaluates on test |
| `models/atom_jepa.py` | Atom-JEPA encoder + readout head |
| `models/baselines.py` | Morgan / RDKit features + readout head |
| `models/gbm.py` | LightGBM on Morgan + RDKit features |
| `models/readouts.py` | Readout heads (linear, MLP). **Add your own here** |
| `conf/` | [Hydra](https://hydra.cc) configs: `config.yaml` (shared), `model/` (one per model), `readout/` (one per head) |
| `prepare_data.sh`, `finetune.sh` | DTU HPC job scripts. Logs go to `outputs/` |

## Setup

```bash
git clone https://github.com/khelverskovp/dtu-cyp-project.git
cd dtu-cyp-project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Run on DTU HPC

**1. Prepare the data (once).** This writes `data/regression/` and `data/classification/` (each with `train.csv`, `val.csv`, `test.csv`) and takes a few minutes. Open a terminal and run:

```bash
bsub < prepare_data.sh
```

**2. Fine-tune.** Pick the model and task with `-env` (default: Atom-JEPA, regression):

```bash
bsub < finetune.sh
bsub -env "MODEL=morgan_mlp, TASK=classification" < finetune.sh
bsub -env "MODEL=atom_jepa, OVERRIDES=readout=linear train.epochs=10" < finetune.sh
```

`MODEL` is `atom_jepa`, `morgan_mlp`, `rdkit_mlp`, `morgan_rdkit_mlp` or `morgan_rdkit_lgbm`. `TASK` is `regression` or `classification`. `OVERRIDES` takes any Hydra settings, separated by spaces. Check jobs with `bjobs`.

## Tasks and results

There are two separate datasets from the [challenge data](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test), one per task:

- **Regression** (`cyp-challenge-TRAIN_inhibition.csv`): pIC50 for CYP1A2, CYP2C9, CYP2D6 and CYP3A4. Leaderboard score: MA-ST-RAE, the soft-threshold relative absolute error averaged over the four enzymes (lower is better).
- **Classification** (`cyp-challenge-TRAIN_TDI.csv`): time-dependent inhibition (TDI) for CYP2D6 and CYP3A4. Leaderboard score: MCC, averaged over the two enzymes (higher is better).

Many molecules appear in both datasets; they are always in the same split for both tasks, so you can safely combine the tasks. Missing labels are ignored in the loss. Each run is saved in `runs/<task>/<model>/<timestamp>/` with the validation history, the best checkpoint, `test_metrics.json` and `test_predictions.csv`. Use the same splits for every model, choose settings on validation, and use the test split only for the final comparison.

## Leaderboard

The original challenge ends before this project starts, so we run our own leaderboard on this repo's test split: **https://khelverskovp.github.io/dtu-cyp-leaderboard/**. There is one board for regression and one for classification, each showing the best submission per team.

**How to submit**

1. Choose your final model on the **validation** set.
2. Take the `test_predictions.csv` from that run (`runs/<task>/<model>/<timestamp>/`).
3. [Open a submission](https://github.com/khelverskovp/dtu-cyp-leaderboard/issues/new?template=submission.yml) (needs a GitHub account): enter your team name, the task and a short model description, and drag the CSV into the form.
4. A bot comments your score on the issue within a few minutes and updates the leaderboard.

**Rules**

- **Fine-tune and tune on the validation set, never on the test set.** Choose models, hyperparameters and epochs by validation scores only. Submit test predictions only for models you have already chosen.
- **One submission per team per task per day** (Copenhagen time). Rejected submissions do not count.
- **Use the splits from `prepare_data.py` with the default seed.** Don't train on test molecules or look at their labels. `data/<task>/test.csv` contains the labels and the data is public, so the leaderboard relies on your honesty.
- **Use the same team name** for all of your team's submissions.

## Models and readouts

| `model=` | Input | Default readout |
|---|---|---|
| `atom_jepa` | 3D conformer → pretrained Atom-JEPA encoder (fine-tuned) | `mlp` |
| `morgan_mlp` | Morgan count fingerprint | `mlp` |
| `rdkit_mlp` | 200 normalized RDKit descriptors | `mlp` |
| `morgan_rdkit_mlp` | Both, concatenated | `mlp` |
| `morgan_rdkit_lgbm` | Both, concatenated | none (LightGBM) |

The MLP baselines follow those in the [Atom-JEPA repository](https://github.com/khelverskovp/atom-jepa). Any readout works with any neural model, e.g. `model=atom_jepa readout=linear`. `morgan_rdkit_lgbm` is a gradient-boosted tree baseline outside the epoch loop: one LightGBM model per enzyme, with early stopping on validation.

> **Extending the code is the core of the project.** Below is how to add your own readout head and your own model.

### Add your own readout

A readout maps a molecule embedding `[batch, in_dim]` to predictions `[batch, n_targets]`.

1. In `models/readouts.py`, write a class whose `__init__` takes `(in_dim, n_targets, **options)` and whose `forward(x)` returns the predictions. `MLPReadout` is a good template.
2. Register it in the same file: `READOUTS['my_readout'] = MyReadout`.
3. Add `conf/readout/my_readout.yaml` with `type: my_readout` and your options, e.g. `hidden_dim: 256`.
4. Use it with any model: `python finetune.py model=atom_jepa readout=my_readout`, or `bsub -env "MODEL=atom_jepa, OVERRIDES=readout=my_readout" < finetune.sh`.

### Add your own model

A model turns a batch of molecules into embeddings and passes them through a readout.

1. Create `models/my_model.py` with an `nn.Module` class. Copy `models/atom_jepa.py` as a template. It needs:
   - `__init__(cfg, readout, n_targets, data_dir, device)`: load your encoder and build the head with `build_readout(readout, embedding_dim, n_targets)`.
   - `loader(frame, targets, batch_size, shuffle)`: a `DataLoader` whose batches contain your inputs and the labels as `y`.
   - `forward(batch)`: returns predictions `[batch, n_targets]`.
   - `param_groups()`: parameters and learning rates for the optimizer.
   - `fit_preprocessing(train)`: anything fitted on the training set (can be empty).
2. Register it in `models/__init__.py`: `MODELS['my_model'] = MyModel`.
3. Add `conf/model/my_model.yaml`. Copy `conf/model/atom_jepa.yaml` and set `model.name: my_model`, `model.type: my_model`, your model's settings and the `train` settings.
4. Run it like the others: `bsub -env "MODEL=my_model, TASK=regression" < finetune.sh`.

Every model goes through the same training loop, metrics and test evaluation, so your results compare directly with the baselines.

## Notes

- Two molecules (`OCNT-0453746`, `OCNT-0453782`) are excluded because their SMILES have impossible stereochemistry, so no 3D conformer exists. Both only have a negative CYP3A4 TDI label.
- Atom-JEPA runs in fp32 by default, because V100 GPUs do not support bf16 natively. On L40s or A100s, `model.precision=bf16` is faster.
