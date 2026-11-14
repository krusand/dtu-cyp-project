#!/usr/bin/env python3
"""Fine-tune one model (Atom-JEPA or a baseline) for regression OR classification.

Configured with Hydra (conf/), e.g. python finetune.py model=morgan_mlp task=classification"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import matthews_corrcoef, mean_absolute_error, r2_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import os

from models import MODELS
from models.gbm import LightGBMModel
from prepare_data import REG, CLS


def masked_loss(pred, y, task, mean, std, model_name):
    mask = torch.isfinite(y)
    if not mask.any():
        return None
    # Index BEFORE applying the loss: multiplying NaN losses by zero is not safe.
    if task == 'regression':
        return nn.functional.mse_loss(pred[mask], ((y - mean) / std)[mask])
    return nn.functional.binary_cross_entropy_with_logits(pred[mask], y[mask])


def metrics(frame, predictions, targets, task):
    result = {}
    
    for j, target in enumerate(targets):
        print(f"{target=}")
        y = frame[target].to_numpy(dtype=float)
        mask = np.isfinite(y)
        y, pred = y[mask], predictions[mask, j]
        print(f"{y=}")
        print(f"{pred=}")
        if not len(y):
            print(f"{not len(y) = }")
            result[target] = {'n': 0, ('ST-RAE' if task == 'regression' else 'MCC'): None}
            continue
        if task == 'regression':
            print("Task = regression")
            low = frame[target + '_conf_low'].to_numpy(dtype=float)[mask]
            high = frame[target + '_conf_high'].to_numpy(dtype=float)[mask]
            if not (np.isfinite(low).all() and np.isfinite(high).all() and (low <= high).all()):
                raise ValueError(f'Invalid credible intervals for {target}')
            distance = lambda p: np.maximum(low - p, 0) + np.maximum(p - high, 0)
            denom = distance(y.mean()).sum()
            result[target] = {'MAE': float(mean_absolute_error(y, pred)),
                              'R2': float(r2_score(y, pred)) if len(y) > 1 else None,
                              'ST-RAE': float(distance(pred).sum() / denom) if denom > 0 else None,
                              'n': len(y)}
        else:
            both = len(np.unique(y)) == 2
            result[target] = {'MCC': float(matthews_corrcoef(y, pred >= .5)),
                              'AUROC': float(roc_auc_score(y, pred)) if both else None,
                              'AP': float(average_precision_score(y, pred)) if both else None,
                              'n': len(y), 'positive_fraction': float(y.mean())}
    primary = 'ST-RAE' if task == 'regression' else 'MCC'
    scores = [result[t][primary] for t in targets]
    # Do not silently omit an endpoint when reporting a macro score.
    print(f"{result=}")
    for s in scores:
        print(f"{s=}")
    result['macro_' + primary] = (float(np.mean(scores))
        if all(s is not None and np.isfinite(s) for s in scores) else None)
    return result


@torch.no_grad()
def predict(model, loader, task, mean, std, cfg):
    model.eval()
    out = []
    num_conformers = cfg.model.num_conformers
    print(f"{num_conformers=}")
    for batch in loader:
        num_conformers_in_batch = batch['atomic_numbers'].shape[1]
        conf_preds = []
        for c in range(num_conformers):
            sub_batch = dict(batch)
            sub_batch["atomic_numbers"] = batch["atomic_numbers"][:, c]
            sub_batch["node_coordinates"] = batch["node_coordinates"][:, c*3:(c*3+3)]
            
            if 'edge_vectors' in batch and batch['edge_vectors'].shape[1] == num_conformers_in_batch * 3:
                sub_batch['edge_vectors'] = batch['edge_vectors'][:, c*3:(c*3+3)]
            if 'edge_lengths' in batch and batch['edge_lengths'].ndim == 2:
                sub_batch['edge_lengths'] = batch['edge_lengths'][:, c]
            
            p = model(sub_batch)
            p = p * std + mean if task == 'regression' else p.sigmoid()
            print(f"{p=}")
            conf_preds.append(p)
        p_stack = torch.stack(conf_preds, dim=0)
        p_avg = p_stack.mean(dim=0)
        print(f"{p_stack=}")
        print(f"{p_avg=}")
        out.append(p_avg.float().cpu().numpy())
    values = np.concatenate(out)
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite predictions')
    print(f"{values=}")
    return values


def train_torch(cfg, train, val, targets, task, data, out):
    """Trains a torch model with the epoch loop; returns a function frame -> predictions."""
    if cfg.train.epochs < 1 or cfg.train.batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    model = MODELS[cfg.model.type](cfg.model, cfg.readout, len(targets), data, cfg.device, task)
    model.fit_preprocessing(train)  # Feature scaling etc. is fitted on the training split ONLY.
    mean = torch.zeros(len(targets), device=model.device)
    std = torch.ones_like(mean)
    if task == 'regression':
        # Fit scaling on the training fold ONLY, separately for each observed endpoint.
        mean = torch.tensor(train[targets].mean().to_numpy(), dtype=torch.float32, device=model.device)
        std = torch.tensor(train[targets].std(ddof=0).to_numpy(), dtype=torch.float32,
                           device=model.device).clamp_min(1e-6)
    batch_size = cfg.train.batch_size
    train_loader = model.loader(train, targets, batch_size, shuffle=True)
    val_loader = model.loader(val, targets, batch_size)
    optimizer = torch.optim.AdamW(model.param_groups(), weight_decay=cfg.train.weight_decay)
    best = float('inf')
    for epoch in tqdm(range(1, cfg.train.epochs + 1)):
        model.train()
        loss_sum, observed = 0., 0
        for batch in tqdm(train_loader):
            y = batch['y'].to(model.device)
            loss = masked_loss(model(batch).float(), y, task, mean, std, cfg.model.name)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],
                                     cfg.train.grad_clip)
            optimizer.step()
            n = int(torch.isfinite(y).sum())
            loss_sum += loss.item() * n
            observed += n
        score = metrics(val, predict(model, val_loader, task, mean, std, cfg), targets, task)
        print(f"{score=}")
        primary = score['macro_ST-RAE' if task == 'regression' else 'macro_MCC']
        if primary is None:
            raise ValueError('Validation metric undefined; check endpoint coverage and split')
        selection = primary if task == 'regression' else -primary
        record = {'epoch': epoch, 'train_loss': loss_sum / observed, **score}
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if selection < best:
            best = selection
            torch.save({'model': model.state_dict(), 'mean': mean, 'std': std,
                        'epoch': epoch, 'config': cfg_dict(cfg)}, out / 'best.pt')
        if cfg.device == 'cuda':
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(f"Peak tensor memory: {peak_gb:.2f} GB | Peak reserved by PyTorch: {reserved_gb:.2f} GB")

    saved = torch.load(out / 'best.pt', map_location=model.device, weights_only=True)
    model.load_state_dict(saved['model'])
    mean, std = saved['mean'], saved['std']
    print(f'Best epoch: {saved["epoch"]}', flush=True)
    return lambda frame: predict(model, model.loader(frame, targets, batch_size), task, mean, std, cfg)


def cfg_dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True)

def is_num_conformers_larger_than_num_conformers_in_data(cfg) -> bool:
    key = os.listdir(Path(cfg.data) / "conformers")[0]
    with np.load(Path(cfg.data) / "conformers" / key) as d:
        num_conformers_in_file = d['z'].shape[1]
    return True if cfg.model.num_conformers > num_conformers_in_file else False

def load_data(cfg, split):
    data = pd.read_csv(Path(cfg.data) / cfg.task / f'{split}.csv')
    n = len(data)
    return data.iloc[:int(n*cfg.frac_data)]



@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def main(cfg):
    print(cfg)
    if cfg.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if cfg.task not in ('regression', 'classification'):
        raise ValueError(f'Unknown task {cfg.task!r}')
    if is_num_conformers_larger_than_num_conformers_in_data(cfg):
        raise ValueError(f'num_conformers larger than number of conformers in data, rerun prepare_data.py with more conformers')
    
    out = Path(HydraConfig.get().runtime.output_dir)
    data = Path(cfg.data)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    task = cfg.task
    targets = REG if task == 'regression' else CLS
    train_all = load_data(cfg, 'train')
    val_all = load_data(cfg, 'val')
    frame = pd.concat([train_all, val_all], ignore_index=True)
    train = train_all[train_all[targets].notna().any(axis=1)].copy()
    val = val_all[val_all[targets].notna().any(axis=1)].copy()
    if not len(train) or not len(val):
        raise ValueError('Empty train or validation set')
    if set(train_all.canonical_smiles) & set(val_all.canonical_smiles):
        raise ValueError('Exact molecule leakage between train and validation')
    for t in targets:
        if not train[t].notna().any() or not val[t].notna().any():
            raise ValueError(f'Missing train/validation labels for {t}')
        if task == 'classification':
            if not set(frame[t].dropna()).issubset({0, 1}):
                raise ValueError(f'Nonbinary labels: {t}')
            if train[t].nunique() != 2 or val[t].nunique() != 2:
                raise ValueError(f'{t} needs both classes in train and validation; revise split')
    config = cfg_dict(cfg)
    packages = ['torch', 'atom-jepa', 'rdkit', 'numpy', 'pandas', 'scikit-learn', 'hydra-core',
                'descriptastorus', 'lightgbm']
    config.update(targets=targets, output_dir=str(out),
                  versions={k: importlib.metadata.version(k) for k in packages},
                  preparation=json.loads((data / 'manifest.json').read_text()),
                  split_sha256={name: hashlib.sha256((data / task / f'{name}.csv').read_bytes()).hexdigest()
                                for name in ['train', 'val', 'test']})
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    frame[['Molecule_Name', 'key', 'split']].to_csv(out / 'split.csv', index=False)
    # Sanity baseline fitted on training labels only.
    baseline = np.tile(train[targets].mean().to_numpy(), (len(val), 1))
    (out / 'baseline_metrics.json').write_text(json.dumps(metrics(val, baseline, targets, task), indent=2))
    if cfg.model.type == 'lightgbm':
        model = LightGBMModel(cfg.model, data, cfg.seed)
        best = model.fit(train, val, targets, task, out)
        (out / 'best_iterations.json').write_text(json.dumps(best, indent=2))
        predict_frame = model.predict
    else:
        predict_frame = train_torch(cfg, train, val, targets, task, data, out)

    validation_predictions = predict_frame(val)
    (out / 'validation_metrics.json').write_text(json.dumps(metrics(val, validation_predictions, targets, task), indent=2))
    vp = val[['SMILES', 'Molecule_Name']].copy()
    vp[targets] = validation_predictions
    vp.to_csv(out / 'validation_predictions.csv', index=False)
    # Only now load and evaluate the held-out labelled test split.
    test = load_data(cfg, 'test')
    if not len(test):
        raise ValueError('Empty test set')
    if set(frame.canonical_smiles) & set(test.canonical_smiles):
        raise ValueError('Exact molecule leakage into the test split')
    predictions = predict_frame(test)
    test_scores = metrics(test, predictions, targets, task)
    (out / 'test_metrics.json').write_text(json.dumps(test_scores, indent=2))
    # Classification exports probabilities; metrics use a fixed 0.5 threshold.
    output = test[['SMILES', 'Molecule_Name']].copy()
    output[targets] = predictions
    output.to_csv(out / 'test_predictions.csv', index=False)
    print(f'Held-out test metrics: {test_scores}')

    if cfg.device == 'cuda':
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"Peak tensor memory: {peak_gb:.2f} GB | Peak reserved by PyTorch: {reserved_gb:.2f} GB")


if __name__ == '__main__':
    main()
