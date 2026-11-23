"""Baselines: Morgan fingerprints and/or RDKit descriptors fed to a readout head from
models/readouts.py. With the MLP readout these follow the feature-vector MLP baselines
in the Atom-JEPA repository (conf/model/*_mlp.yaml)."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import QuantileTransformer
from models.readouts import build_readout

_RDKIT_2D = None


def morgan(smiles, radius=2, n_bits=1024, use_counts=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetCountFingerprintAsNumPy(mol) if use_counts else gen.GetFingerprintAsNumPy(mol)
    return fp.astype(np.float32)


def rdkit(smiles):
    """The 200 CDF-normalized RDKit 2D descriptors from descriptastorus, in [0, 1]."""
    global _RDKIT_2D
    if _RDKIT_2D is None:
        from descriptastorus.descriptors import rdNormalizedDescriptors
        _RDKIT_2D = rdNormalizedDescriptors.RDKit2DNormalized()
    out = _RDKIT_2D.process(smiles)
    if out is None or not out[0]:
        raise ValueError(f'RDKit descriptors failed for {smiles}')
    x = np.asarray(out[1:], dtype=np.float32)
    return np.nan_to_num(x, nan=0., posinf=1., neginf=0.)


FEATURIZERS = {'morgan': morgan, 'rdkit': rdkit}


def load_features(frame, name, kwargs, folder, workers):
    """Features for each row of frame, cached per featurizer setting in folder."""
    tag = hashlib.sha256(json.dumps([name, kwargs], sort_keys=True).encode()).hexdigest()[:10]
    path = folder / f'{name}_{tag}.npz'
    cache = {}
    if path.exists():
        with np.load(path) as d:
            cache = dict(zip(d['keys'], d['X']))
    todo = frame.drop_duplicates('key')
    todo = todo[~todo.key.isin(cache.keys())]
    if len(todo):
        print(f'Computing {name} features for {len(todo)} molecules', flush=True)
        with ProcessPoolExecutor(workers) as pool:
            cache.update(zip(todo.key, pool.map(partial(FEATURIZERS[name], **kwargs),
                                                todo.SMILES, chunksize=32)))
        folder.mkdir(exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.npz')
        np.savez_compressed(tmp, keys=np.array(list(cache)), X=np.stack(list(cache.values())))
        tmp.replace(path)
    return np.stack([cache[k] for k in frame.key])


class Features(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {'features': self.X[i], 'y': self.y[i]}


class FeatureModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg, self.folder = cfg, data_dir / 'features'
        self.kwargs = {n: OmegaConf.to_container(cfg[n]) if n in cfg else {} for n in cfg.featurizers}
        in_dim = sum(len(FEATURIZERS[n]('C', **self.kwargs[n])) for n in cfg.featurizers)
        self.readout = build_readout(readout, in_dim, n_targets).to(device)
        self.scaler = None

    @property
    def device(self):
        return next(self.parameters()).device

    def features(self, frame):
        return np.concatenate([load_features(frame, n, self.kwargs[n], self.folder,
                                             self.cfg.feature_workers)
                               for n in self.cfg.featurizers], axis=1)

    def fit_preprocessing(self, train):
        if self.cfg.scaling == 'quantile':
            X = self.features(train)
            self.scaler = QuantileTransformer(n_quantiles=min(1000, len(X)), subsample=len(X),
                                              random_state=0).fit(X)
        elif self.cfg.scaling != 'none':
            raise ValueError(f'Unknown scaling {self.cfg.scaling!r}')

    def loader(self, frame, targets, batch_size, shuffle=False):
        X = self.features(frame)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)
        return DataLoader(Features(X, y), batch_size=batch_size, shuffle=shuffle)

    def param_groups(self):
        return [{'params': list(self.parameters()), 'lr': self.cfg.readout_lr}]

    def forward(self, batch):
        return self.readout(batch['features'].to(self.device))
