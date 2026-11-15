"""Pretrained Atom-JEPA encoder with a readout head from models/readouts.py."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from atom_jepa import AtomJEPA
from models.readouts import build_readout
from loguru import logger

class Conformers(Dataset):
    def __init__(self, frame, folder, targets, num_conformers):
        self.frame, self.folder = frame.reset_index(drop=True), folder
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)
        self.split = frame["split"].iloc[0]
        self.num_conformers=num_conformers

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        with np.load(self.folder / f'{self.frame.iloc[i].key}.npz') as d:
            z, pos = d['z'], d['pos']
            if z.ndim==1:
                z=z[:, None]
            num_conformers_in_file = z.shape[1]

            if self.split == 'train':
                c = 0 if self.num_conformers == 1 else np.random.randint(0, num_conformers_in_file)

                return {'atomic_numbers': torch.tensor(z[:,c], dtype=torch.long),
                        'node_coordinates': torch.tensor(pos[:, c*3:c*3+3], dtype=torch.float32),
                        'y': torch.tensor(self.y[i])}
            else: # test or val
                confs = []
                for c in range(self.num_conformers):
                    idx = min(c, num_conformers_in_file-1)
                    confs.append({'atomic_numbers': torch.tensor(z[:,idx], dtype=torch.long),
                        'node_coordinates': torch.tensor(pos[:, idx*3:idx*3+3], dtype=torch.float32),
                        'y': torch.tensor(self.y[i])})
                return confs


class AtomJEPAModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device, task):
        super().__init__()
        self.cfg, self.folder = cfg, data_dir / 'conformers'
        self.encoder = AtomJEPA.from_pretrained(cfg.checkpoint, device=device,
                                                precision=cfg.precision, cuequivariance=False)
        self.encoder.set_grad_checkpointing(not cfg.freeze_encoder)
        self.encoder.requires_grad_(not cfg.freeze_encoder)
        self.readout = build_readout(readout, self.encoder.embedding_dim, n_targets).to(self.encoder.device)

    @property
    def device(self):
        return self.encoder.device

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def _collate_fn(self, batch_list):
        return [
            self.encoder.collate([mol_confs[c] for mol_confs in batch_list])
            for c in range(self.cfg.num_conformers)
        ]

    def loader(self, frame, targets, batch_size, shuffle=False):
        dataset = Conformers(frame, self.folder, targets, self.cfg.num_conformers)
        collate_fn = self.encoder.collate if dataset.split == 'train' else self._collate_fn
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=shuffle, collate_fn=collate_fn, num_workers=0)

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({'params': list(self.encoder.parameters()), 'lr': self.cfg.encoder_lr})
        return groups

    def forward(self, batch):
        return self.readout(self.encoder(batch))  # Differentiable forward; do NOT use encoder.embed().
