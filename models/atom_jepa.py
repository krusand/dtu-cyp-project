"""Pretrained Atom-JEPA encoder with a readout head from models/readouts.py."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from atom_jepa import AtomJEPA
from models.readouts import build_readout


class Conformers(Dataset):
    def __init__(self, frame, folder, targets):
        self.frame, self.folder = frame.reset_index(drop=True), folder
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        with np.load(self.folder / f'{self.frame.iloc[i].key}.npz') as d:
            num_conformers_in_file = d['z'].shape[1]
            sample_idx = np.random.randint(0, num_conformers_in_file)
            pos_sample_from_idx = sample_idx*3
            pos_sample_to_idx = sample_idx*3+3

            return {'atomic_numbers': torch.tensor(d['z'][:,sample_idx], dtype=torch.long),
                    'node_coordinates': torch.tensor(d['pos'][:, pos_sample_from_idx:pos_sample_to_idx], dtype=torch.float32),
                    'y': torch.tensor(self.y[i])}


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

    def loader(self, frame, targets, batch_size, shuffle=False):
        return DataLoader(Conformers(frame, self.folder, targets), batch_size=batch_size,
                          shuffle=shuffle, collate_fn=self.encoder.collate, num_workers=0)

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({'params': list(self.encoder.parameters()), 'lr': self.cfg.encoder_lr})
        return groups

    def forward(self, batch):
        return self.readout(self.encoder(batch))  # Differentiable forward; do NOT use encoder.embed().
