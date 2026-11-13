"""Readout heads: map molecule embeddings [batch, in_dim] to predictions [batch, n_targets].

To add a head, write a class taking (in_dim, n_targets, **options), register it in
READOUTS and add conf/readout/<name>.yaml. Use it with e.g. readout=<name>."""
from torch import nn


class LinearReadout(nn.Module):
    def __init__(self, in_dim, n_targets):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_targets)

    def forward(self, x):
        return self.linear(x)


class MLPReadout(nn.Module):
    """num_layers Linear layers (including the output layer) with ReLU and dropout."""

    def __init__(self, in_dim, n_targets, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers, nn.Linear(dims[-1], n_targets))

    def forward(self, x):
        return self.mlp(x)

class MLPBNReadout(nn.Module):
    """num_layers Linear layers (including the output layer) with ReLU and dropout."""

    def __init__(self, in_dim, n_targets, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers, nn.Linear(dims[-1], n_targets))

    def forward(self, x):
        return self.mlp(x)



READOUTS = {'linear': LinearReadout, 'mlp': MLPReadout, 'mlp_bn': MLPBNReadout}


def build_readout(cfg, in_dim, n_targets):
    options = {k: v for k, v in cfg.items() if k != 'type'}
    return READOUTS[cfg.type](in_dim, n_targets, **options)
