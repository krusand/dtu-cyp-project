#!/usr/bin/env python3
"""Download the CYP regression and classification training data, generate conformers,
and split each task 70/15/15 (a molecule in both tasks gets the same split)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from atom_jepa.conformers import smiles_to_samples
from rdkit import Chem, rdBase
from sklearn.model_selection import GroupShuffleSplit
from loguru import logger

DATA_REVISION = '146282d2ce1532f1803679a834a9bec9dac1e482'
ATOM_JEPA_VERSION = importlib.metadata.version('atom-jepa')
REG = [f'{c}_pIC50_direct_inhibition' for c in ['CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4']]
CLS = ['CYP2D6_is_TDI', 'CYP3A4_is_TDI']
FILES = ['cyp-challenge-TRAIN_inhibition.csv', 'cyp-challenge-TRAIN_TDI.csv']
TASK_LABELS = {'regression': REG, 'classification': CLS}


def conformer(job):
    key, smiles, folder, seed = job
    path = Path(folder) / f'{key}.npz'
    if path.exists():
        return key, ''
    try:
        # Use the original SMILES and the upstream defaults, without retries or centering.
        samples = smiles_to_samples(smiles, num_conformers=1, seed=seed, mmff_max_iters=500)

        zs = np.column_stack([sample["atomic_numbers"].numpy() for sample in samples])
        poss = np.column_stack([sample["node_coordinates"].numpy() for sample in samples])
        
        np.savez_compressed(path, 
                            z=zs,
                            pos=poss)
        return key, ''
    except Exception as exc:
        return key, str(exc)


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def assign_splits(frame, seed, val_fraction=.15, test_fraction=.15):
    # Canonical duplicates stay together, including across the two task tables.
    n_groups = frame.canonical_smiles.nunique()
    n_test = max(1, round(n_groups * test_fraction))
    n_val = max(1, round(n_groups * val_fraction))
    if n_groups <= n_test + n_val:
        raise ValueError('Not enough distinct molecules for three nonempty splits')
    split = GroupShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
    development, test = next(split.split(frame, groups=frame.canonical_smiles))
    dev = frame.iloc[development]
    split = GroupShuffleSplit(n_splits=1, test_size=n_val,
                              random_state=seed + 1)
    _, val = next(split.split(dev, groups=dev.canonical_smiles))
    frame = frame.copy()
    frame['split'] = 'train'
    frame.loc[frame.index[test], 'split'] = 'test'
    frame.loc[frame.index[development[val]], 'split'] = 'val'
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('data'))
    p.add_argument('--raw-dir', type=Path, help='Folder containing the two labelled CSVs')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=.15)
    p.add_argument('--test-fraction', type=float, default=.15)
    p.add_argument('--limit', type=int, help='Small preparation smoke test only: first N rows')
    args = p.parse_args()
    if (args.workers < 1 or not 0 <= args.seed < 2**31 - 1 or
        min(args.val_fraction, args.test_fraction) <= 0 or
        args.val_fraction + args.test_fraction >= 1 or
        (args.limit is not None and args.limit < 3)):
        p.error('Use positive fractions summing to < 1, workers >= 1, limit >= 3 and a valid seed')
    args.out.mkdir(parents=True, exist_ok=True)
    config = {'data_revision': DATA_REVISION, 'seed': args.seed,
              'val_fraction': args.val_fraction, 'test_fraction': args.test_fraction,
              'limit': args.limit, 'rdkit': rdBase.rdkitVersion,
              'protocol': 'atom-jepa-' + ATOM_JEPA_VERSION}
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['config'] != config:
        raise ValueError('Settings or conformer protocol changed; use a NEW --out directory')
    raw = args.raw_dir or args.out / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = raw / name
        if not path.exists():
            if args.raw_dir:
                raise FileNotFoundError(path)
            url = f'https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/resolve/{DATA_REVISION}/{name}'
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
            path.write_bytes(content)
    hashes = {name: hashlib.sha256((raw / name).read_bytes()).hexdigest() for name in FILES}
    if manifest_path.exists() and json.loads(manifest_path.read_text())['sha256'] != hashes:
        raise ValueError('Raw files changed; use a new --out directory')
    manifest_path.write_text(json.dumps({'config': config, 'sha256': hashes}, indent=2))
    reg, cls = [pd.read_csv(raw / f) for f in FILES]
    for frame in (reg, cls):
        if frame.Molecule_Name.isna().any() or not frame.Molecule_Name.is_unique:
            raise ValueError('Missing or duplicate Molecule_Name')
    # Use regression labels from its table and released TDI labels from its own table.
    frame = reg.merge(cls[['Molecule_Name', 'SMILES'] + CLS], on=['Molecule_Name', 'SMILES'],
                      how='outer', validate='one_to_one')
    if not frame.Molecule_Name.is_unique:
        raise ValueError('Inconsistent SMILES for a molecule across the two tables')
    for col in CLS:
        mapping = {'true': 1., 'false': 0., '1': 1., '0': 0., '1.0': 1., '0.0': 0.}
        values = frame[col].astype(str).str.lower()
        if (frame[col].notna() & ~values.isin(mapping)).any():
            raise ValueError(f'Unexpected boolean values in {col}')
        frame[col] = values.map(mapping)
    frame = frame[frame[REG + CLS].notna().any(axis=1)].copy()
    if args.limit:
        frame = frame.head(args.limit).copy()
    frame['canonical_smiles'] = frame.SMILES.map(canonical)
    # Raw-SMILES cache keys preserve the exact input and atom ordering of the pipeline.
    frame['key'] = frame.SMILES.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    folder = args.out / 'conformers'
    folder.mkdir(exist_ok=True)
    unique = frame.drop_duplicates('key')
    work = [(r.key, r.SMILES, str(folder), args.seed) for r in unique.itertuples()]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(conformer, work), 1):
            results.append(result)
            if i % 100 == 0:
                print(f'Conformers {i}/{len(work)}', flush=True)
    report = pd.DataFrame(results, columns=['key', 'error'])
    report.to_csv(args.out / 'conformer_report.csv', index=False)
    failed = set(report.loc[report.error != '', 'key'])
    frame[frame.key.isin(failed)].to_csv(args.out / 'excluded_conformer_failures.csv', index=False)
    frame = frame[~frame.key.isin(failed)].reset_index(drop=True)
    # Splits are assigned over both tasks together, then each task gets its own dataset.
    frame = assign_splits(frame, args.seed, args.val_fraction, args.test_fraction)
    extra = ['canonical_smiles', 'key', 'split']
    columns = {'regression': list(reg.columns) + extra,
               'classification': ['Molecule_Name', 'SMILES'] + CLS + extra}
    counts = {}
    for task, labels in TASK_LABELS.items():
        data = frame[frame[labels].notna().any(axis=1)][columns[task]]
        (args.out / task).mkdir(exist_ok=True)
        for split in ['train', 'val', 'test']:
            data[data.split == split].to_csv(args.out / task / f'{split}.csv', index=False)
        counts[task] = data.split.value_counts().to_dict()
    manifest = {'config': config, 'sha256': hashes, 'counts': counts, 'failed_conformers': len(failed)}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
