import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models.readouts import build_readout
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from scipy.special import softmax

class ChemBERTaDataset(Dataset):
    def __init__(self, df, tokenizer, targets, include_labels=True):
        self.encodings = tokenizer(df["SMILES"].tolist(), truncation=True, padding=True)
        self.y = df.reindex(columns=targets).to_numpy(dtype=np.float32)
        print(self.y)
        self.include_labels = include_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.include_labels and self.y is not None:
            item["y"] = torch.tensor(self.y[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class ChemBERTa(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device, task):
        super().__init__()
        self.cfg, self.folder = cfg, data_dir / 'features'

        config = RobertaConfig.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        self.tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        if task == "classification":
            config.num_labels = 2
            config.problem_type = "single_label_classification"
        if task == "regression":
            config.num_labels = 1
            config.problem_type = "regression"
        
        config.output_hidden_states = True
        self.encoder = RobertaForSequenceClassification.from_pretrained(
            "DeepChem/ChemBERTa-77M-MLM", 
            device_map=device,
            config=config, 
        )

        if cfg.freeze_encoder:
            for name, param in self.encoder.base_model.named_parameters():
                param.requires_grad = False

        self.readout = build_readout(readout, config.hidden_size, n_targets).to(self.encoder.device)

    @property
    def device(self):
        return next(self.parameters()).device

    def fit_preprocessing(self, train):
        pass

    def loader(self, frame, targets, batch_size, shuffle=False):
        frame = frame[["SMILES"] + targets]
        data_loader = DataLoader(ChemBERTaDataset(frame, self.tokenizer, targets), batch_size=batch_size, shuffle=shuffle)
        return data_loader

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({'params': list(self.encoder.parameters()), 'lr': self.cfg.encoder_lr})
        return groups

    def forward(self, batch):
        with torch.no_grad():
            output = self.encoder(**batch)
        features = output.hidden_states[1].mean(dim=1)
        return self.readout(features)