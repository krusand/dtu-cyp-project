"""Models selectable with model.type. A model is an nn.Module taking
(model_cfg, readout_cfg, n_targets, data_dir, device) with fit_preprocessing(train),
loader(frame, targets, batch_size, shuffle), param_groups() and
forward(batch) -> [batch, n_targets] predictions. Readout heads are in models/readouts.py.
The LightGBM baseline (models/gbm.py, model.type=lightgbm) is handled separately in finetune.py."""
from models.atom_jepa import AtomJEPAModel
from models.baselines import FeatureModel
from models.chemberta import ChemBERTa

MODELS = {'atom_jepa': AtomJEPAModel, 'features': FeatureModel, 'chemberta': ChemBERTa}
