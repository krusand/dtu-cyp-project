
# Test the runs
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix, matthews_corrcoef

RANKED_RUNS_FILE_PATH = "top5_atom_jepa_classification_tier2.json"
TASK = "classification"
METRICS = {
    'regression': [ "CYP1A2_pIC50_direct_inhibition", "CYP2C9_pIC50_direct_inhibition", "CYP2D6_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition"],
    'classification': ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]
}
DATA_PATH = Path(f"data/{TASK}")
PLOT_FILE_NAME = "atom_jepa_confusion_matrix"

def test_prediction_path(path: Path):
    assert os.path.exists(path), f"Folder {path} does not exists"
    assert os.path.exists(path / "validation_predictions.csv"), f"No validation predictions in folder {path}"
    assert os.path.exists(path / "test_predictions.csv"), f"No test predictions in folder {path}"

def calculate_errors(actual, preds):

    errors = (
        actual
        .merge(preds, how='left', left_on=['Molecule_Name','SMILES'], right_on=['Molecule_Name','SMILES'])
    )
    for col in METRICS[TASK]:
        errors[col + "_err"] = errors[col] - errors[col + "_hat"]

    return errors

def plot_residuals(errors, split):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs = axs.reshape(-1)

    for i, metric in enumerate(METRICS[TASK]):
        sns.scatterplot(x=errors[metric], y=errors[metric + '_err'], ax=axs[i])
        axs[i].set_ylabel("Residuals")
    
    if not os.path.exists("plots"):
        os.makedirs("plots")

    fig.set_size_inches(12.,6.)
    plt.tight_layout()
    plt.savefig(f'plots/{PLOT_FILE_NAME}_{split}.png', dpi=144)


def plot_confusion_matrix(actual, preds, split):
    merged = actual.merge(
        preds, how='left', left_on=['Molecule_Name', 'SMILES'], right_on=['Molecule_Name', 'SMILES']
    )
    targets = METRICS[TASK]

    fig, axs = plt.subplots(nrows=1, ncols=len(targets), figsize=(4.5 * len(targets), 4))
    if len(targets) == 1:
        axs = [axs]

    for ax, metric in zip(axs, targets):
        valid = merged[[metric, metric + '_hat']].dropna()
        y_true = valid[metric].astype(int)
        y_pred = (valid[metric + '_hat'] >= 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred, normalize='true')
        mcc = matthews_corrcoef(y_true, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            cbar=False,
            vmin=0,
            vmax=1,
            xticklabels=['Non-inh.', 'Inhib.'],
            yticklabels=['Non-inh.', 'Inhib.'],
            ax=ax,
        )
        ax.set_title(f"{metric}\n(n={len(valid)}, MCC={mcc:.2f}) : {split.upper()} set")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'plots/atom_jepa_confusion_matrix_{split}.png', dpi=144)
    plt.close()

def main():
    with open(RANKED_RUNS_FILE_PATH, 'r') as fp:
        ranked_runs = json.load(fp)

    best_run = ranked_runs[TASK][0]
    best_run_run_dir = Path(best_run["run_dir"])

    test_prediction_path(best_run_run_dir)

    val_actual = pd.read_csv(DATA_PATH / "val.csv")
    test_actual = pd.read_csv(DATA_PATH / "test.csv")

    val_preds = pd.read_csv(best_run_run_dir / "validation_predictions.csv")
    test_preds = pd.read_csv(best_run_run_dir / "test_predictions.csv")

    val_actual = val_actual[["Molecule_Name", "SMILES"] + METRICS[TASK]]
    test_actual = test_actual[["Molecule_Name", "SMILES"] + METRICS[TASK]]

    col_mapper = {key: key + '_hat' for key in METRICS[TASK]}

    val_preds = val_preds.rename(columns=col_mapper)
    test_preds = test_preds.rename(columns=col_mapper)



    if TASK == 'regression':
        val_errors = calculate_errors(val_actual, val_preds)
        test_errors = calculate_errors(test_actual, test_preds)
        
        plot_residuals(val_errors, split='val')
        plot_residuals(test_errors, split='test')

    elif TASK == 'classification':
        plot_confusion_matrix(val_actual, val_preds, split='val')
        plot_confusion_matrix(test_actual, test_preds, split='test')


if __name__ == '__main__':
    main()
