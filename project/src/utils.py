from __future__ import annotations

import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from typing import Dict, Any
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ------------------ config ------------------

def load_config(path: str) -> Dict[str, Any]:
    """Charger un fichier YAML de configuration et le retourner en dict Python."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ------------------ metrics & plots ------------------

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }
    return metrics


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confmat(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------ persistence helpers ------------------

def dump_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
