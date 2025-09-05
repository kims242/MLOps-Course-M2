import os
import pandas as pd
import yaml
import joblib
import mlflow
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
 
# === Charger la configuration ===
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
CONFIG_PATH = os.path.abspath(CONFIG_PATH)
 
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
 
# === Construire les chemins ===
project_root = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(project_root, config["data_path"])
model_dir = os.path.join(project_root, config["model_dir"])
model_path = os.path.join(model_dir, config["model_name"])
 
# === Charger les données ===
df = pd.read_csv(data_path, header=0)
X = df.drop("target", axis=1)
y = df["target"]
 
# === Charger le modèle entraîné ===
clf = joblib.load(model_path)
 
# === Faire des prédictions ===
y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)[:, 1]  # pour ROC
 
# === Évaluer ===
acc = accuracy_score(y, y_pred)
print(f"📊 Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
 
# === MLflow setup ===
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])
 
with mlflow.start_run():
    # Logger métriques
    mlflow.log_metric("accuracy_eval", acc)
 
    # === Courbe ROC ===
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
 
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join(model_dir, "roc_curve.png")
    plt.savefig(roc_path)
    mlflow.log_artifact(roc_path)
    plt.close()
 
    # === Confusion Matrix ===
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()
 
    print(f"✅ Résultats loggés dans MLflow avec AUC={roc_auc:.3f}")
 
 