from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV

from utils import (
    load_config,
    load_csv,
    split_features_target,
    make_split,
    make_cv,
    ensure_dir,
    save_model,
    dump_json,
    compute_metrics,
)
from pipeline import build_pipeline


def _prefix_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {f"model__{k}": v for k, v in params.items()}


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    # MLflow setup from env (.env or export before)
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")
    mlflow.set_experiment(exp_name)

    # Load data
    df = load_csv(cfg.data["csv_path"])
    X, y = split_features_target(df, cfg.data["target"], cfg.data.get("positive_class"))

    X_train, X_val, y_train, y_val = make_split(
        X, y, test_size=cfg.data["test_size"], random_state=cfg.data["random_state"]
    )

    numeric = cfg.features["numeric"]
    categorical = cfg.features["categorical"]
    pipe = build_pipeline(numeric, categorical, cfg.model["type"])  # base pipeline

    param_grid = _prefix_params(cfg.model.get("params", {}))
    cv = make_cv(cfg.cv)

    with mlflow.start_run(run_name="train"):
        mlflow.log_params({"model_type": cfg.model["type"]})

        # Enable autologging for sklearn
        mlflow.sklearn.autolog(log_models=False)  # we'll log best model explicitly

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv,
            scoring=cfg.cv.get("scoring", "roc_auc"),
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        mlflow.log_params({"best_params": grid.best_params_})
        mlflow.log_metrics({"best_cv_score": grid.best_score_})

        # Evaluate on hold-out validation set quickly
        y_proba = best_model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        val_metrics = compute_metrics(y_val, y_proba, y_pred)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # Save artifacts to local folder and log to MLflow
        ensure_dir("artifacts")
        model_path = os.path.join("artifacts", "best_model.pkl")
        save_model(best_model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # Optional: log feature names after fit
        try:
            pre = best_model.named_steps["pre"]
            feature_names = []
            if hasattr(pre, "get_feature_names_out"):
                feature_names = pre.get_feature_names_out()
            dump_json(
                {"feature_names": list(map(str, feature_names))},
                os.path.join("artifacts", "feature_names.json"),
            )
            mlflow.log_artifact(
                os.path.join("artifacts", "feature_names.json"),
                artifact_path="model",
            )
        except Exception as e:
            print("Could not log feature names:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--register_model", type=str, default="")
    main(parser.parse_args())
