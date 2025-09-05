from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    model_type: str = "logreg"
) -> Pipeline:
    """
    Build a preprocessing + model pipeline.

    Parameters
    ----------
    numeric : list[str]
        Numeric feature names.
    categorical : list[str]
        Categorical feature names.
    model_type : str
        "logreg" or "random_forest".
    """

    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # NOTE: use sparse_output=False (scikit-learn >=1.2).
    # If using older scikit-learn, replace with sparse=False.
    cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop",
        n_jobs=None,
    )

    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe
