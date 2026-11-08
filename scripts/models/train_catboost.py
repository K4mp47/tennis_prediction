"""Train a simple CatBoost model for tennis match prediction."""

from __future__ import annotations

import argparse
import json
import hashlib
import platform
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

from training_utils import (
    chronological_holdout_split,
    feature_schema,
    load_feature_dataset,
)


# Shared preprocessing below is also imported by prediction and notebook code.
# Both training and inference use this token for missing categorical values.
MISSING_CATEGORY = "__MISSING__"


# Command-line interface: resolve default dataset and artifact paths from the project root.
def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train the CatBoost tennis model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/interim/tennis_matches_features.data",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "data/interim/tennis_matches_features_metadata.json",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=root / "data/interim/catboost_model.cbm",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=root / "data/interim/catboost_metrics.json",
    )
    return parser.parse_args()


# Feature preparation: keep the same input representation during training and inference.
def prepare_catboost_features(
    features: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Prepare numeric and categorical columns for CatBoost."""

    features = features.copy()

    # Invalid numeric values become NaN, which CatBoost handles as missing data.
    for column in numeric_features:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    # CatBoost categorical inputs must be strings with no null values.
    for column in categorical_features:
        features[column] = (
            features[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
        )

    return features


# Holdout selection: split distinct dates so matches on one date stay together.
def chronological_split(
    data: pd.DataFrame,
    test_fraction: float = 0.15,
) -> tuple[pd.Series, pd.Series]:
    """Use old matches for training and the newest dates for testing."""

    train_indices, test_indices = chronological_holdout_split(
        data["date"], test_fraction
    )
    train_mask = pd.Series(False, index=data.index)
    test_mask = pd.Series(False, index=data.index)
    train_mask.iloc[train_indices] = True
    test_mask.iloc[test_indices] = True
    return train_mask, test_mask


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification_metrics(target, probability) -> dict[str, float]:
    """Evaluate a binary probability vector, including uncovered market subsets."""
    target = np.asarray(target)
    probability = np.asarray(probability, dtype=float)
    if not len(target):
        return dict.fromkeys(
            ("accuracy", "roc_auc", "log_loss", "brier_score"), float("nan")
        )
    return {
        "accuracy": float(accuracy_score(target, probability >= 0.5)),
        "roc_auc": float(roc_auc_score(target, probability))
        if len(np.unique(target)) == 2
        else float("nan"),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability)),
    }


def evaluate_market(target, model_probability, market_probability) -> dict:
    """Compare both predictors on exactly the rows with valid market probabilities."""
    market = np.asarray(market_probability, dtype=float)
    valid = np.isfinite(market) & (market > 0) & (market < 1)
    return {
        "rows": int(valid.sum()),
        "coverage": float(valid.mean()) if len(valid) else 0.0,
        "catboost": classification_metrics(
            np.asarray(target)[valid], np.asarray(model_probability)[valid]
        ),
        "market": classification_metrics(np.asarray(target)[valid], market[valid]),
    }


def load_verified_model(
    model_path: Path,
    metrics: dict,
    *,
    input_path: Path | None = None,
    metadata_path: Path | None = None,
) -> CatBoostClassifier:
    """Reject mixed artifacts; evaluation additionally verifies its original inputs."""
    numeric, categorical = feature_schema(metrics)
    hashes = metrics.get("sha256", {})
    for name, path in (
        ("model", model_path),
        ("input", input_path),
        ("metadata", metadata_path),
    ):
        if path is not None and hashes.get(name) != sha256_file(path):
            raise ValueError(
                f"Artifact hash mismatch: {name}. Regenerate matching training artifacts."
            )
    if metadata_path is not None:
        saved_schema = feature_schema(
            json.loads(metadata_path.read_text(encoding="utf-8"))
        )
        if saved_schema != (numeric, categorical):
            raise ValueError("Training metadata and model feature schema differ.")
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    if model.feature_names_ != list(numeric + categorical):
        raise ValueError("Model feature order does not match the metrics schema.")
    if model.get_cat_feature_indices() != list(
        range(len(numeric), len(numeric + categorical))
    ):
        raise ValueError("Model categorical features do not match the metrics schema.")
    classes = model.classes_
    if classes is None or list(classes) != [0, 1]:
        raise ValueError("Model must predict classes [0, 1] in that order.")
    return model


# Training workflow: load -> prepare -> split -> fit -> evaluate -> save.
def main() -> None:
    args = parse_args()

    dataset = load_feature_dataset(args.input, args.metadata)
    numeric_features = list(dataset.numeric_features)
    categorical_features = list(dataset.categorical_features)
    forbidden = {
        "date",
        "player_a_win",
        "match_id_internal",
        "winner_name",
        "loser_name",
        "winner_sackmann_id",
        "loser_sackmann_id",
    }
    if forbidden.intersection(dataset.features.columns):
        raise ValueError("Identifiers, dates and outcomes must not be model features.")
    X = prepare_catboost_features(
        dataset.features, numeric_features, categorical_features
    )
    y = dataset.target
    train_indices, test_indices = chronological_holdout_split(dataset.dates, 0.15)
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Training and holdout must both contain classes 0 and 1.")

    # 3. Create the model.
    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=3,
        loss_function="Logloss",
        cat_features=categorical_features,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    # 4. Train.
    model.fit(X_train, y_train)

    train_metrics = classification_metrics(y_train, model.predict_proba(X_train)[:, 1])
    holdout_metrics = classification_metrics(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Accuracy: {holdout_metrics['accuracy']:.4f}")
    print(f"ROC AUC: {holdout_metrics['roc_auc']:.4f}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = args.output_model.with_suffix(".cbm.tmp")
    model.save_model(str(temporary_model))
    temporary_model.replace(args.output_model)

    train_dates = dataset.dates.iloc[train_indices]
    holdout_dates = dataset.dates.iloc[test_indices]
    metrics = {
        "algorithm": "CatBoostClassifier",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ("catboost", "pandas", "scikit-learn", "numpy")
            },
        },
        "sha256": {
            "input": sha256_file(args.input),
            "metadata": sha256_file(args.metadata),
            "model": sha256_file(args.output_model),
            "trainer": sha256_file(Path(__file__)),
        },
        "params": model.get_params(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_rows": len(X_train),
        "holdout_rows": len(X_test),
        "train_date_range": [
            train_dates.min().date().isoformat(),
            train_dates.max().date().isoformat(),
        ],
        "holdout_date_range": [
            holdout_dates.min().date().isoformat(),
            holdout_dates.max().date().isoformat(),
        ],
        "split_date": holdout_dates.min().date().isoformat(),
        "holdout_fraction": 0.15,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_accuracy": holdout_metrics["accuracy"],
        "holdout_roc_auc": holdout_metrics["roc_auc"],
        "evaluation_note": "Reused historical holdout; not a new independent test.",
    }

    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# Run the CLI only when executed directly; imports can reuse the helpers.
if __name__ == "__main__":
    main()
