from __future__ import annotations

"""
Evaluate the repository Decision Tree model.

Usage:
    python evaluate_decision_tree.py

Optional dependencies to install (if missing):
    pip install pandas numpy scikit-learn joblib
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate Decision Tree model on test data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "interim" / "tennis_matches_features.data",
        help="Feature dataset path.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "data" / "interim" / "tennis_matches_features_metadata.json",
        help="Metadata JSON containing numeric_features and categorical_features.",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=None,
        help="Optional separate test dataset path.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=root / "artifacts" / "decision_tree_model.joblib",
        help="Path to pre-trained Decision Tree pipeline.",
    )
    parser.add_argument(
        "--fallback-model-path",
        type=Path,
        default=root / "data" / "interim" / "decision_tree_model.joblib",
        help="Legacy fallback model path.",
    )
    parser.add_argument(
        "--metrics-source",
        type=Path,
        default=root / "data" / "interim" / "decision_tree_metrics.json",
        help="Optional training metrics JSON used to recover best_params for re-training.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="player_a_win",
        help="Target column name.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test fraction used when --test-input is not provided.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state used for reproducible split and CV.",
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help="Run optional 5-fold cross-validation summary (configurable with --cv-folds).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for optional cross-validation.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=root / "artifacts",
        help="Output artifacts directory.",
    )
    return parser.parse_args()


def ensure_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name}: {path}. Build data first with scripts/data_cleaner/build_features.py."
        )


def load_dataset(path: Path, target_column: str) -> pd.DataFrame:
    ensure_file(path, "dataset")
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Missing target column '{target_column}' in {path}.")
    return df


def load_metadata(path: Path) -> dict:
    ensure_file(path, "metadata")
    return json.loads(path.read_text(encoding="utf-8"))


def select_features(
    df: pd.DataFrame,
    metadata: dict,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    numeric_features = [c for c in metadata.get("numeric_features", []) if c in df.columns]
    categorical_features = [c for c in metadata.get("categorical_features", []) if c in df.columns]
    if not numeric_features and not categorical_features:
        raise ValueError("No model features found from metadata in the provided dataset.")

    X = df[numeric_features + categorical_features].copy()
    y = df[target_column].astype(int)
    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features: list[str], categorical_features: list[str], random_state: int) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", DecisionTreeClassifier(criterion="gini", random_state=random_state)),
        ]
    )


def maybe_apply_best_params(pipeline: Pipeline, metrics_source: Path) -> Pipeline:
    if not metrics_source.exists():
        return pipeline

    try:
        payload = json.loads(metrics_source.read_text(encoding="utf-8"))
        best_params = payload.get("best_params")
        if isinstance(best_params, dict) and best_params:
            pipeline.set_params(**best_params)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        warnings.warn(f"Could not read best params from {metrics_source}: {exc}")
    return pipeline


def choose_model_path(primary: Path, fallback: Path) -> Path | None:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return None


def build_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
    labels = sorted(pd.unique(y_true).tolist())
    is_binary = len(labels) == 2

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "is_binary": is_binary,
        "labels": labels,
        "roc_auc": None,
        "roc_auc_warning": None,
    }

    if is_binary:
        positive_label = 1 if 1 in labels else labels[-1]
        metrics["precision_binary"] = float(
            precision_score(y_true, y_pred, average="binary", pos_label=positive_label, zero_division=0)
        )
        metrics["recall_binary"] = float(
            recall_score(y_true, y_pred, average="binary", pos_label=positive_label, zero_division=0)
        )
        metrics["f1_binary"] = float(
            f1_score(y_true, y_pred, average="binary", pos_label=positive_label, zero_division=0)
        )

        if y_proba is not None:
            try:
                positive_index = labels.index(positive_label)
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, positive_index]))
            except Exception as exc:  # noqa: BLE001
                warning = f"roc_auc not computed: {exc}"
                metrics["roc_auc_warning"] = warning
                warnings.warn(warning)
        else:
            warning = "roc_auc skipped: predict_proba not available on model."
            metrics["roc_auc_warning"] = warning
            warnings.warn(warning)
    else:
        warning = "roc_auc skipped: multiclass target detected."
        metrics["roc_auc_warning"] = warning
        warnings.warn(warning)

    return metrics


def run_optional_cv(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, folds: int, random_state: int) -> dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted"},
        n_jobs=-1,
    )

    return {
        "folds": folds,
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "f1_weighted_mean": float(np.mean(scores["test_f1_weighted"])),
        "f1_weighted_std": float(np.std(scores["test_f1_weighted"])),
    }


def main() -> None:
    args = parse_args()

    metadata = load_metadata(args.metadata)
    train_df = load_dataset(args.input, args.target_column)
    X_all, y_all, numeric_features, categorical_features = select_features(
        train_df,
        metadata,
        args.target_column,
    )

    if args.test_input is not None:
        test_df = load_dataset(args.test_input, args.target_column)
        X_train = X_all
        y_train = y_all
        X_test, y_test, _, _ = select_features(test_df, metadata, args.target_column)
        split_info = f"External test set: {args.test_input}"
    else:
        stratify = y_all if y_all.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify,
        )
        split_info = (
            f"Recreated train/test split with test_size={args.test_size} and random_state={args.random_state}"
        )

    model_path = choose_model_path(args.model_path, args.fallback_model_path)
    if model_path is not None:
        model = joblib.load(model_path)
        model_source = f"Loaded existing model from {model_path}"
    else:
        model = build_pipeline(numeric_features, categorical_features, args.random_state)
        model = maybe_apply_best_params(model, args.metrics_source)
        model.fit(X_train, y_train)
        model_source = "No pre-trained model found; trained a new Decision Tree pipeline"

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = build_metrics(y_test, y_pred, y_proba)

    labels = metrics["labels"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    cls_report_text = classification_report(y_test, y_pred, zero_division=0)

    cv_summary = None
    if args.run_cv:
        cv_model = build_pipeline(numeric_features, categorical_features, args.random_state)
        cv_model = maybe_apply_best_params(cv_model, args.metrics_source)
        cv_summary = run_optional_cv(cv_model, X_train, y_train, args.cv_folds, args.random_state)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.artifacts_dir / "metrics_decision_tree.json"
    cm_path = args.artifacts_dir / "confusion_matrix_decision_tree.csv"
    report_path = args.artifacts_dir / "classification_report_decision_tree.txt"

    payload = {
        "model_source": model_source,
        "split_info": split_info,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "numeric_features_count": int(len(numeric_features)),
        "categorical_features_count": int(len(categorical_features)),
        "metrics": metrics,
        "cross_validation": cv_summary,
    }

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cm_df.to_csv(cm_path, index=True)
    report_path.write_text(cls_report_text, encoding="utf-8")

    print("=" * 72)
    print("DECISION TREE EVALUATION SUMMARY")
    print("=" * 72)
    print(model_source)
    print(split_info)
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    elif metrics.get("roc_auc_warning"):
        print(f"ROC AUC: skipped ({metrics['roc_auc_warning']})")

    if cv_summary is not None:
        print("\nCross-validation (StratifiedKFold)")
        print(
            f"Accuracy mean±std: {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}"
        )
        print(
            "F1 weighted mean±std: "
            f"{cv_summary['f1_weighted_mean']:.4f} ± {cv_summary['f1_weighted_std']:.4f}"
        )

    print("\nSaved artifacts:")
    print(f"- {metrics_path}")
    print(f"- {cm_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
