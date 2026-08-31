"""Train and evaluate CatBoost on the engineered tennis match dataset.

CatBoost consumes categorical columns natively.  Model selection uses
expanding chronological folds based on unique match dates, followed by an
untouched chronological holdout evaluation.

Typical usage from the project root::

    uv run python scripts/models/train_catboost.py \
        --input data/interim/tennis_matches_features.data \
        --metadata data/interim/tennis_matches_features_metadata.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import catboost
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from training_utils import (
    chronological_holdout_split,
    load_feature_dataset,
    make_date_based_folds,
    parse_excluded_features,
)


MISSING_CATEGORY = "__MISSING__"
T = TypeVar("T", int, float)


def comma_separated_values(value: str, value_type: type[T]) -> list[T]:
    """Convert a comma-separated CLI value into a non-empty typed list."""

    try:
        parsed = [value_type(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Valore non valido nella lista '{value}'."
        ) from error
    if not parsed:
        raise argparse.ArgumentTypeError("La lista dei valori non puo' essere vuota.")
    return parsed


def integer_grid(value: str) -> list[int]:
    return comma_separated_values(value, int)


def float_grid(value: str) -> list[float]:
    return comma_separated_values(value, float)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Train CatBoost on the engineered tennis match dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_root / "data" / "interim" / "tennis_matches_features.data",
        help="Output di build_features.py",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=(
            default_root / "data" / "interim" / "tennis_matches_features_metadata.json"
        ),
        help="Metadata JSON prodotto da build_features.py",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.15,
        help="Frazione delle date piu' recenti riservata al test finale",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Numero di fold temporali usati dalla grid search",
    )
    parser.add_argument(
        "--exclude-features",
        type=str,
        default="",
        help="Feature da escludere, separate da virgola",
    )
    parser.add_argument(
        "--iterations",
        type=integer_grid,
        default="300,600",
        help="Valori CatBoost iterations da provare, separati da virgola",
    )
    parser.add_argument(
        "--depth",
        type=integer_grid,
        default="4,6,8",
        help="Valori CatBoost depth da provare, separati da virgola",
    )
    parser.add_argument(
        "--learning-rate",
        type=float_grid,
        default="0.03,0.1",
        help="Learning rate da provare, separati da virgola",
    )
    parser.add_argument(
        "--l2-leaf-reg",
        type=float_grid,
        default="3,7",
        help="Regolarizzazione L2 da provare, separata da virgola",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Processi paralleli usati da GridSearchCV (-1 usa tutte le CPU)",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=default_root / "data" / "interim" / "catboost_metrics.json",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=default_root / "data" / "interim" / "catboost_model.cbm",
        help="Percorso del modello CatBoost addestrato",
    )
    return parser.parse_args()


def prepare_catboost_features(
    features: pd.DataFrame,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
) -> pd.DataFrame:
    """Normalize dtypes while preserving CatBoost's native missing numerics."""

    prepared = features.copy()
    for feature in numeric_features:
        prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce")
    for feature in categorical_features:
        prepared[feature] = (
            prepared[feature].astype("string").fillna(MISSING_CATEGORY).astype(str)
        )
    return prepared


def candidate_summary(search: GridSearchCV, limit: int = 10) -> list[dict[str, object]]:
    """Return the best CV candidates in a compact JSON-friendly form."""

    results = pd.DataFrame(search.cv_results_).sort_values(
        ["rank_test_score", "mean_fit_time"]
    )
    summary: list[dict[str, object]] = []
    for _, row in results.head(limit).iterrows():
        summary.append(
            {
                "rank": int(row["rank_test_score"]),
                "mean_cv_accuracy": float(row["mean_test_score"]),
                "std_cv_accuracy": float(row["std_test_score"]),
                "mean_fit_time_seconds": float(row["mean_fit_time"]),
                "params": row["params"],
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    excluded_features = parse_excluded_features(args.exclude_features)
    dataset = load_feature_dataset(args.input, args.metadata, excluded_features)
    features = prepare_catboost_features(
        dataset.features,
        dataset.numeric_features,
        dataset.categorical_features,
    )

    print("=" * 70)
    print("TRAIN CATBOOST")
    print("=" * 70)
    print(f"Righe: {len(features):,}")
    print(f"Feature numeriche: {len(dataset.numeric_features)}")
    print(f"Feature categoriche: {len(dataset.categorical_features)}")
    print(f"Feature escluse: {sorted(excluded_features)}")

    train_indices, holdout_indices = chronological_holdout_split(
        dataset.dates, args.holdout_fraction
    )
    X_train = features.iloc[train_indices]
    y_train = dataset.target.iloc[train_indices]
    dates_train = dataset.dates.iloc[train_indices]
    X_holdout = features.iloc[holdout_indices]
    y_holdout = dataset.target.iloc[holdout_indices]
    dates_holdout = dataset.dates.iloc[holdout_indices]

    cv_folds = make_date_based_folds(dates_train.reset_index(drop=True), args.cv_folds)
    print(f"Training set: {len(X_train):,} righe")
    print(
        f"Holdout finale: {len(X_holdout):,} righe "
        f"(ultimo {args.holdout_fraction:.0%} delle date)"
    )
    print(f"Fold di cross-validation: {len(cv_folds)}")

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Accuracy",
        # A tuple is intentionally used here. CatBoost copies lists internally,
        # which violates scikit-learn 1.9's strict estimator-cloning contract.
        cat_features=dataset.categorical_features,
        random_seed=args.random_seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=1,
    )
    parameter_grid = {
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "l2_leaf_reg": args.l2_leaf_reg,
    }

    candidate_count = (
        len(args.iterations)
        * len(args.depth)
        * len(args.learning_rate)
        * len(args.l2_leaf_reg)
    )
    print(f"Combinazioni di iperparametri: {candidate_count}")
    print("Grid search in corso...")
    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=cv_folds,
        n_jobs=args.jobs,
        refit=True,
        verbose=2,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    best_model: CatBoostClassifier = search.best_estimator_
    holdout_predictions = best_model.predict(X_holdout).astype(int).ravel()
    holdout_probabilities = best_model.predict_proba(X_holdout)[:, 1]
    holdout_accuracy = accuracy_score(y_holdout, holdout_predictions)

    print(f"Migliori parametri: {search.best_params_}")
    print(f"Miglior accuratezza CV: {search.best_score_:.4f}")
    print(f"Accuratezza su holdout finale: {holdout_accuracy:.4f}")
    print()
    print(classification_report(y_holdout, holdout_predictions))

    importances = sorted(
        zip(features.columns, best_model.feature_importances_, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top_importances = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in importances[:20]
    ]
    print("Top 10 feature per importanza:")
    for item in top_importances[:10]:
        print(f"  {item['feature']:<40} {item['importance']:.4f}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    best_model.save_model(str(args.output_model))

    report = classification_report(
        y_holdout,
        holdout_predictions,
        output_dict=True,
    )
    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "CatBoostClassifier",
        "catboost_version": catboost.__version__,
        "input_path": str(args.input),
        "model_path": str(args.output_model),
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_holdout)),
        "train_date_range": [
            dates_train.min().date().isoformat(),
            dates_train.max().date().isoformat(),
        ],
        "holdout_date_range": [
            dates_holdout.min().date().isoformat(),
            dates_holdout.max().date().isoformat(),
        ],
        "holdout_fraction": args.holdout_fraction,
        "cv_folds": len(cv_folds),
        "excluded_features": sorted(excluded_features),
        "numeric_features": list(dataset.numeric_features),
        "categorical_features": list(dataset.categorical_features),
        "missing_category_value": MISSING_CATEGORY,
        "parameter_grid": parameter_grid,
        "best_params": search.best_params_,
        "best_cv_accuracy": float(search.best_score_),
        "holdout_accuracy": float(holdout_accuracy),
        "holdout_roc_auc": float(roc_auc_score(y_holdout, holdout_probabilities)),
        "holdout_log_loss": float(log_loss(y_holdout, holdout_probabilities)),
        "holdout_brier_score": float(
            brier_score_loss(y_holdout, holdout_probabilities)
        ),
        "confusion_matrix_holdout": confusion_matrix(
            y_holdout, holdout_predictions
        ).tolist(),
        "classification_report_holdout": report,
        "top_feature_importances": top_importances,
        "top_cv_candidates": candidate_summary(search),
    }

    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Modello salvato in: {args.output_model}")
    print(f"Metriche salvate in: {args.output_metrics}")
    print("Pipeline completata.")


if __name__ == "__main__":
    main()
