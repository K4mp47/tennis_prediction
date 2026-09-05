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
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import catboost
import numpy as np
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
from sklearn.model_selection import RandomizedSearchCV

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
        default="500,1000,1500",
        help="Valori CatBoost iterations da provare, separati da virgola",
    )
    parser.add_argument(
        "--depth",
        type=integer_grid,
        default="5,6,7,8",
        help="Valori CatBoost depth da provare, separati da virgola",
    )
    parser.add_argument(
        "--learning-rate",
        type=float_grid,
        default="0.01,0.02,0.03,0.05",
        help="Learning rate da provare, separati da virgola",
    )
    parser.add_argument(
        "--l2-leaf-reg",
        type=float_grid,
        default="3,5,10,15",
        help="Regolarizzazione L2 da provare, separata da virgola",
    )
    parser.add_argument(
        "--random-strength",
        type=float_grid,
        default="0.5,1,2",
        help="Random strength da provare, separati da virgola",
    )
    parser.add_argument(
        "--bagging-temperature",
        type=float_grid,
        default="0,0.5,1",
        help="Bagging temperature da provare, separate da virgola",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=12,
        help="Numero di combinazioni casuali da valutare",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=100,
        help="Round senza miglioramenti prima dell'arresto anticipato",
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
    parser.add_argument(
        "--output-holdout-predictions",
        type=Path,
        default=default_root / "data/interim/catboost_holdout_predictions.csv",
    )
    parser.add_argument(
        "--output-match-diagnostics",
        type=Path,
        default=default_root / "data/interim/catboost_match_diagnostics.csv",
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


def candidate_summary(
    search: RandomizedSearchCV,
    selection_metric: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Return the best CV candidates in a compact JSON-friendly form."""

    results = pd.DataFrame(search.cv_results_).sort_values(
        ["rank_test_score", "mean_fit_time"]
    )
    summary: list[dict[str, object]] = []
    for _, row in results.head(limit).iterrows():
        summary.append(
            {
                "rank": int(row["rank_test_score"]),
                "selection_metric": selection_metric,
                "mean_cv_score": float(row["mean_test_score"]),
                "std_cv_score": float(row["std_test_score"]),
                "mean_fit_time_seconds": float(row["mean_fit_time"]),
                "params": row["params"],
            }
        )
    return summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_disjoint_pairs(
    match_ids: pd.Series,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    split_name: str,
) -> None:
    overlap = set(match_ids.iloc[left_indices]) & set(match_ids.iloc[right_indices])
    if overlap:
        raise ValueError(
            f"{split_name} separa {len(overlap)} coppie tra le due partizioni."
        )


def validate_mirrored_pairs(
    match_ids: pd.Series,
    target: pd.Series,
    dates: pd.Series | None = None,
) -> None:
    pairs = pd.DataFrame(
        {
            "match_id": match_ids.reset_index(drop=True),
            "target": target.reset_index(drop=True),
        }
    )
    if dates is not None:
        pairs["date"] = dates.reset_index(drop=True)
    grouped = pairs.groupby("match_id", sort=False)
    invalid_dates = dates is not None and not grouped["date"].nunique().eq(1).all()
    if (
        not grouped.size().eq(2).all()
        or not grouped["target"].sum().eq(1).all()
        or invalid_dates
    ):
        raise ValueError(
            "Ogni match_id_internal deve avere due orientamenti opposti sulla stessa data."
        )


def symmetrize_pair_probabilities(
    match_ids: pd.Series,
    target: pd.Series,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average both orientations and return player-A and real-winner probabilities."""

    pairs = pd.DataFrame(
        {
            "match_id": match_ids.reset_index(drop=True),
            "target": target.reset_index(drop=True),
            "probability": probabilities,
        }
    )
    validate_mirrored_pairs(match_ids, target)
    grouped = pairs.groupby("match_id", sort=False)

    pairs["actual_winner_probability"] = np.where(
        pairs["target"].eq(1),
        pairs["probability"],
        1 - pairs["probability"],
    )
    pair_winner_probability = grouped["actual_winner_probability"].transform("mean")
    symmetric_probability = np.where(
        pairs["target"].eq(1),
        pair_winner_probability,
        1 - pair_winner_probability,
    )
    return symmetric_probability, pair_winner_probability.to_numpy()


def probability_metrics(target: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "log_loss": float(log_loss(target, probabilities)),
        "brier_score": float(brier_score_loss(target, probabilities)),
    }


def main() -> None:
    args = parse_args()
    if args.search_iterations < 1 or args.early_stopping_rounds < 1:
        raise ValueError("Search iterations and early-stopping rounds must be positive.")

    excluded_features = parse_excluded_features(args.exclude_features)
    dataset = load_feature_dataset(args.input, args.metadata, excluded_features)
    source_frame = pd.read_csv(args.input)
    source_frame["date"] = pd.to_datetime(source_frame["date"], errors="coerce")
    source_frame = source_frame.dropna(subset=["date"]).reset_index(drop=True)
    if "match_id_internal" not in source_frame:
        raise ValueError("Il dataset deve contenere 'match_id_internal'.")
    if len(source_frame) != len(dataset.features):
        raise ValueError("Dataset e feature preparate non sono allineati.")
    validate_mirrored_pairs(
        source_frame["match_id_internal"], dataset.target, source_frame["date"]
    )

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
    all_match_ids = source_frame["match_id_internal"]
    ensure_disjoint_pairs(
        all_match_ids, train_indices, holdout_indices, "Holdout cronologico"
    )
    X_train = features.iloc[train_indices]
    y_train = dataset.target.iloc[train_indices]
    dates_train = dataset.dates.iloc[train_indices]
    X_holdout = features.iloc[holdout_indices]
    y_holdout = dataset.target.iloc[holdout_indices]
    dates_holdout = dataset.dates.iloc[holdout_indices]

    cv_folds = make_date_based_folds(dates_train.reset_index(drop=True), args.cv_folds)
    training_match_ids = all_match_ids.iloc[train_indices].reset_index(drop=True)
    for fold_number, (fold_train, fold_validation) in enumerate(cv_folds, start=1):
        ensure_disjoint_pairs(
            training_match_ids,
            fold_train,
            fold_validation,
            f"Fold CV {fold_number}",
        )
    print(f"Training set: {len(X_train):,} righe")
    print(
        f"Holdout finale: {len(X_holdout):,} righe "
        f"(ultimo {args.holdout_fraction:.0%} delle date)"
    )
    print(f"Fold di cross-validation: {len(cv_folds)}")

    common_model_params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "cat_features": dataset.categorical_features,
        "random_seed": args.random_seed,
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": 1,
    }
    model = CatBoostClassifier(
        # A tuple is intentionally used here. CatBoost copies lists internally,
        # which violates scikit-learn 1.9's strict estimator-cloning contract.
        **common_model_params,
    )
    parameter_grid = {
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_strength": args.random_strength,
        "bagging_temperature": args.bagging_temperature,
    }

    candidate_count = int(np.prod([len(values) for values in parameter_grid.values()]))
    sampled_candidates = min(args.search_iterations, candidate_count)
    print(f"Combinazioni disponibili: {candidate_count}")
    print(f"Combinazioni campionate: {sampled_candidates}")
    print("Random search in corso...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameter_grid,
        n_iter=sampled_candidates,
        scoring="neg_log_loss",
        cv=cv_folds,
        n_jobs=args.jobs,
        refit=False,
        random_state=args.random_seed,
        verbose=2,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    best_search_params = dict(search.best_params_)
    early_train_indices, early_validation_indices = cv_folds[-1]
    early_stopping_model = CatBoostClassifier(
        **common_model_params,
        **best_search_params,
    )
    early_stopping_model.fit(
        X_train.iloc[early_train_indices],
        y_train.iloc[early_train_indices],
        eval_set=(
            X_train.iloc[early_validation_indices],
            y_train.iloc[early_validation_indices],
        ),
        early_stopping_rounds=args.early_stopping_rounds,
        use_best_model=True,
    )
    best_iteration_count = int(early_stopping_model.tree_count_)
    best_params = {**best_search_params, "iterations": best_iteration_count}

    best_model = CatBoostClassifier(**common_model_params, **best_params)
    best_model.fit(X_train, y_train)
    raw_holdout_predictions = best_model.predict(X_holdout).astype(int).ravel()
    raw_holdout_probabilities = best_model.predict_proba(X_holdout)[:, 1]
    holdout_match_ids = source_frame.iloc[holdout_indices][
        "match_id_internal"
    ].reset_index(drop=True)
    holdout_probabilities, actual_winner_probabilities = (
        symmetrize_pair_probabilities(
            holdout_match_ids,
            y_holdout.reset_index(drop=True),
            raw_holdout_probabilities,
        )
    )
    holdout_predictions = (holdout_probabilities >= 0.5).astype(int)
    holdout_accuracy = accuracy_score(y_holdout, holdout_predictions)

    print(f"Migliori parametri della ricerca: {best_search_params}")
    print(f"Iterazioni dopo early stopping: {best_iteration_count}")
    print(f"Miglior neg_log_loss CV: {search.best_score_:.4f}")
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

    identifying_columns = [
        "date",
        "match_id_internal",
        "winner_name",
        "loser_name",
        "surface",
        "round",
        "series",
        "tournament_bucket",
        "winner_rank",
        "loser_rank",
        "winner_points",
        "loser_points",
        "normalized_market_prob_diff",
        "market_overround_avg",
        "elo_diff",
        "surface_elo_diff",
        "smoothed_win_rate_diff",
        "smoothed_recent5_diff",
        "smoothed_surface_diff",
        "smoothed_h2h_diff",
        "h2h_matches_before",
    ]
    holdout_details = source_frame.iloc[holdout_indices][
        [column for column in identifying_columns if column in source_frame]
    ].reset_index(drop=True)
    holdout_details = holdout_details.rename(
        columns={"match_id_internal": "match_pair_id"}
    )
    holdout_details["actual_player_a_win"] = y_holdout.to_numpy()
    holdout_details["predicted_player_a_win"] = raw_holdout_predictions
    holdout_details["player_a_win_probability"] = raw_holdout_probabilities
    holdout_details["correct"] = raw_holdout_predictions == y_holdout.to_numpy()
    holdout_details["confidence"] = np.maximum(
        raw_holdout_probabilities, 1 - raw_holdout_probabilities
    )
    holdout_details["actual_winner_probability"] = np.where(
        y_holdout.to_numpy() == 1,
        raw_holdout_probabilities,
        1 - raw_holdout_probabilities,
    )
    args.output_holdout_predictions.parent.mkdir(parents=True, exist_ok=True)
    holdout_details.to_csv(args.output_holdout_predictions, index=False)

    pair_probability_gap = holdout_details.groupby("match_pair_id")[
        "actual_winner_probability"
    ].transform(lambda values: values.max() - values.min())
    match_diagnostics = holdout_details.loc[
        holdout_details["actual_player_a_win"].eq(1)
    ].copy()
    winner_probability_by_row = pd.Series(actual_winner_probabilities)
    match_diagnostics["actual_winner_probability"] = winner_probability_by_row.loc[
        match_diagnostics.index
    ].to_numpy()
    match_diagnostics["confidence"] = match_diagnostics[
        "actual_winner_probability"
    ]
    match_diagnostics["correct"] = match_diagnostics["confidence"].ge(0.5)
    match_diagnostics["orientation_probability_gap"] = pair_probability_gap.loc[
        match_diagnostics.index
    ].to_numpy()
    if {"winner_name", "loser_name"}.issubset(match_diagnostics):
        match_diagnostics["predicted_winner_name"] = np.where(
            match_diagnostics["correct"],
            match_diagnostics["winner_name"],
            match_diagnostics["loser_name"],
        )
    args.output_match_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    match_diagnostics.to_csv(args.output_match_diagnostics, index=False)

    report = classification_report(
        y_holdout,
        holdout_predictions,
        output_dict=True,
    )
    market_baseline = None
    market_accuracy_lift = None
    if "winner_market_prob_normalized" in X_holdout:
        market_probabilities = pd.to_numeric(
            X_holdout["winner_market_prob_normalized"], errors="coerce"
        ).to_numpy()
        covered = np.isfinite(market_probabilities)
        if covered.any():
            market_baseline = {
                "rows": int(covered.sum()),
                "coverage": float(covered.mean()),
                **probability_metrics(
                    y_holdout.reset_index(drop=True).loc[covered],
                    market_probabilities[covered],
                ),
            }
            market_accuracy_lift = (
                float(holdout_accuracy) - market_baseline["accuracy"]
            )

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "CatBoostClassifier",
        "catboost_version": catboost.__version__,
        "input_path": str(args.input),
        "model_path": str(args.output_model),
        "training_script_sha256": file_sha256(Path(__file__)),
        "input_sha256": file_sha256(args.input),
        "metadata_sha256": file_sha256(args.metadata),
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
        "search_strategy": "random",
        "selection_metric": "neg_log_loss",
        "parameter_grid": parameter_grid,
        "best_search_params": best_search_params,
        "best_params": best_params,
        "best_cv_score": float(search.best_score_),
        "best_cv_accuracy": None,
        "early_stopping_rounds": args.early_stopping_rounds,
        "early_stopping_best_iteration": best_iteration_count,
        "pair_symmetry_enforced": True,
        "raw_holdout_accuracy": float(
            accuracy_score(y_holdout, raw_holdout_predictions)
        ),
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
        "market_baseline_holdout": market_baseline,
        "catboost_accuracy_lift_over_market": market_accuracy_lift,
        "top_feature_importances": top_importances,
        "top_cv_candidates": candidate_summary(search, "neg_log_loss"),
    }

    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Modello salvato in: {args.output_model}")
    print(f"Metriche salvate in: {args.output_metrics}")
    print("Pipeline completata.")


if __name__ == "__main__":
    main()
