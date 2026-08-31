"""Score pre-engineered tennis match rows with a trained CatBoost model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from train_catboost import prepare_catboost_features


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Predict whether oriented player A wins. Input rows must contain the "
            "engineered features declared in the CatBoost metrics file."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=default_root / "data" / "interim" / "catboost_model.cbm",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=default_root / "data" / "interim" / "catboost_metrics.json",
        help="Training metrics containing the exact feature schema",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "data" / "interim" / "catboost_predictions.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to predict player A (default: 0.5)",
    )
    return parser.parse_args()


def score_rows(
    frame: pd.DataFrame,
    model: CatBoostClassifier,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return the input rows with player-A probabilities and winner labels."""

    if not 0 < threshold < 1:
        raise ValueError("The prediction threshold must be between 0 and 1.")

    feature_names = numeric_features + categorical_features
    missing_features = sorted(set(feature_names).difference(frame.columns))
    if missing_features:
        raise ValueError(
            "Input is missing model features: " + ", ".join(missing_features)
        )

    prepared = prepare_catboost_features(
        frame.loc[:, feature_names], numeric_features, categorical_features
    )
    player_a_probability = model.predict_proba(prepared)[:, 1]
    player_a_prediction = (player_a_probability >= threshold).astype(int)

    scored = frame.copy()
    scored["player_a_win_probability"] = player_a_probability
    scored["player_b_win_probability"] = 1 - player_a_probability
    scored["predicted_player_a_win"] = player_a_prediction
    scored["predicted_winner_side"] = np.where(
        player_a_prediction == 1, "player_a", "player_b"
    )

    # Symmetric feature datasets retain the historical winner/loser column
    # names, although they represent oriented player A/player B after swapping.
    if {"winner_name", "loser_name"}.issubset(scored.columns):
        scored["predicted_winner_name"] = np.where(
            player_a_prediction == 1,
            scored["winner_name"],
            scored["loser_name"],
        )

    return scored


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    numeric_features = tuple(metrics["numeric_features"])
    categorical_features = tuple(metrics["categorical_features"])

    model = CatBoostClassifier()
    model.load_model(str(args.model))
    input_frame = pd.read_csv(args.input)
    scored = score_rows(
        input_frame,
        model,
        numeric_features,
        categorical_features,
        args.threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"Scored rows: {len(scored):,}")
    print(f"Predictions: {args.output}")


if __name__ == "__main__":
    main()
