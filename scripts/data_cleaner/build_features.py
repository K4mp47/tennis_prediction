from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ORDINAL_ROUND_ORDER = [
    "1st Round",
    "2nd Round",
    "3rd Round",
    "4th Round",
    "Quarterfinals",
    "Semifinals",
    "Final",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pre-match engineered features and run RFECV feature selection."
    )
    default_root = Path(__file__).resolve().parents[2]

    parser.add_argument(
        "--input",
        type=Path,
        default=default_root / "data" / "interim" / "tennis_matches_cleaned.data",
    )
    parser.add_argument(
        "--output-features",
        type=Path,
        default=default_root / "data" / "interim" / "tennis_matches_features.data",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=default_root / "data" / "interim" / "tennis_matches_features_metadata.json",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=20,
        help="Minimum count for tournament/location category before mapping to 'other'.",
    )
    parser.add_argument(
        "--rfecv-step",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--rfecv-min-features",
        type=int,
        default=8,
    )
    return parser.parse_args()


def load_clean_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in cleaned dataset")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    return df.sort_values(
        ["date", "source_year", "atp", "tournament", "winner_name", "loser_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def map_rare_categories(series: pd.Series, min_count: int) -> pd.Series:
    counts = series.value_counts(dropna=False)
    keep_values = set(counts[counts >= min_count].index)
    return series.where(series.isin(keep_values), "other").fillna("other")


def build_player_history_features(df: pd.DataFrame) -> pd.DataFrame:
    matches = df[["date", "winner_name", "loser_name", "surface", "court"]].copy()
    matches["match_id"] = np.arange(len(matches), dtype=np.int64)

    winner_view = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "player_name": matches["winner_name"],
            "surface": matches["surface"],
            "is_win": 1,
        }
    )

    loser_view = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "player_name": matches["loser_name"],
            "surface": matches["surface"],
            "is_win": 0,
        }
    )

    long_df = pd.concat([winner_view, loser_view], axis=0, ignore_index=True)
    long_df = long_df.sort_values(["player_name", "date", "match_id"], kind="mergesort")

    grouped = long_df.groupby("player_name", sort=False)

    long_df["matches_before"] = grouped.cumcount().astype(float)
    long_df["wins_before"] = grouped["is_win"].cumsum().shift(1).fillna(0.0)
    long_df["win_rate_before"] = np.where(
        long_df["matches_before"] > 0,
        long_df["wins_before"] / long_df["matches_before"],
        np.nan,
    )

    long_df["recent5_matches_before"] = grouped.cumcount().clip(upper=5).astype(float)
    long_df["recent5_wins_before"] = (
        grouped["is_win"]
        .rolling(window=5, min_periods=1)
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    long_df["recent5_win_rate_before"] = np.where(
        long_df["recent5_matches_before"] > 0,
        long_df["recent5_wins_before"] / long_df["recent5_matches_before"],
        np.nan,
    )

    surface_group = long_df.groupby(["player_name", "surface"], sort=False)
    long_df["surface_matches_before"] = surface_group.cumcount().astype(float)
    long_df["surface_wins_before"] = surface_group["is_win"].cumsum().shift(1).fillna(0.0)
    long_df["surface_win_rate_before"] = np.where(
        long_df["surface_matches_before"] > 0,
        long_df["surface_wins_before"] / long_df["surface_matches_before"],
        np.nan,
    )

    long_df["days_since_last_match"] = grouped["date"].diff().dt.days.astype(float)

    history_cols = [
        "matches_before",
        "wins_before",
        "win_rate_before",
        "recent5_matches_before",
        "recent5_wins_before",
        "recent5_win_rate_before",
        "surface_matches_before",
        "surface_wins_before",
        "surface_win_rate_before",
        "days_since_last_match",
    ]

    winner_hist = (
        long_df[["match_id", "player_name", *history_cols]]
        .rename(
            columns={
                "player_name": "winner_name",
                **{col: f"winner_{col}" for col in history_cols},
            }
        )
        .drop_duplicates(subset=["match_id", "winner_name"], keep="last")
    )

    loser_hist = (
        long_df[["match_id", "player_name", *history_cols]]
        .rename(
            columns={
                "player_name": "loser_name",
                **{col: f"loser_{col}" for col in history_cols},
            }
        )
        .drop_duplicates(subset=["match_id", "loser_name"], keep="last")
    )

    out = df.copy()
    out["match_id"] = np.arange(len(out), dtype=np.int64)
    out = out.merge(winner_hist, on=["match_id", "winner_name"], how="left")
    out = out.merge(loser_hist, on=["match_id", "loser_name"], how="left")
    out = out.drop(columns=["match_id"])

    return out


def build_head_to_head_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["pair_key"] = out.apply(
        lambda row: "||".join(sorted([str(row["winner_name"]), str(row["loser_name"])])),
        axis=1,
    )

    out["winner_is_player1"] = out.apply(
        lambda row: int(str(row["winner_name"]) <= str(row["loser_name"])),
        axis=1,
    )

    out = out.sort_values(["date", "pair_key"], kind="mergesort").reset_index(drop=True)

    out["h2h_matches_before"] = out.groupby("pair_key").cumcount().astype(float)

    out["winner_pair_win"] = out["winner_is_player1"].astype(float)
    out["pair_player1_wins_before"] = (
        out.groupby("pair_key")["winner_pair_win"].cumsum().shift(1).fillna(0.0)
    )

    out["winner_h2h_wins_before"] = np.where(
        out["winner_is_player1"] == 1,
        out["pair_player1_wins_before"],
        out["h2h_matches_before"] - out["pair_player1_wins_before"],
    )

    out["loser_h2h_wins_before"] = out["h2h_matches_before"] - out["winner_h2h_wins_before"]

    out["winner_h2h_win_rate_before"] = np.where(
        out["h2h_matches_before"] > 0,
        out["winner_h2h_wins_before"] / out["h2h_matches_before"],
        np.nan,
    )

    out["loser_h2h_win_rate_before"] = np.where(
        out["h2h_matches_before"] > 0,
        out["loser_h2h_wins_before"] / out["h2h_matches_before"],
        np.nan,
    )

    return out.drop(columns=["pair_key", "winner_is_player1", "winner_pair_win", "pair_player1_wins_before"])


def encode_round_ordinal(series: pd.Series) -> pd.Series:
    encoder = OrdinalEncoder(
        categories=[ORDINAL_ROUND_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    values = series.fillna("Unknown").to_numpy().reshape(-1, 1)
    encoded = encoder.fit_transform(values).reshape(-1)
    return pd.Series(encoded, index=series.index)


def add_model_features(df: pd.DataFrame, rare_threshold: int) -> pd.DataFrame:
    out = df.copy()

    out["round_ordinal"] = encode_round_ordinal(out["round"]).astype(float)

    out["winner_rank_missing"] = out["winner_rank"].isna().astype(int)
    out["loser_rank_missing"] = out["loser_rank"].isna().astype(int)
    out["winner_points_missing"] = out["winner_points"].isna().astype(int)
    out["loser_points_missing"] = out["loser_points"].isna().astype(int)

    out["rank_diff"] = out["loser_rank"] - out["winner_rank"]
    out["points_diff"] = out["winner_points"] - out["loser_points"]

    out["winner_implied_prob_avg"] = 1.0 / out["avg_winner"]
    out["loser_implied_prob_avg"] = 1.0 / out["avg_loser"]
    out.loc[out["avg_winner"].isna(), "winner_implied_prob_avg"] = np.nan
    out.loc[out["avg_loser"].isna(), "loser_implied_prob_avg"] = np.nan
    out["implied_prob_diff_avg"] = out["winner_implied_prob_avg"] - out["loser_implied_prob_avg"]

    history_pairs = [
        ("matches_before", "experience_diff"),
        ("win_rate_before", "win_rate_diff"),
        ("recent5_win_rate_before", "recent5_win_rate_diff"),
        ("surface_win_rate_before", "surface_win_rate_diff"),
        ("days_since_last_match", "rest_days_diff"),
    ]

    for suffix, out_col in history_pairs:
        out[out_col] = out[f"winner_{suffix}"] - out[f"loser_{suffix}"]

    out["tournament_bucket"] = map_rare_categories(out["tournament"].fillna("Unknown"), rare_threshold)
    out["location_bucket"] = map_rare_categories(out["location"].fillna("Unknown"), rare_threshold)

    out["player_a_win"] = 1

    out = out.sort_values("date", kind="mergesort").reset_index(drop=True)
    return out


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    categorical_features = [
        "series",
        "court",
        "surface",
        "tournament_bucket",
        "location_bucket",
    ]

    numeric_features = [
        "source_year",
        "atp",
        "best_of",
        "winner_rank",
        "loser_rank",
        "winner_points",
        "loser_points",
        "winner_rank_missing",
        "loser_rank_missing",
        "winner_points_missing",
        "loser_points_missing",
        "rank_diff",
        "points_diff",
        "has_any_market_odds",
        "winner_implied_prob_avg",
        "loser_implied_prob_avg",
        "implied_prob_diff_avg",
        "round_ordinal",
        "winner_matches_before",
        "loser_matches_before",
        "experience_diff",
        "winner_win_rate_before",
        "loser_win_rate_before",
        "win_rate_diff",
        "winner_recent5_win_rate_before",
        "loser_recent5_win_rate_before",
        "recent5_win_rate_diff",
        "winner_surface_win_rate_before",
        "loser_surface_win_rate_before",
        "surface_win_rate_diff",
        "winner_days_since_last_match",
        "loser_days_since_last_match",
        "rest_days_diff",
        "h2h_matches_before",
        "winner_h2h_win_rate_before",
        "loser_h2h_win_rate_before",
    ]

    available_categorical = [c for c in categorical_features if c in df.columns]
    available_numeric = [c for c in numeric_features if c in df.columns]

    X = df[available_numeric + available_categorical].copy()
    y = df["player_a_win"].astype(int)

    return X, y, available_numeric, available_categorical


def run_baseline_and_rfecv(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    rfecv_step: int,
    rfecv_min_features: int,
) -> dict:
    split_count = max(3, min(5, len(X) // 500))
    cv = TimeSeriesSplit(n_splits=split_count)

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
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

    baseline_model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    baseline_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", baseline_model),
        ]
    )

    baseline_scores = cross_val_score(
        baseline_pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    prefit = preprocess.fit(X, y)
    X_prepared = prefit.transform(X)
    feature_names = prefit.get_feature_names_out().tolist()

    selector_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    rfecv = RFECV(
        estimator=selector_model,
        step=rfecv_step,
        min_features_to_select=min(rfecv_min_features, max(1, X_prepared.shape[1] - 1)),
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    rfecv.fit(X_prepared, y)

    support_mask = rfecv.support_
    selected_feature_names = [name for name, keep in zip(feature_names, support_mask) if keep]

    baseline_pipeline.fit(X, y)
    train_predictions = baseline_pipeline.predict(X)

    return {
        "samples": int(len(X)),
        "baseline_cv_accuracy_mean": float(np.mean(baseline_scores)),
        "baseline_cv_accuracy_std": float(np.std(baseline_scores)),
        "baseline_train_accuracy": float(accuracy_score(y, train_predictions)),
        "baseline_feature_count": int(len(feature_names)),
        "rfecv_selected_feature_count": int(rfecv.n_features_),
        "rfecv_selected_features": selected_feature_names,
        "rfecv_best_cv_accuracy": float(np.max(rfecv.cv_results_["mean_test_score"])),
        "rfecv_cv_accuracy_path": [float(v) for v in rfecv.cv_results_["mean_test_score"]],
    }


def main() -> None:
    args = parse_args()

    cleaned_df = load_clean_data(args.input)
    with_hist = build_player_history_features(cleaned_df)
    with_h2h = build_head_to_head_features(with_hist)
    featured_df = add_model_features(with_h2h, rare_threshold=args.rare_threshold)

    X, y, numeric_features, categorical_features = build_feature_matrix(featured_df)

    metrics = run_baseline_and_rfecv(
        X,
        y,
        numeric_features,
        categorical_features,
        rfecv_step=args.rfecv_step,
        rfecv_min_features=args.rfecv_min_features,
    )

    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(args.output_features, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "output_features_path": str(args.output_features),
        "output_rows": int(len(featured_df)),
        "output_columns": list(featured_df.columns),
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "rare_threshold": int(args.rare_threshold),
        "rfecv_step": int(args.rfecv_step),
        "rfecv_min_features": int(args.rfecv_min_features),
        "metrics": metrics,
    }

    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Created feature dataset: {args.output_features}")
    print(f"Created feature metadata: {args.output_metadata}")
    print(f"Baseline CV accuracy (mean): {metrics['baseline_cv_accuracy_mean']:.4f}")
    print(f"RFECV selected features: {metrics['rfecv_selected_feature_count']}")


if __name__ == "__main__":
    main()
