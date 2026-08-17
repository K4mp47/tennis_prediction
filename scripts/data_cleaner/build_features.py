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


# ---------------------------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pre-match engineered features from the enriched "
            "Tennis-Data/Sackmann dataset and run RFECV."
        )
    )

    default_root = Path(__file__).resolve().parents[2]

    parser.add_argument(
        "--input",
        type=Path,
        default=(
                default_root
                / "data"
                / "interim"
                / "tennis_matches_enriched.data"
        ),
        help="Input enriched dataset.",
    )

    parser.add_argument(
        "--output-features",
        type=Path,
        default=(
                default_root
                / "data"
                / "interim"
                / "tennis_matches_features.data"
        ),
    )

    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=(
                default_root
                / "data"
                / "interim"
                / "tennis_matches_features_metadata.json"
        ),
    )

    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=20,
        help=(
            "Minimum number of observations required to keep a "
            "tournament/location category."
        ),
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


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_clean_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column.")

    required = [
        "winner_name",
        "loser_name",
        "surface",
        "round",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.dropna(subset=["date"]).copy()

    sort_cols = [
        c
        for c in [
            "date",
            "source_year",
            "atp",
            "tournament",
            "winner_name",
            "loser_name",
        ]
        if c in df.columns
    ]

    df = (
        df.sort_values(
            sort_cols,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

def map_rare_categories(
        series: pd.Series,
        min_count: int,
) -> pd.Series:
    series = series.astype("object")

    counts = series.value_counts(
        dropna=False,
    )

    keep_values = set(
        counts[counts >= min_count].index
    )

    return (
        series.where(
            series.isin(keep_values),
            "other",
        )
        .fillna("other")
    )


def encode_round_ordinal(
        series: pd.Series,
) -> pd.Series:
    encoder = OrdinalEncoder(
        categories=[ORDINAL_ROUND_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    values = (
        series
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
        .reshape(-1, 1)
    )

    encoded = encoder.fit_transform(values).reshape(-1)

    return pd.Series(
        encoded,
        index=series.index,
        dtype=float,
    )


# ---------------------------------------------------------------------------
# PLAYER HISTORY
# ---------------------------------------------------------------------------

def build_player_history_features(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes historical player features strictly BEFORE each match.

    Important:
    We never use the current match outcome when calculating a feature
    for that same match.
    """

    out = df.copy()

    out["match_id_internal"] = np.arange(
        len(out),
        dtype=np.int64,
    )

    matches = out[
        [
            "match_id_internal",
            "date",
            "winner_name",
            "loser_name",
            "surface",
        ]
    ].copy()

    winner_view = pd.DataFrame(
        {
            "match_id_internal": matches["match_id_internal"],
            "date": matches["date"],
            "player_name": matches["winner_name"],
            "surface": matches["surface"],
            "is_win": 1.0,
        }
    )

    loser_view = pd.DataFrame(
        {
            "match_id_internal": matches["match_id_internal"],
            "date": matches["date"],
            "player_name": matches["loser_name"],
            "surface": matches["surface"],
            "is_win": 0.0,
        }
    )

    long_df = pd.concat(
        [winner_view, loser_view],
        ignore_index=True,
    )

    long_df = long_df.sort_values(
        [
            "player_name",
            "date",
            "match_id_internal",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # GLOBAL HISTORY
    # ------------------------------------------------------------------

    player_group = long_df.groupby(
        "player_name",
        sort=False,
    )

    long_df["matches_before"] = (
        player_group.cumcount().astype(float)
    )

    # CRITICAL:
    # cumulative wins MINUS current match result.
    # This avoids the previous global .shift(1) bug.
    long_df["wins_before"] = (
            player_group["is_win"]
            .cumsum()
            - long_df["is_win"]
    )

    long_df["win_rate_before"] = np.where(
        long_df["matches_before"] > 0,
        long_df["wins_before"]
        / long_df["matches_before"],
        np.nan,
        )

    # ------------------------------------------------------------------
    # RECENT 5 MATCHES
    # ------------------------------------------------------------------

    long_df["recent5_matches_before"] = (
        long_df.groupby(
            "player_name",
            sort=False,
        )
        .cumcount()
        .clip(upper=5)
        .astype(float)
    )

    long_df["recent5_wins_before"] = (
        long_df.groupby(
            "player_name",
            sort=False,
        )["is_win"]
        .transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    window=5,
                    min_periods=1,
                )
                .sum()
            )
        )
        .fillna(0.0)
    )

    long_df["recent5_win_rate_before"] = np.where(
        long_df["recent5_matches_before"] > 0,
        long_df["recent5_wins_before"]
        / long_df["recent5_matches_before"],
        np.nan,
        )

    # ------------------------------------------------------------------
    # SURFACE HISTORY
    # ------------------------------------------------------------------

    surface_group = long_df.groupby(
        ["player_name", "surface"],
        sort=False,
    )

    long_df["surface_matches_before"] = (
        surface_group.cumcount().astype(float)
    )

    long_df["surface_wins_before"] = (
            surface_group["is_win"]
            .cumsum()
            - long_df["is_win"]
    )

    long_df["surface_win_rate_before"] = np.where(
        long_df["surface_matches_before"] > 0,
        long_df["surface_wins_before"]
        / long_df["surface_matches_before"],
        np.nan,
        )

    # ------------------------------------------------------------------
    # REST
    # ------------------------------------------------------------------

    long_df["days_since_last_match"] = (
        long_df.groupby(
            "player_name",
            sort=False,
        )["date"]
        .diff()
        .dt.days
        .astype(float)
    )

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
        long_df[
            [
                "match_id_internal",
                "player_name",
                *history_cols,
            ]
        ]
        .rename(
            columns={
                "player_name": "winner_name",
                **{
                    c: f"winner_{c}"
                    for c in history_cols
                },
            }
        )
        .drop_duplicates(
            subset=[
                "match_id_internal",
                "winner_name",
            ],
            keep="last",
        )
    )

    loser_hist = (
        long_df[
            [
                "match_id_internal",
                "player_name",
                *history_cols,
            ]
        ]
        .rename(
            columns={
                "player_name": "loser_name",
                **{
                    c: f"loser_{c}"
                    for c in history_cols
                },
            }
        )
        .drop_duplicates(
            subset=[
                "match_id_internal",
                "loser_name",
            ],
            keep="last",
        )
    )

    out = out.merge(
        winner_hist,
        on=[
            "match_id_internal",
            "winner_name",
        ],
        how="left",
    )

    out = out.merge(
        loser_hist,
        on=[
            "match_id_internal",
            "loser_name",
        ],
        how="left",
    )

    return out


# ---------------------------------------------------------------------------
# HEAD TO HEAD
# ---------------------------------------------------------------------------

def build_head_to_head_features(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    H2H features are calculated strictly before the current match.

    No global shift is used, because that can move information from one
    player pair to another pair.
    """

    out = df.copy()

    out["pair_key"] = (
        out.apply(
            lambda row: "||".join(
                sorted(
                    [
                        str(row["winner_name"]),
                        str(row["loser_name"]),
                    ]
                )
            ),
            axis=1,
        )
    )

    out["winner_is_player1"] = (
        out.apply(
            lambda row: int(
                str(row["winner_name"])
                <= str(row["loser_name"])
            ),
            axis=1,
        )
    )

    out = out.sort_values(
        [
            "pair_key",
            "date",
            "match_id_internal",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    pair_group = out.groupby(
        "pair_key",
        sort=False,
    )

    out["h2h_matches_before"] = (
        pair_group.cumcount().astype(float)
    )

    # Current winner is player 1 -> 1
    # Current winner is player 2 -> 0
    current_player1_win = (
        out["winner_is_player1"].astype(float)
    )

    # Again: cumulative result minus current result.
    out["pair_player1_wins_before"] = (
            pair_group["winner_is_player1"]
            .cumsum()
            - current_player1_win
    )

    out["winner_h2h_wins_before"] = np.where(
        out["winner_is_player1"] == 1,
        out["pair_player1_wins_before"],
        (
                out["h2h_matches_before"]
                - out["pair_player1_wins_before"]
        ),
        )

    out["loser_h2h_wins_before"] = (
            out["h2h_matches_before"]
            - out["winner_h2h_wins_before"]
    )

    out["winner_h2h_win_rate_before"] = np.where(
        out["h2h_matches_before"] > 0,
        out["winner_h2h_wins_before"]
        / out["h2h_matches_before"],
        np.nan,
        )

    out["loser_h2h_win_rate_before"] = np.where(
        out["h2h_matches_before"] > 0,
        out["loser_h2h_wins_before"]
        / out["h2h_matches_before"],
        np.nan,
        )

    return out.drop(
        columns=[
            "pair_key",
            "winner_is_player1",
            "pair_player1_wins_before",
        ]
    )


# ---------------------------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------------------------

def add_model_features(
        df: pd.DataFrame,
        rare_threshold: int,
) -> pd.DataFrame:
    out = df.copy()

    # ---------------------------------------------------------------
    # ROUND
    # ---------------------------------------------------------------

    if "round" in out.columns:
        out["round_ordinal"] = encode_round_ordinal(
            out["round"]
        )

    # ---------------------------------------------------------------
    # MISSING FLAGS
    # ---------------------------------------------------------------

    for column in [
        "winner_rank",
        "loser_rank",
        "winner_points",
        "loser_points",
    ]:
        if column in out.columns:
            out[f"{column}_missing"] = (
                out[column].isna().astype(int)
            )

    # ---------------------------------------------------------------
    # RANK / POINT DIFFERENCES
    # ---------------------------------------------------------------

    if {
        "winner_rank",
        "loser_rank",
    }.issubset(out.columns):
        out["rank_diff"] = (
                out["loser_rank"]
                - out["winner_rank"]
        )

    if {
        "winner_points",
        "loser_points",
    }.issubset(out.columns):
        out["points_diff"] = (
                out["winner_points"]
                - out["loser_points"]
        )

    # ---------------------------------------------------------------
    # BETTING MARKET
    # ---------------------------------------------------------------

    if "avg_winner" in out.columns:
        out["winner_implied_prob_avg"] = (
                1.0 / out["avg_winner"]
        )

    if "avg_loser" in out.columns:
        out["loser_implied_prob_avg"] = (
                1.0 / out["avg_loser"]
        )

    if {
        "winner_implied_prob_avg",
        "loser_implied_prob_avg",
    }.issubset(out.columns):
        out["implied_prob_diff_avg"] = (
                out["winner_implied_prob_avg"]
                - out["loser_implied_prob_avg"]
        )

    # ---------------------------------------------------------------
    # PLAYER HISTORY DIFFERENCES
    # ---------------------------------------------------------------

    history_pairs = [
        (
            "matches_before",
            "experience_diff",
        ),
        (
            "win_rate_before",
            "win_rate_diff",
        ),
        (
            "recent5_win_rate_before",
            "recent5_win_rate_diff",
        ),
        (
            "surface_win_rate_before",
            "surface_win_rate_diff",
        ),
        (
            "days_since_last_match",
            "rest_days_diff",
        ),
        (
            "h2h_win_rate_before",
            "h2h_win_rate_diff",
        ),
    ]

    for suffix, output_column in history_pairs:
        winner_col = f"winner_{suffix}"
        loser_col = f"loser_{suffix}"

        if {
            winner_col,
            loser_col,
        }.issubset(out.columns):
            out[output_column] = (
                    out[winner_col]
                    - out[loser_col]
            )

    # ---------------------------------------------------------------
    # SACKMANN DIFFERENCES
    # ---------------------------------------------------------------

    sackmann_pairs = [
        (
            "height",
            "height_diff",
        ),
        (
            "age_years",
            "age_diff",
        ),
        (
            "ace_rate_before",
            "ace_rate_diff",
        ),
        (
            "df_rate_before",
            "df_rate_diff",
        ),
        (
            "first_in_rate_before",
            "first_in_rate_diff",
        ),
        (
            "first_won_rate_before",
            "first_won_rate_diff",
        ),
        (
            "second_won_rate_before",
            "second_won_rate_diff",
        ),
        (
            "bp_saved_rate_before",
            "bp_saved_rate_diff",
        ),
        (
            "bp_faced_per_sv_game_before",
            "bp_faced_per_sv_game_diff",
        ),
    ]

    for suffix, output_column in sackmann_pairs:
        winner_col = f"winner_{suffix}"
        loser_col = f"loser_{suffix}"

        if {
            winner_col,
            loser_col,
        }.issubset(out.columns):
            out[output_column] = (
                    out[winner_col]
                    - out[loser_col]
            )

    # ---------------------------------------------------------------
    # CATEGORICAL BUCKETS
    # ---------------------------------------------------------------

    if "tournament" in out.columns:
        out["tournament_bucket"] = map_rare_categories(
            out["tournament"],
            rare_threshold,
        )

    if "location" in out.columns:
        out["location_bucket"] = map_rare_categories(
            out["location"],
            rare_threshold,
        )

    out = out.sort_values(
        [
            "date",
            "match_id_internal",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# SYMMETRIC DATASET
# ---------------------------------------------------------------------------

def make_symmetric_dataset(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converts each real match:

        winner vs loser -> player_a_win = 1

    into two observations:

        winner vs loser -> 1
        loser vs winner -> 0

    This removes dependence on the arbitrary winner/loser orientation.
    """

    a_win = df.copy()
    a_win["player_a_win"] = 1

    a_lose = df.copy()
    a_lose["player_a_win"] = 0

    # ---------------------------------------------------------------
    # Swap player-specific variables
    # ---------------------------------------------------------------

    swap_pairs = [
        ("winner_rank", "loser_rank"),
        ("winner_points", "loser_points"),
        ("winner_rank_missing", "loser_rank_missing"),
        ("winner_points_missing", "loser_points_missing"),

        ("winner_matches_before", "loser_matches_before"),
        ("winner_wins_before", "loser_wins_before"),
        ("winner_win_rate_before", "loser_win_rate_before"),

        (
            "winner_recent5_matches_before",
            "loser_recent5_matches_before",
        ),
        (
            "winner_recent5_wins_before",
            "loser_recent5_wins_before",
        ),
        (
            "winner_recent5_win_rate_before",
            "loser_recent5_win_rate_before",
        ),

        (
            "winner_surface_matches_before",
            "loser_surface_matches_before",
        ),
        (
            "winner_surface_wins_before",
            "loser_surface_wins_before",
        ),
        (
            "winner_surface_win_rate_before",
            "loser_surface_win_rate_before",
        ),

        (
            "winner_days_since_last_match",
            "loser_days_since_last_match",
        ),

        (
            "winner_h2h_wins_before",
            "loser_h2h_wins_before",
        ),
        (
            "winner_h2h_win_rate_before",
            "loser_h2h_win_rate_before",
        ),

        (
            "winner_height",
            "loser_height",
        ),
        (
            "winner_age_years",
            "loser_age_years",
        ),

        (
            "winner_ace_rate_before",
            "loser_ace_rate_before",
        ),
        (
            "winner_df_rate_before",
            "loser_df_rate_before",
        ),
        (
            "winner_first_in_rate_before",
            "loser_first_in_rate_before",
        ),
        (
            "winner_first_won_rate_before",
            "loser_first_won_rate_before",
        ),
        (
            "winner_second_won_rate_before",
            "loser_second_won_rate_before",
        ),
        (
            "winner_bp_saved_rate_before",
            "loser_bp_saved_rate_before",
        ),
        (
            "winner_bp_faced_per_sv_game_before",
            "loser_bp_faced_per_sv_game_before",
        ),

        (
            "winner_implied_prob_avg",
            "loser_implied_prob_avg",
        ),
    ]

    for left, right in swap_pairs:
        if left in a_lose.columns and right in a_lose.columns:
            tmp = a_lose[left].copy()
            a_lose[left] = a_lose[right].copy()
            a_lose[right] = tmp

    # ---------------------------------------------------------------
    # Swap player names / Sackmann IDs / hand
    # ---------------------------------------------------------------

    generic_swap_pairs = [
        ("winner_name", "loser_name"),
        ("winner_sackmann_id", "loser_sackmann_id"),
        ("winner_hand", "loser_hand"),
        ("winner_ioc", "loser_ioc"),
    ]

    for left, right in generic_swap_pairs:
        if left in a_lose.columns and right in a_lose.columns:
            tmp = a_lose[left].copy()
            a_lose[left] = a_lose[right].copy()
            a_lose[right] = tmp

    # ---------------------------------------------------------------
    # Flip difference features
    # ---------------------------------------------------------------

    sign_flip_columns = [
        "rank_diff",
        "points_diff",
        "implied_prob_diff_avg",

        "experience_diff",
        "win_rate_diff",
        "recent5_win_rate_diff",
        "surface_win_rate_diff",
        "rest_days_diff",
        "h2h_win_rate_diff",

        "height_diff",
        "age_diff",
        "ace_rate_diff",
        "df_rate_diff",
        "first_in_rate_diff",
        "first_won_rate_diff",
        "second_won_rate_diff",
        "bp_saved_rate_diff",
        "bp_faced_per_sv_game_diff",
    ]

    for column in sign_flip_columns:
        if column in a_lose.columns:
            a_lose[column] = -a_lose[column]

    out = pd.concat(
        [
            a_win,
            a_lose,
        ],
        ignore_index=True,
    )

    out = out.sort_values(
        [
            "date",
            "match_id_internal",
            "player_a_win",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# FEATURE MATRIX
# ---------------------------------------------------------------------------

def build_feature_matrix(
        df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
    list[str],
]:
    categorical_features = [
        "series",
        "court",
        "surface",
        "tournament_bucket",
        "location_bucket",
    ]

    numeric_features = [
        # Tournament/match
        "source_year",
        "atp",
        "best_of",
        "round_ordinal",

        # Ranking
        "winner_rank",
        "loser_rank",
        "rank_diff",

        # Ranking missingness
        "winner_rank_missing",
        "loser_rank_missing",

        # Points
        "winner_points",
        "loser_points",
        "points_diff",
        "winner_points_missing",
        "loser_points_missing",

        # Market
        "has_any_market_odds",
        "winner_implied_prob_avg",
        "loser_implied_prob_avg",
        "implied_prob_diff_avg",

        # Global player history
        "winner_matches_before",
        "loser_matches_before",
        "experience_diff",

        "winner_win_rate_before",
        "loser_win_rate_before",
        "win_rate_diff",

        # Recent form
        "winner_recent5_win_rate_before",
        "loser_recent5_win_rate_before",
        "recent5_win_rate_diff",

        # Surface
        "winner_surface_win_rate_before",
        "loser_surface_win_rate_before",
        "surface_win_rate_diff",

        # Rest
        "winner_days_since_last_match",
        "loser_days_since_last_match",
        "rest_days_diff",

        # H2H
        "h2h_matches_before",
        "winner_h2h_win_rate_before",
        "loser_h2h_win_rate_before",
        "h2h_win_rate_diff",

        # Sackmann physical
        "winner_height",
        "loser_height",
        "height_diff",

        "winner_age_years",
        "loser_age_years",
        "age_diff",

        # Sackmann serve/return
        "winner_ace_rate_before",
        "loser_ace_rate_before",
        "ace_rate_diff",

        "winner_df_rate_before",
        "loser_df_rate_before",
        "df_rate_diff",

        "winner_first_in_rate_before",
        "loser_first_in_rate_before",
        "first_in_rate_diff",

        "winner_first_won_rate_before",
        "loser_first_won_rate_before",
        "first_won_rate_diff",

        "winner_second_won_rate_before",
        "loser_second_won_rate_before",
        "second_won_rate_diff",

        "winner_bp_saved_rate_before",
        "loser_bp_saved_rate_before",
        "bp_saved_rate_diff",

        "winner_bp_faced_per_sv_game_before",
        "loser_bp_faced_per_sv_game_before",
        "bp_faced_per_sv_game_diff",
    ]

    available_categorical = [
        c
        for c in categorical_features
        if c in df.columns
    ]

    available_numeric = [
        c
        for c in numeric_features
        if c in df.columns
    ]

    if "player_a_win" not in df.columns:
        raise ValueError(
            "Missing target column 'player_a_win'."
        )

    X = df[
        available_numeric
        + available_categorical
        ].copy()

    y = df["player_a_win"].astype(int)

    return (
        X,
        y,
        available_numeric,
        available_categorical,
    )


# ---------------------------------------------------------------------------
# MODEL / RFECV
# ---------------------------------------------------------------------------

def run_baseline_and_rfecv(
        X: pd.DataFrame,
        y: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
        rfecv_step: int,
        rfecv_min_features: int,
) -> dict:

    if len(X) < 1000:
        split_count = 3
    elif len(X) < 5000:
        split_count = 4
    else:
        split_count = 5

    cv = TimeSeriesSplit(
        n_splits=split_count
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                            ),
                        ),
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
        min_samples_leaf=2,
    )

    baseline_pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                preprocess,
            ),
            (
                "model",
                baseline_model,
            ),
        ]
    )

    print()
    print("MODEL BASELINE")
    print(
        f"TimeSeriesSplit: {split_count} folds"
    )

    baseline_scores = cross_val_score(
        baseline_pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    # ---------------------------------------------------------------
    # RFECV
    #
    # We transform once here only for feature-selection diagnostics.
    # The baseline CV above remains the primary unbiased estimate.
    # ---------------------------------------------------------------

    print()
    print("Preparing RFECV matrix...")

    prefit = preprocess.fit(X, y)

    X_prepared = prefit.transform(X)

    feature_names = (
        prefit
        .get_feature_names_out()
        .tolist()
    )

    selector_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    max_features = max(
        1,
        X_prepared.shape[1] - 1,
        )

    min_features = min(
        rfecv_min_features,
        max_features,
    )

    print(
        f"Encoded feature count: "
        f"{X_prepared.shape[1]}"
    )

    print(
        f"RFECV minimum features: "
        f"{min_features}"
    )

    rfecv = RFECV(
        estimator=selector_model,
        step=rfecv_step,
        min_features_to_select=min_features,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )

    rfecv.fit(
        X_prepared,
        y,
    )

    support_mask = rfecv.support_

    selected_feature_names = [
        name
        for name, keep in zip(
            feature_names,
            support_mask,
        )
        if keep
    ]

    # ---------------------------------------------------------------
    # Train accuracy is diagnostic only.
    # Do NOT use it as generalisation performance.
    # ---------------------------------------------------------------

    baseline_pipeline.fit(
        X,
        y,
    )

    train_predictions = (
        baseline_pipeline.predict(X)
    )

    return {
        "samples": int(len(X)),
        "cv_folds": int(split_count),

        "baseline_cv_accuracy_mean": float(
            np.mean(baseline_scores)
        ),

        "baseline_cv_accuracy_std": float(
            np.std(baseline_scores)
        ),

        "baseline_cv_accuracy_folds": [
            float(v)
            for v in baseline_scores
        ],

        "baseline_train_accuracy": float(
            accuracy_score(
                y,
                train_predictions,
            )
        ),

        "baseline_feature_count": int(
            len(feature_names)
        ),

        "rfecv_selected_feature_count": int(
            rfecv.n_features_
        ),

        "rfecv_selected_features":
            selected_feature_names,

        "rfecv_best_cv_accuracy": float(
            np.max(
                rfecv.cv_results_[
                    "mean_test_score"
                ]
            )
        ),

        "rfecv_cv_accuracy_path": [
            float(v)
            for v in
            rfecv.cv_results_[
                "mean_test_score"
            ]
        ],
    }


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_features(
        df: pd.DataFrame,
) -> None:

    print()
    print("FEATURE VALIDATION")

    problems = []

    # Probability/rate columns should remain in [0, 1].
    rate_columns = [
        c
        for c in df.columns
        if (
                c.endswith("_rate_before")
                or c.endswith("_win_rate_before")
                or c.endswith("_rate_diff")
        )
    ]

    for column in rate_columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        if column.endswith("_diff"):
            continue

        invalid = (
                (values < 0)
                | (values > 1)
        ).sum()

        if invalid:
            problems.append(
                f"{column}: "
                f"{invalid} values outside [0,1]"
            )

    # Age sanity check.
    for column in [
        "winner_age_years",
        "loser_age_years",
    ]:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        invalid = (
                (values < 14)
                | (values > 60)
        ).sum()

        if invalid:
            problems.append(
                f"{column}: "
                f"{invalid} implausible ages"
            )

    if problems:
        print("PROBLEMS FOUND:")

        for problem in problems:
            print(f"  - {problem}")

        raise ValueError(
            "Feature validation failed."
        )

    print("OK - nessuna anomalia rilevante.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("BUILD FEATURES")
    print("=" * 70)

    print()
    print(
        f"Input: {args.input}"
    )

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    df = load_clean_data(
        args.input
    )

    print(
        f"Match caricati: {len(df):,}"
    )

    # ---------------------------------------------------------------
    # Historical player features
    # ---------------------------------------------------------------

    print()
    print(
        "Costruzione storico giocatori..."
    )

    df = build_player_history_features(
        df
    )

    # ---------------------------------------------------------------
    # H2H
    # ---------------------------------------------------------------

    print(
        "Costruzione feature H2H..."
    )

    df = build_head_to_head_features(
        df
    )

    # ---------------------------------------------------------------
    # Model features
    # ---------------------------------------------------------------

    print(
        "Costruzione feature modello..."
    )

    df = add_model_features(
        df,
        rare_threshold=args.rare_threshold,
    )

    # ---------------------------------------------------------------
    # Validation before symmetry
    # ---------------------------------------------------------------

    validate_features(df)

    # ---------------------------------------------------------------
    # Symmetric representation
    # ---------------------------------------------------------------

    print()
    print(
        "Costruzione dataset simmetrico..."
    )

    df = make_symmetric_dataset(
        df
    )

    print(
        f"Righe finali dopo simmetrizzazione: "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------------
    # Feature matrix
    # ---------------------------------------------------------------

    X, y, numeric_features, categorical_features = (
        build_feature_matrix(df)
    )

    print()
    print(
        f"Feature numeriche: "
        f"{len(numeric_features)}"
    )

    print(
        f"Feature categoriche: "
        f"{len(categorical_features)}"
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    metrics = run_baseline_and_rfecv(
        X,
        y,
        numeric_features,
        categorical_features,
        rfecv_step=args.rfecv_step,
        rfecv_min_features=args.rfecv_min_features,
    )

    # ---------------------------------------------------------------
    # Remove internal column from saved dataset
    # ---------------------------------------------------------------

    if "match_id_internal" in df.columns:
        df = df.drop(
            columns=["match_id_internal"]
        )

    # ---------------------------------------------------------------
    # Save features
    # ---------------------------------------------------------------

    args.output_features.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        args.output_features,
        index=False,
    )

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------

    metadata = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "input_path": str(
            args.input
        ),

        "output_features_path": str(
            args.output_features
        ),

        "output_rows": int(
            len(df)
        ),

        "output_columns": list(
            df.columns
        ),

        "numeric_features": (
            numeric_features
        ),

        "categorical_features": (
            categorical_features
        ),

        "rare_threshold": int(
            args.rare_threshold
        ),

        "rfecv_step": int(
            args.rfecv_step
        ),

        "rfecv_min_features": int(
            args.rfecv_min_features
        ),

        "metrics": metrics,
    }

    args.output_metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_metadata.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE BUILD COMPLETATO")
    print("=" * 70)

    print(
        f"Dataset: "
        f"{args.output_features}"
    )

    print(
        f"Shape: "
        f"{df.shape}"
    )

    print()
    print(
        "BASELINE"
    )

    print(
        "CV accuracy mean: "
        f"{metrics['baseline_cv_accuracy_mean']:.4f}"
    )

    print(
        "CV accuracy std:  "
        f"{metrics['baseline_cv_accuracy_std']:.4f}"
    )

    print(
        "CV folds: "
        f"{metrics['baseline_cv_accuracy_folds']}"
    )

    print()
    print(
        "RFECV"
    )

    print(
        "Feature iniziali dopo encoding: "
        f"{metrics['baseline_feature_count']}"
    )

    print(
        "Feature selezionate: "
        f"{metrics['rfecv_selected_feature_count']}"
    )

    print(
        "Best RFECV CV accuracy: "
        f"{metrics['rfecv_best_cv_accuracy']:.4f}"
    )

    print()
    print(
        "Metadata:"
        f" {args.output_metadata}"
    )

    print()
    print(
        "Pipeline completata."
    )


if __name__ == "__main__":
    main()