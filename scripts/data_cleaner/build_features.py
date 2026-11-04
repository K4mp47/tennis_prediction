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


# ARGUMENTS-----------------------------------------------------------------
def parse_args() -> argparse.Namespace: #configure input, output e parametric without touching code.
    parser = argparse.ArgumentParser( #create parser to read arguments from terminal
        description=(
            "Build pre-match engineered features from the enriched "
            "Tennis-Data/Sackmann dataset and run RFECV."
        )
    )

    default_root = Path(__file__).resolve().parents[2] #find original path
    #defines input file
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
    #defines where to save new file with enriched data
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
    #defines where to save metadata in json
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
    #minimum number(20) of occurrences to keep a category in a column
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=20,
        help=(
            "Minimum number of observations required to keep a "
            "tournament/location category."
        ),
    )
    #number of feature to delite at every RFECV
    parser.add_argument(
        "--rfecv-step",
        type=int,
        default=1,
    )
    #minimum number of feature to keep at every RFECV
    parser.add_argument(
        "--rfecv-min-features",
        type=int,
        default=8,
    )
    #to avoid random forest and RFECV
    parser.add_argument(
        "--skip-model-analysis",
        action="store_true",
        help=(
            "Build and save the feature dataset without running the optional "
            "Random Forest baseline and RFECV diagnostics. Recommended when "
            "the next step performs its own model selection, such as CatBoost."
        ),
    )

    return parser.parse_args() #return each argument collected

# LOAD----------------------------------------------------------------------
def load_clean_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    df = pd.read_csv(path) #load dataset in pandas dataframe

    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column.")

    required = [ #columns required
        "winner_name",
        "loser_name",
        "surface",
        "round",
    ]

    missing = [c for c in required if c not in df.columns] #Identifies required missing

    if missing: #stop if col are missing
        raise ValueError(
            f"Missing required columns: {missing}"
        )
    #converts dates to datetime objects and marks invalid dates as missing
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.dropna(subset=["date"]).copy() #Removes rows with invalid or missing dates
    # Selects columns for chronological sorting
    sort_cols = [
        c
        for c in [ #una per una col
            "date",
            "source_year",
            "atp",
            "tournament",
            "winner_name",
            "loser_name",
        ]
        if c in df.columns
    ]
    #Sorts matches chronologically and resets row index
    df = (
        df.sort_values(
            sort_cols,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return df

# UTILS--------------------------------------------------------------------
def map_rare_categories(
        series: pd.Series,
        min_count: int,
) -> pd.Series:
    series = series.astype("object") #Converts the column to object type
    #Counts each category appearance
    counts = series.value_counts(
        dropna=False,
    )
    #keeps only those with a certain count
    keep_values = set(
        counts[counts >= min_count].index
    )
    #replaces with "other"
    return (
        series.where(
            series.isin(keep_values),
            "other",
        )
        .fillna("other")
    )


def encode_round_ordinal(
        series: pd.Series,
) -> pd.Series: #encoder with the predefined chronological order
    encoder = OrdinalEncoder(
        categories=[ORDINAL_ROUND_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    #Replaces missing values, entries to strings,re shapes data in two-dimensional format
    values = (
        series
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
        .reshape(-1, 1)
    )
    #tournament rounds in numerical ordinal values, Unknown categories = -1.
    encoded = encoder.fit_transform(values).reshape(-1)

    #Returns encoded values with original index
    return pd.Series(
        encoded,
        index=series.index,
        dtype=float,
    )

# PLAYER HISTORY---------------------------------------------------------------
def build_player_history_features(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes historical player features strictly BEFORE each match.

    Important:
    We never use the current match outcome when calculating a feature
    for that same match.
    """

    out = df.copy() #copy to avoid modifies the original dataset
    #creates an unique ID to each match
    out["match_id_internal"] = np.arange( #np.arange creates an array, out[] creates a col
        len(out), #counts raw
        dtype=np.int64, #big ID
    )
    #extract data to build player history
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
            "is_win": 1.0, #winner
        }
    )

    loser_view = pd.DataFrame(
        {
            "match_id_internal": matches["match_id_internal"],
            "date": matches["date"],
            "player_name": matches["loser_name"],
            "surface": matches["surface"],
            "is_win": 0.0, #loser
        }
    )

    long_df = pd.concat( #combine 2players data in 1 table, Each match -> 2 observation
        [winner_view, loser_view],
        ignore_index=True,
    )
    #player data chronologically
    long_df = long_df.sort_values(
        [
            "player_name",
            "date",
            "match_id_internal",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # GLOBAL HISTORY---------------------------------------------------------
    #Groups all match records by player
    player_group = long_df.groupby(
        "player_name",
        sort=False, #Keeps categories order they appear instead of frequency
    )
    #how many matches each player has played BEFORE the current match
    long_df["matches_before"] = (
        player_group.cumcount().astype(float) #cumcount starts at 0
    )

    # CRITICAL:
    #This avoids the previous global .shift(1) bug
    #total number of wins before the current match
    long_df["wins_before"] = (
            player_group["is_win"]
            .cumsum() #cumsum includes current result, subtract is_win to exclude it
            - long_df["is_win"]
    )
    #player's historical win rate before current match
    long_df["win_rate_before"] = np.where(
        long_df["matches_before"] > 0,
        long_df["wins_before"]
        / long_df["matches_before"],
        np.nan, #Returns NaN when the player has no previous matches
    )

    #RECENT 5 MATCHES-------------------------------------------------
    #count last 5 matches
    long_df["recent5_matches_before"] = (
        long_df.groupby(
            "player_name",
            sort=False,
        )
        .cumcount()
        .clip(upper=5)
        .astype(float) #convert to float
    )

    long_df["recent5_wins_before"] = (
        long_df.groupby(
            "player_name",
            sort=False,
        )["is_win"]
        .transform(
            lambda s: (
                s.shift(1) #excludes current result
                .rolling(
                    window=5, #at most previous five observations
                    min_periods=1, #allows calculations < five matches
                )
                .sum()
            )
        )
        .fillna(0.0) #missing data replace with 0
    )
    #win rate previous five matches, otherwise NaN
    long_df["recent5_win_rate_before"] = np.where(
        long_df["recent5_matches_before"] > 0,
        long_df["recent5_wins_before"]
        / long_df["recent5_matches_before"],
        np.nan,
        )

    #SURFACE HISTORY--------------------------------------------------
    #group match by player and surface
    surface_group = long_df.groupby(
        ["player_name", "surface"], #Hard, Clay, Grass, other
        sort=False,
    )

    long_df["surface_matches_before"] = ( #counts match on that surface of 1 player
        surface_group.cumcount().astype(float)
    )

    long_df["surface_wins_before"] = (#counts wins on that surface of 1 player
            surface_group["is_win"]
            .cumsum()
            - long_df["is_win"] #excludes current match
    )
    #win rate on that surface, or NaN if there aren't previous match
    long_df["surface_win_rate_before"] = np.where(
        long_df["surface_matches_before"] > 0,
        long_df["surface_wins_before"]
        / long_df["surface_matches_before"],
        np.nan,
        )

    # REST-----------------------------------------------------------

    long_df["days_since_last_match"] = (
        long_df.groupby(
            "player_name",
            sort=False,
        )["date"] #select the col with date match
        .diff() #subtracts previous match date from current
        .dt.days #entracts number of days
        .astype(float)
    )

    history_cols = [ #historical features
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

    winner_hist = ( #Selects each match's historical features for the winning player
        long_df[
            [
                "match_id_internal",
                "player_name",
                *history_cols,
            ]
        ]
        .rename(
            columns={ #Renames player_name to winner_name and prefixes feature names with winner_
                "player_name": "winner_name",
                **{
                    c: f"winner_{c}"
                    for c in history_cols
                },
            }
        )
        .drop_duplicates( #Removes duplicate records for the same match and winner.
            subset=[
                "match_id_internal",
                "winner_name",
            ],
            keep="last",
        )
    )

    loser_hist = ( #same for the loser
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
    #winner historical features to original match dataset
    out = out.merge(
        winner_hist,
        on=[ #match raw
            "match_id_internal",
            "winner_name",
        ],
        how="left", 
    )
    #same for the loser
    out = out.merge(
        loser_hist,
        on=[
            "match_id_internal",
            "loser_name",
        ],
        how="left", #left join
    )

    return out #original match dataset enriched


# HEAD TO HEAD--------------------------------------------------------------

def build_head_to_head_features(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    H2H features are calculated strictly before the current match.

    No global shift is used, because that can move information from one
    player pair to another pair.
    """

    out = df.copy() #copy original dataframe per nn modificarlo

    out["pair_key"] = ( #create key
        out.apply(
            lambda row: "||".join( #unifies names of playe, so we've same group
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

    out["winner_is_player1"] = ( #verifica se vincitore del match è 1 nome alfabeticamente
        out.apply(
            lambda row: int(
                str(row["winner_name"])
                <= str(row["loser_name"])
            ),
            axis=1,
        )
    )

    out = out.sort_values( #riordina
        [
            "pair_key",
            "date",
            "match_id_internal",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    pair_group = out.groupby( #raggruppa in base alla copia di giocatori senza invertire ordine alfabitico
        "pair_key",
        sort=False,
    )

    out["h2h_matches_before"] = ( #count previous mstch between the 2 players
        pair_group.cumcount().astype(float)
    )

    #salva risultato del match corrente per il primo giocatore
    current_player1_win = (
        out["winner_is_player1"].astype(float)
    )

    #count wins 1° player before this match, cumulative result minus current result.
    out["pair_player1_wins_before"] = (
            pair_group["winner_is_player1"]
            .cumsum()
            - current_player1_win
    )
    #count wins current winner before this match, cumulative result minus current result.
    out["winner_h2h_wins_before"] = np.where(
        out["winner_is_player1"] == 1,
        out["pair_player1_wins_before"],
        (
                out["h2h_matches_before"]
                - out["pair_player1_wins_before"]
        ),
        )
    #count loose current loser before this match, cumulative result minus current result.
    out["loser_h2h_wins_before"] = (
            out["h2h_matches_before"]
            - out["winner_h2h_wins_before"]
    )
    #% win of current winner before match, otherwise NaN
    out["winner_h2h_win_rate_before"] = np.where(
        out["h2h_matches_before"] > 0,
        out["winner_h2h_wins_before"]
        / out["h2h_matches_before"],
        np.nan,
        )
    #% loose of current loser before match, otherwise NaN
    out["loser_h2h_win_rate_before"] = np.where(
        out["h2h_matches_before"] > 0,
        out["loser_h2h_wins_before"]
        / out["h2h_matches_before"],
        np.nan,
        )
    #final dataframe without temporany col
    return out.drop(
        columns=[
            "pair_key",
            "winner_is_player1",
            "pair_player1_wins_before",
        ]
    )

# MODEL FEATURES-----------------------------------------------------------
def add_model_features(  #more feature
        df: pd.DataFrame,
        rare_threshold: int,
) -> pd.DataFrame:
    out = df.copy()
    #Converts rounds into ordinal numerical values.
    if "round" in out.columns:
        out["round_ordinal"] = encode_round_ordinal(
            out["round"]
        )
    #create FLAGS for missing player rankings and points
    for column in [
        "winner_rank",
        "loser_rank",
        "winner_points",
        "loser_points",
    ]:
        if column in out.columns:
            out[f"{column}_missing"] = (
                out[column].isna().astype(int) #value 1
            ) #otherwise 0

    #calculate player RANK and POINT DIFFERENCES---------------------------------------------------------

    if { #calculates rank difference
        "winner_rank",
        "loser_rank",
    }.issubset(out.columns):
        out["rank_diff"] = (
                out["loser_rank"]
                - out["winner_rank"]
        )

    if { #calculates points difference
        "winner_points",
        "loser_points",
    }.issubset(out.columns):
        out["points_diff"] = (
                out["winner_points"]
                - out["loser_points"]
        )

    #BETTING MARKET------------------------------------------------
    #Converts the average betting odds for winner -> implied probability by taking reciprocal of the odds.
    if "avg_winner" in out.columns:
        out["winner_implied_prob_avg"] = (
                1.0 / out["avg_winner"]
        )

    if "avg_loser" in out.columns: #same for loser
        out["loser_implied_prob_avg"] = (
                1.0 / out["avg_loser"]
        )

    if { #difference, + value = higher probability for winner
        "winner_implied_prob_avg",
        "loser_implied_prob_avg",
    }.issubset(out.columns):
        out["implied_prob_diff_avg"] = (
                out["winner_implied_prob_avg"]
                - out["loser_implied_prob_avg"]
        )

    # PLAYER HISTORY DIFFERENCES--------------------------------------------------
    #new and old feature for winner/loser pair
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

    # Creates one difference feature for each pair, calculate as winner - loser value
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

    # SACKMANN DIFFERENCES------------------------------------------
    #pairs of player characteristics and performance from sackman dataset that specifies source feature and the output difference
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
    #winner-minus-loser difference for every pair
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

    # CATEGORICAL BUCKETS------------------------------------------------------
    if "tournament" in out.columns: #"other" category
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
    ).reset_index(drop=True) #nuovo indice dopo il sort, elimina quello vecchio

    return out #new dataset with new features


# SYMMETRIC DATASET---------------------------------------------------------
#reduces dependence on the arbitrary winner/loser orientation, helps learn comparisons between two players
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

    #Swap player variables
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

    for left, right in swap_pairs: #two columns exchange contents
        if left in a_lose.columns and right in a_lose.columns: #if cols exist
            tmp = a_lose[left].copy() #temporary variable preserves original values
            a_lose[left] = a_lose[right].copy()
            a_lose[right] = tmp

    # Swap player names / Sackmann IDs / hand------------------------------------------------
    generic_swap_pairs = [
        ("winner_name", "loser_name"),
        ("winner_sackmann_id", "loser_sackmann_id"),
        ("winner_hand", "loser_hand"),
        ("winner_ioc", "loser_ioc"),
    ]

    for left, right in generic_swap_pairs: #swap
        if left in a_lose.columns and right in a_lose.columns:
            tmp = a_lose[left].copy()
            a_lose[left] = a_lose[right].copy()
            a_lose[right] = tmp

    # Flip difference features----------------------------------------------------------
    #features calculated as winner - loser, must change sign to preserve meaning
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

    for column in sign_flip_columns:  #swap sign +++ -> ---
        if column in a_lose.columns: #check in table a_lose and in col
            a_lose[column] = -a_lose[column]

    out = pd.concat( #combine in a single DataFrame
        [
            a_win,
            a_lose,
        ],
        ignore_index=True,
    )

    #sorts the observations by date, match ID, and target value, keeps observations for each match together
    out = out.sort_values(
        [
            "date",
            "match_id_internal",
            "player_a_win",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return out


# FEATURE MATRIX-------------------------------------------------------------

def build_feature_matrix(
        df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
    list[str],
]: #return tuple of 4 objects
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

    available_categorical = [  #Keeps categorical features that exist in input DataFrame
        c
        for c in categorical_features
        if c in df.columns
    ]

    available_numeric = [ #same for numeric
        c
        for c in numeric_features
        if c in df.columns
    ]

    if "player_a_win" not in df.columns: #if target column exists
        raise ValueError(
            "Missing target column 'player_a_win'."
        )

    #using all available numerical and categorical features, each row = 1 observation, and each column = 1 model input
    X = df[
        available_numeric
        + available_categorical
        ].copy()

    y = df["player_a_win"].astype(int) #1 = player A won; 0 = player A lost

    return ( #return matrix
        X,
        y,
        available_numeric,
        available_categorical,
    )

# MODEL / RFECV-------------------------------------------------------------
# Trains and evaluates a model using time-based cross-validation, and prepares the workflow for Recursive Feature Elimination
def run_baseline_and_rfecv(
        X: pd.DataFrame,
        y: pd.Series,
        numeric_features: list[str],
        categorical_features: list[str],
        rfecv_step: int,
        rfecv_min_features: int,
) -> dict: #return a dictionary
    # Selects the number of cross-validation splits based on dataset size
    if len(X) < 1000:
        split_count = 3
    elif len(X) < 5000:
        split_count = 4
    else:
        split_count = 5
    #testare l'affidabilità di un modello con dati post training
    cv = TimeSeriesSplit(
        n_splits=split_count
    )
    #separate preprocessing pipelines for numerical and categorical data
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[ #missing numerical values with the median
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        ( #Standardizes numerical features by removing the mean, scaling them to unit variance
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat", #etichetta
                Pipeline( #concatenare
                    steps=[ #serie tuple
                        (
                            "imputer",
                            SimpleImputer( #missing value
                                strategy="most_frequent" #most frequently occurring category
                            ),
                        ),
                        ( #Converts text in binary indicator columns
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", #ignores categories unseen during fitting
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop", #Drops columns that are not explicitly included in the transformers
    )

    baseline_model = RandomForestClassifier(
        n_estimators=250, #decision trees
        random_state=42, #seed
        n_jobs=-1, #parallel processing on other CPU cores
        class_weight="balanced_subsample", #adjusts class weights separately for each tree's sample
        min_samples_leaf=2, #two training samples in each leaf
    )

    baseline_pipeline = Pipeline( #combine
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

    baseline_scores = cross_val_score( #Accuracy evaluation
        baseline_pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    #RFECV --- Recursive Feature Elimination with Cross-Validation ------------------------
    #We transform once here only for feature-selection diagnostics.
    #The baseline CV above remains the primary unbiased estimate.

    print()
    print("Preparing RFECV matrix...")

    prefit = preprocess.fit(X, y) #studia i dati X per capire come migliorarli
    #applica punto precedente, fitted preprocessing steps to transform X into a numerical matrix
    X_prepared = prefit.transform(X)

    feature_names = (
        prefit #Access the fitted preprocessing object
        .get_feature_names_out()
        .tolist()
    )

    selector_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42, #seed
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    max_features = max( #max number feature
        1, #at least 1 feature
        X_prepared.shape[1] - 1, #limit to one fewer than the total number of prepared features
        )

    min_features = min( #minimum number of features RFECV should retain
        rfecv_min_features, #user-defined minimum feature count as the preferred limit
        max_features, #minimum does not exceed the calculated maximum
    )

    print(
        f"Encoded feature count: "
        f"{X_prepared.shape[1]}" #number of columns in the prepared feature matrix
    )

    print(
        f"RFECV minimum features: "
        f"{min_features}"
    )

    rfecv = RFECV(
        estimator=selector_model, #configured Random Forest
        step=rfecv_step, #how many features to remove at each elimination step
        min_features_to_select=min_features, #minimum number of features that must be retained
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2, #Display progress information while RFECV is running
    )

    rfecv.fit( #run rfecv
        X_prepared,
        y,
    )
    #per filtrare tab X e tenere col buone
    support_mask = rfecv.support_ #Boolean vector indicating which features were selected and not

    selected_feature_names = [ #create list
        name
        for name, keep in zip( #Pair each feature name with its corresponding selection indicator
            feature_names,
            support_mask, #Provide the Boolean indicators returned by RFECV
        )
        if keep # Include the feature only if RFECV selected it
    ]

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

    if args.skip_model_analysis:
        print()
        print("MODEL BASELINE / RFECV skipped by request.")
        metrics = {
            "skipped": True,
            "reason": (
                "Feature artifacts built for a downstream model with its own "
                "temporal model-selection process."
            ),
        }
    else:
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

        "model_analysis_skipped": bool(
            args.skip_model_analysis
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

    if metrics.get("skipped"):
        print()
        print("BASELINE / RFECV: skipped")
    else:
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