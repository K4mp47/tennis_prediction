from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RAW_TO_NORMALIZED_COLUMNS = {
    "source_year": "source_year",
    "ATP": "atp",
    "Location": "location",
    "Tournament": "tournament",
    "Date": "date",
    "Series": "series",
    "Court": "court",
    "Surface": "surface",
    "Round": "round",
    "Best of": "best_of",
    "Winner": "winner_name",
    "Loser": "loser_name",
    "WRank": "winner_rank",
    "LRank": "loser_rank",
    "WPts": "winner_points",
    "LPts": "loser_points",
    "Comment": "comment",
    "B365W": "b365_winner",
    "B365L": "b365_loser",
    "PSW": "ps_winner",
    "PSL": "ps_loser",
    "MaxW": "max_winner",
    "MaxL": "max_loser",
    "AvgW": "avg_winner",
    "AvgL": "avg_loser",
    "BFEW": "bfe_winner",
    "BFEL": "bfe_loser",
}

LEAKAGE_COLUMNS = [
    "W1", "L1",
    "W2", "L2",
    "W3", "L3",
    "W4", "L4",
    "W5", "L5",
    "Wsets", "Lsets",
]

FINAL_COLUMNS = [
    "source_year",
    "atp",
    "location",
    "tournament",
    "date",
    "series",
    "court",
    "surface",
    "round",
    "best_of",
    "winner_name",
    "loser_name",
    "winner_rank",
    "loser_rank",
    "winner_points",
    "loser_points",
    "b365_winner",
    "b365_loser",
    "ps_winner",
    "ps_loser",
    "max_winner",
    "max_loser",
    "avg_winner",
    "avg_loser",
    "bfe_winner",
    "bfe_loser",
    "has_any_market_odds",
]


NUMERIC_COLUMNS = [
    "source_year",
    "atp",
    "best_of",
    "winner_rank",
    "loser_rank",
    "winner_points",
    "loser_points",
    "b365_winner",
    "b365_loser",
    "ps_winner",
    "ps_loser",
    "max_winner",
    "max_loser",
    "avg_winner",
    "avg_loser",
    "bfe_winner",
    "bfe_loser",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw Tennis-Data matches before feature engineering."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(Path(__file__).resolve().parents[2] / "data" / "interim" / "tennis_matches_raw.data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(Path(__file__).resolve().parents[2] / "data" / "interim" / "tennis_matches_cleaned.data"),
    )
    return parser.parse_args()


def clean_string(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    text = " ".join(text.split())

    if text == "":
        return None

    return text


def normalize_player_name(value: object) -> str | None:
    text = clean_string(value)

    if text is None:
        return None

    # Keep the Tennis-Data format, but remove repeated whitespace.
    # Do not aggressively rewrite names here: aliases are a later entity-resolution problem.
    return text


def normalize_series(value: object) -> str:
    text = clean_string(value)

    if text is None:
        return "Unknown"

    aliases = {
        "ATP250": "ATP250",
        "ATP500": "ATP500",
        "Masters 1000": "Masters1000",
        "Masters1000": "Masters1000",
        "Grand Slam": "GrandSlam",
        "GrandSlam": "GrandSlam",
        "International": "International",
    }

    return aliases.get(text, text)


def normalize_court(value: object) -> str:
    text = clean_string(value)

    if text is None:
        return "Unknown"

    text = text.title()

    if text in {"Indoor", "Outdoor"}:
        return text

    return "Unknown"


def normalize_surface(value: object) -> str:
    text = clean_string(value)

    if text is None:
        return "Unknown"

    text = text.title()

    aliases = {
        "Hard": "Hard",
        "Clay": "Clay",
        "Grass": "Grass",
        "Carpet": "Carpet",
    }

    return aliases.get(text, "Unknown")


def normalize_round(value: object) -> str:
    text = clean_string(value)

    if text is None:
        return "Unknown"

    aliases = {
        "The Final": "Final",
        "Final": "Final",
        "Semifinals": "Semifinals",
        "Semi-Finals": "Semifinals",
        "Quarterfinals": "Quarterfinals",
        "Quarter-Finals": "Quarterfinals",
        "4th Round": "4th Round",
        "3rd Round": "3rd Round",
        "2nd Round": "2nd Round",
        "1st Round": "1st Round",
        "Round Robin": "Round Robin",
    }

    return aliases.get(text, text)


def load_raw_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    return pd.read_csv(path)


def normalize_table(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    metadata: dict = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": int(len(raw_df)),
        "input_columns": list(raw_df.columns),
    }

    missing_required = [
        col for col in RAW_TO_NORMALIZED_COLUMNS
        if col not in raw_df.columns
    ]

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Remove exact duplicate raw rows before any transformation.
    duplicate_count = int(raw_df.duplicated().sum())
    df = raw_df.drop_duplicates().copy()

    # Filter completed matches only.
    df["Comment"] = df["Comment"].map(clean_string)
    before_completed_filter = len(df)
    df = df[df["Comment"].str.lower() == "completed"].copy()
    removed_not_completed = before_completed_filter - len(df)

    # Rename only columns useful before feature engineering.
    df = df.rename(columns=RAW_TO_NORMALIZED_COLUMNS)

    # Normalize dates.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    invalid_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    # Normalize strings.
    for col in ["location", "tournament"]:
        df[col] = df[col].map(clean_string).fillna("Unknown")

    df["series"] = df["series"].map(normalize_series)
    df["court"] = df["court"].map(normalize_court)
    df["surface"] = df["surface"].map(normalize_surface)
    df["round"] = df["round"].map(normalize_round)
    df["winner_name"] = df["winner_name"].map(normalize_player_name)
    df["loser_name"] = df["loser_name"].map(normalize_player_name)

    missing_players = int(
        df["winner_name"].isna().sum() + df["loser_name"].isna().sum()
    )
    df = df.dropna(subset=["winner_name", "loser_name"])

    # Numeric coercion.
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Enforce integer-like columns as nullable integers.
    for col in ["source_year", "atp", "best_of"]:
        df[col] = df[col].round().astype("Int64")

    # Basic validity filters.
    before_validity = len(df)

    df = df[df["winner_name"] != df["loser_name"]].copy()

    # Ranks and points can be missing in raw data, but impossible values should become missing.
    for col in ["winner_rank", "loser_rank"]:
        df.loc[df[col] <= 0, col] = np.nan

    for col in ["winner_points", "loser_points"]:
        df.loc[df[col] < 0, col] = np.nan

    # Decimal odds should be > 1.0. Invalid odds become missing.
    odds_columns = [
        "b365_winner",
        "b365_loser",
        "ps_winner",
        "ps_loser",
        "max_winner",
        "max_loser",
        "avg_winner",
        "avg_loser",
        "bfe_winner",
        "bfe_loser",
    ]

    for col in odds_columns:
        df.loc[df[col] <= 1.0, col] = np.nan

    df["has_any_market_odds"] = df[odds_columns].notna().any(axis=1).astype(int)

    removed_invalid = before_validity - len(df)

    # Keep only pre-feature-engineering normalized columns.
    df = df[FINAL_COLUMNS].copy()

    # Stable ordering.
    df = df.sort_values(
        ["date", "source_year", "atp", "tournament", "winner_name", "loser_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Store dates in ISO format for .data CSV.
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    metadata.update(
        {
            "output_rows": int(len(df)),
            "output_columns": list(df.columns),
            "removed_exact_duplicates": duplicate_count,
            "removed_not_completed": int(removed_not_completed),
            "removed_invalid_dates": invalid_dates,
            "removed_missing_players": missing_players,
            "removed_invalid_rows": int(removed_invalid),
            "leakage_columns_removed": LEAKAGE_COLUMNS + ["Comment"],
            "normalization_rules": {
                "date": "parsed to ISO yyyy-mm-dd",
                "strings": "trimmed and whitespace-collapsed",
                "court": "normalized to Indoor/Outdoor/Unknown",
                "surface": "normalized to Hard/Clay/Grass/Carpet/Unknown",
                "round": "normalized through explicit aliases",
                "numeric": "coerced with invalid values set to missing",
                "odds": "decimal odds <= 1.0 treated as missing",
                "retired_matches": "removed by keeping only Comment == Completed",
            },
            "next_phase": [
                "orient Winner/Loser into neutral Player A/Player B",
                "create binary target player_a_win",
                "build ranking/points differences",
                "build historical form features",
                "build Elo features",
                "build head-to-head features",
                "create model pipeline for imputation, scaling, and categorical encoding",
            ],
        }
    )

    return df, metadata


def main() -> None:
    args = parse_args()

    raw_df = load_raw_table(args.input)
    normalized_df, metadata = normalize_table(raw_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    normalized_df.to_csv(args.output, index=False)

    print(f"Created normalized data: {args.output}")
    print(f"Rows: {metadata['input_rows']:,} -> {metadata['output_rows']:,}")
    print(f"Removed duplicates: {metadata['removed_exact_duplicates']:,}")
    print(f"Removed not completed: {metadata['removed_not_completed']:,}")
    print(f"Removed invalid dates: {metadata['removed_invalid_dates']:,}")


if __name__ == "__main__":
    main()
