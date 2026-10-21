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
#colonne con esiti partite -> escluse
LEAKAGE_COLUMNS = [
    "W1", "L1",
    "W2", "L2",
    "W3", "L3",
    "W4", "L4",
    "W5", "L5",
    "Wsets", "Lsets",
]
#nel dataset finale
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

#col contengono numeri
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

#Sceglie file da leggere e dove salvi dati puliti
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser( #legge comandi da terminale
        description="Normalize raw Tennis-Data matches before feature engineering."
    )
    parser.add_argument(#definisce percorso file di partenza
        "--input",
        type=Path,
        default=Path(Path(__file__).resolve().parents[2] / "data" / "interim" / "tennis_matches_raw.data"), #specifica percorso
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(Path(__file__).resolve().parents[2] / "data" / "interim" / "tennis_matches_cleaned.data"),
    )
    return parser.parse_args() #legge argomenti passati da terminale, restituisce oggetto con informazioni

#elimina spazi inutili e uniforma i valori testuali
def clean_string(value: object) -> str | None: #può ricevere valori di tipi diversi
    if pd.isna(value): #se valore mancante
        return None

    text = str(value).strip() #str(value) da valore a testo, strip() elimina spazi inizio e fine.
    text = " ".join(text.split()) #split() divide testo dove ci sono spazi, join ricompone lasciando 1 spazio tra ogni parola

    if text == "": #se testo è vuoto
        return None

    return text

#Pulisce nome giocatori
def normalize_player_name(value: object) -> str | None:
    text = clean_string(value) #passa nome a clean_string

    if text is None: #se testo vuoto
        return None

    #Keep the Tennis-Data format, but remove repeated whitespace.
    #Don't aggressively rewrite names here: aliases are a later entity-resolution problem.
    return text

#Uniforma la categoria del torneo
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

    return aliases.get(text, text) #cerca nel dizionario aliases, se trova da nuova forma standard; altrimenti originale.

#Uniforma il tipo di campo, distingue indoor da outdoor
def normalize_court(value: object) -> str:
    text = clean_string(value)

    if text is None:
        return "Unknown"

    text = text.title() #1° lettera maiuscolo ogni parola

    if text in {"Indoor", "Outdoor"}:
        return text

    return "Unknown"

#superficie campo a quattro categorie
def normalize_surface(value: object) -> str:
    text = clean_string(value)

    if text is None:
        return "Unknown"

    text = text.title() #1° letter uppercase

    aliases = {
        "Hard": "Hard",
        "Clay": "Clay",
        "Grass": "Grass",
        "Carpet": "Carpet",
    }

    return aliases.get(text, "Unknown") #return one of the four or unknown

#standardize tournment turn
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

    return aliases.get(text, text) #search in aliases dictionary, if new standard form; otherwise original one

#load raw dataset
def load_raw_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    return pd.read_csv(path) #read csv and return dataframe

#main cleaning function
def normalize_table(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]: #from raw dataset -> clean dataset + metadata dictionary with transformation done
    metadata: dict = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), #date + hour -> string
        "input_rows": int(len(raw_df)), #count raw
        "input_columns": list(raw_df.columns), #list column names
    }

    missing_required = [ #check if there are all the col
        col for col in RAW_TO_NORMALIZED_COLUMNS
        if col not in raw_df.columns
    ]

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Remove exact duplicate raw rows before any transformation.
    duplicate_count = int(raw_df.duplicated().sum()) #count duplicates
    df = raw_df.drop_duplicates().copy() #eliminate duplicates and create new copy of the tab

    # Filter completed matches only.
    df["Comment"] = df["Comment"].map(clean_string) #clean col "comment"
    before_completed_filter = len(df)
    df = df[df["Comment"].str.lower() == "completed"].copy() #keep raw where value is completed, not considering upper and lowercase
    removed_not_completed = before_completed_filter - len(df)

    # Rename only columns useful before feature engineering.
    df = df.rename(columns=RAW_TO_NORMALIZED_COLUMNS)

    # Normalize dates.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    invalid_dates = int(df["date"].isna().sum()) #count na
    df = df.dropna(subset=["date"]) #not valid date -> NaT -> deleted

    # Normalize strings. value mancanti di location and tournament -> Unknown
    for col in ["location", "tournament"]:
        df[col] = df[col].map(clean_string).fillna("Unknown")

    df["series"] = df["series"].map(normalize_series)
    df["court"] = df["court"].map(normalize_court)
    df["surface"] = df["surface"].map(normalize_surface)
    df["round"] = df["round"].map(normalize_round)
    df["winner_name"] = df["winner_name"].map(normalize_player_name)
    df["loser_name"] = df["loser_name"].map(normalize_player_name)

    missing_players = int( #delete match without the names of winner/loser
        df["winner_name"].isna().sum() + df["loser_name"].isna().sum()
    )
    df = df.dropna(subset=["winner_name", "loser_name"])

    # Numeric conversion o NaN
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce") #coerce transform to NaN unparseable values

    # Enforce integer-like columns as nullable integers.
    for col in ["source_year", "atp", "best_of"]:
        df[col] = df[col].round().astype("Int64") #round and int64 works for both int and NaN

    # Basic validity filters.
    before_validity = len(df)
    #delete raw where winner and loser have same name
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
    #1 se è disponibile almeno una quota, altrimenti 0
    df["has_any_market_odds"] = df[odds_columns].notna().any(axis=1).astype(int)

    removed_invalid = before_validity - len(df) #number of raw deleted by validity check

    # Keep only pre-feature-engineering normalized columns.
    df = df[FINAL_COLUMNS].copy()

    # Stable ordering.
    df = df.sort_values(
        ["date", "source_year", "atp", "tournament", "winner_name", "loser_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Store dates in ISO format for .data CSV.
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    #register how many raw are "still standing"
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
    #Legge gli argomenti, carica i dati, li normalizza
    args = parse_args()
    raw_df = load_raw_table(args.input)
    normalized_df, metadata = normalize_table(raw_df)
    #Crea la cartella di destinazione se non esiste
    args.output.parent.mkdir(parents=True, exist_ok=True)
    #Scrive il dataset pulito su disco
    normalized_df.to_csv(args.output, index=False)

    print(f"Created normalized data: {args.output}")
    print(f"Rows: {metadata['input_rows']:,} -> {metadata['output_rows']:,}")
    print(f"Removed duplicates: {metadata['removed_exact_duplicates']:,}")
    print(f"Removed not completed: {metadata['removed_not_completed']:,}")
    print(f"Removed invalid dates: {metadata['removed_invalid_dates']:,}")


if __name__ == "__main__":
    main()
