"""
enrich_with_sackmann.py

Arricchisce il dataset Tennis-Data già pulito con anagrafica proveniente dalla
repository Jeff Sackmann / tennis_atp.

Input:
    data/interim/tennis_matches_cleaned.data

Dati esterni:
    external/tennis_atp/atp_players.csv

Output:
    data/interim/tennis_matches_enriched.data

FEATURES AGGIUNTE
-----------------

Anagrafica:
    winner_height
    loser_height
    winner_hand
    loser_hand
    winner_age_years
    loser_age_years

Le statistiche di servizio basate sulla sola data di inizio torneo sono escluse
nel main: non dimostrano la disponibilità prima della partita. Gli helper storici
restano disponibili per analisi separate, ma non alimentano il modello corrente.

Differenze: height_diff e age_diff.
La biografia deriva da uno snapshot, non da un archivio storico completo.

NAME MATCHING
-------------

Tennis-Data:
    "Nadal R."

Sackmann:
    "Rafael Nadal"

Prima si prova:
    cognome + iniziale

Poi:
    normalizzazione del nome

Poi, solo se sicuro:
    fuzzy matching sul cognome.

I match fuzzy non vengono accettati se il risultato è ambiguo.

Il report:
    data/interim/sackmann_name_matching_review.csv

contiene tutte le decisioni di matching.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz import process as rf_process


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich the cleaned Tennis-Data dataset with "
            "Jeff Sackmann tennis_atp data."
        )
    )

    script_root = Path(__file__).resolve().parents[2]

    parser.add_argument(
        "--cleaned-input",
        type=Path,
        default=script_root / "data" / "interim" / "tennis_matches_cleaned.data",
        help="Dataset prodotto da normalize_data.py.",
    )

    parser.add_argument(
        "--sackmann-dir",
        type=Path,
        required=True,
        help=(
            "Cartella contenente atp_players.csv per l’arricchimento anagrafico."
        ),
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility option; biography enrichment does not load matches."
        ),
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility option; biography enrichment does not load matches."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
                script_root
                / "data"
                / "interim"
                / "tennis_matches_enriched.data"
        ),
    )

    parser.add_argument(
        "--unmatched-report",
        type=Path,
        default=(
                script_root
                / "data"
                / "interim"
                / "sackmann_name_matching_review.csv"
        ),
        help="Report dettagliato del name matching.",
    )

    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=92,
        help=(
            "Score minimo fuzzy sul cognome. "
            "Default 92."
        ),
    )

    parser.add_argument(
        "--fuzzy-margin",
        type=int,
        default=5,
        help=(
            "Margine minimo tra primo e secondo candidato fuzzy. "
            "Default 5."
        ),
    )

    return parser.parse_args()


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def normalize_text(value: Any) -> str:
    """
    Normalizza testo per confronti robusti.

    Esempio:
        "García-López" -> "garcia lopez"
    """

    if value is None or pd.isna(value):
        return ""

    text = str(value)

    text = unicodedata.normalize(
        "NFKD",
        text,
    ).encode(
        "ascii",
        "ignore",
    ).decode(
        "ascii",
    )

    text = text.lower()

    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace(".", " ")
    text = text.replace("'", "")
    text = text.replace("’", "")

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return " ".join(text.split())

# The function converts a pandas Series to numeric values, 
# coercing errors to NaN. The function is as follows:
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ============================================================================
# NAME PARSING
# ============================================================================

def tennis_data_name_parts(name: object) -> tuple[str, str]:
    """
    Tennis-Data:
        "Nadal R." -> ("nadal", "r")

    Gestisce anche:
        "Van De Zandschulp B."
        "Roger-Vasselin E."
    """

    text = normalize_text(name)

    if not text:
        return "", ""

    parts = text.split()

    if len(parts) == 1:
        return parts[0], ""

    initial = parts[-1][0]
    surname = " ".join(parts[:-1])

    return surname, initial


def sackmann_name_parts(full_name: object) -> tuple[str, str]:
    """
    Sackmann:
        "Rafael Nadal" -> ("nadal", "r")
    """

    text = normalize_text(full_name)

    if not text:
        return "", ""

    parts = text.split()

    if len(parts) == 1:
        return parts[0], parts[0][0]

    first_name = parts[0]
    surname = " ".join(parts[1:])

    return surname, first_name[0]


# ============================================================================
# LOAD PLAYERS
# ============================================================================

def load_sackmann_players(sackmann_dir: Path) -> pd.DataFrame:
    path = sackmann_dir / "atp_players.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"\nManca il file:\n{path}\n"
            "Controlla --sackmann-dir."
        )

    print(f"Carico giocatori Sackmann: {path}")

    df = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
    )

    # Compatibilità con eventuali versioni differenti.
    rename_map = {}

    if "name_first" in df.columns:
        rename_map["name_first"] = "first_name"

    if "name_last" in df.columns:
        rename_map["name_last"] = "last_name"

    if "dob" in df.columns:
        rename_map["dob"] = "birth_date"

    if "ht" in df.columns:
        rename_map["ht"] = "height"

    df = df.rename(columns=rename_map)

    required = {
        "player_id",
        "first_name",
        "last_name",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "atp_players.csv: colonne mancanti: "
            f"{sorted(missing)}"
        )

    df["player_id"] = df["player_id"].astype(str).str.strip()

    df["full_name"] = (
            df["first_name"].fillna("")
            + " "
            + df["last_name"].fillna("")
    ).str.strip()

    if "birth_date" in df.columns:
        df["birth_date"] = pd.to_datetime(
            df["birth_date"],
            format="%Y%m%d",
            errors="coerce",
        )
    else:
        df["birth_date"] = pd.NaT

    if "height" in df.columns:
        df["height"] = safe_numeric(df["height"])
    else:
        df["height"] = np.nan

    if "hand" not in df.columns:
        df["hand"] = np.nan

    df["hand"] = df["hand"].replace(
        {
            "R": "R",
            "L": "L",
            "U": np.nan,
        }
    )

    df = df[
        [
            "player_id",
            "full_name",
            "hand",
            "height",
            "birth_date",
        ]
    ].copy()

    df = df.drop_duplicates(
        subset=["player_id"],
        keep="first",
    )

    print(f"Giocatori caricati: {len(df):,}")

    return df


# ============================================================================
# VALIDATE CSV
# ============================================================================

def validate_csv_file(path: Path) -> bool:
    """
    Controlla rapidamente che il file non sia un HTML/404 salvato
    con estensione .csv.
    """

    if not path.exists():
        return False

    try:
        with path.open(
                "r",
                encoding="utf-8",
                errors="replace",
        ) as f:
            first_line = f.readline().strip().lower()

        if (
                first_line.startswith("<!doctype html")
                or first_line.startswith("<html")
                or "404: not found" in first_line
        ):
            print(
                f"ATTENZIONE: file non valido ignorato: {path}"
            )
            return False

        return True

    except OSError:
        return False


# ============================================================================
# LOAD MATCHES
# ============================================================================

def load_sackmann_matches(
        sackmann_dir: Path,
        start_year: int,
        end_year: int,
) -> pd.DataFrame:

    frames: list[pd.DataFrame] = []
    loaded_years: list[int] = []
    missing_years: list[int] = []

    for year in range(start_year, end_year + 1):

        path = sackmann_dir / f"atp_matches_{year}.csv"

        if not path.exists():
            missing_years.append(year)
            continue

        if not validate_csv_file(path):
            continue

        try:
            df = pd.read_csv(
                path,
                low_memory=False,
            )
        except Exception as exc:
            print(
                f"ATTENZIONE: impossibile leggere {path}: {exc}"
            )
            continue

        required = {
            "tourney_date",
            "winner_id",
            "loser_id",
        }

        missing = required - set(df.columns)

        if missing:
            print(
                f"ATTENZIONE: {path.name} ignorato; "
                f"colonne mancanti: {sorted(missing)}"
            )
            continue

        frames.append(df)
        loaded_years.append(year)

    if not frames:
        raise FileNotFoundError(
            "\nNessun atp_matches_YYYY.csv valido trovato.\n"
            f"Directory: {sackmann_dir}\n"
            f"Range: {start_year}-{end_year}"
        )

    matches = pd.concat(
        frames,
        ignore_index=True,
    )

    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"],
        format="%Y%m%d",
        errors="coerce",
    )

    matches["winner_id"] = (
        matches["winner_id"]
        .astype("string")
        .str.strip()
    )

    matches["loser_id"] = (
        matches["loser_id"]
        .astype("string")
        .str.strip()
    )

    matches = matches.dropna(
        subset=[
            "tourney_date",
            "winner_id",
            "loser_id",
        ]
    ).copy()

    matches = matches.sort_values(
        [
            "tourney_date",
            "tourney_id",
            "match_num",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ID interno stabile.
    matches["match_id"] = np.arange(
        len(matches),
        dtype=np.int64,
    )

    print(
        f"Anni Sackmann caricati: "
        f"{loaded_years[0]}-{loaded_years[-1]}"
    )

    print(
        f"Match Sackmann caricati: "
        f"{len(matches):,}"
    )

    if missing_years:
        print(
            f"Anni mancanti/non presenti: "
            f"{len(missing_years)}"
        )

    return matches


# ============================================================================
# NAME MATCHING
# ============================================================================

def build_name_map(
        tennis_data_names: pd.Series,
        sackmann_players: pd.DataFrame,
        fuzzy_threshold: int,
        fuzzy_margin: int,
) -> tuple[dict[str, str], pd.DataFrame]:

    # Copy the player table so temporary matching columns do not alter the input.
    players = sackmann_players.copy()

    # Parse each full name into (normalized surname, first-name initial).
    # map() produces tuples; apply(pd.Series) expands them into two columns.
    players[
        ["surname_norm", "initial"]
    ] = players["full_name"].map(
        sackmann_name_parts
    ).apply(
        pd.Series
    )

    # Group player records by normalized surname for exact lookups.
    by_surname: dict[str, list[dict]] = {}

    # Visit each player row; the unused DataFrame index is assigned to _.
    for _, row in players.iterrows():

        # Read the normalized surname used as the lookup key.
        surname = row["surname_norm"]

        # Skip Sackmann players with no usable surname.
        if not surname:
            # Advance to the next Sackmann player.
            continue

        # Create an empty list for a new surname, then append this player as a dict.
        by_surname.setdefault(
            surname,
            [],
        ).append(
            row.to_dict()
        )

    # Compare fuzzy queries against distinct surnames, not individual players.
    surname_list = list(by_surname.keys())

    # Map accepted original Tennis-Data names to Sackmann player IDs.
    name_map: dict[str, str] = {}
    # Collect accepted and rejected decisions for the review table.
    review_rows: list[dict] = []

    # Drop missing names, stringify, reject whitespace-only values, deduplicate,
    # and sort. strip() checks emptiness but does not change stored spellings.
    unique_names = sorted(
        {
            str(x)
            for x in tennis_data_names.dropna()
            if str(x).strip()
        }
    )

    # Resolve each distinct Tennis-Data name once.
    for td_name in unique_names:

        # Parse an abbreviated name: "Nadal R." becomes ("nadal", "r").
        surname, initial = tennis_data_name_parts(td_name)

        # Shared report fields; **row_base copies these into decision dictionaries.
        row_base = {
            "tennis_data_name": td_name,
            "tennis_data_surname": surname,
            "tennis_data_initial": initial,
        }

        # Report an unparseable Tennis-Data name without assigning a player ID.
        if not surname:
            # Append the decision and its scores to the review report.
            review_rows.append(
                {
                    **row_base,
                    "sackmann_name": None,
                    "player_id": None,
                    "match_method": "invalid_name",
                    "score": None,
                    "second_score": None,
                }
            )
            # This name is handled; advance to the next Tennis-Data name.
            continue

        # ------------------------------------------------------------
        # 1. Exact surname + initial
        # ------------------------------------------------------------

        # Fetch players with this exact normalized surname, or an empty list.
        candidates = by_surname.get(
            surname,
            [],
        )

        # Require the initial to match when supplied; otherwise keep all candidates.
        exact = [
            c
            for c in candidates
            if not initial
               or c["initial"] == initial
        ]

        # Accept only a unique candidate, including surname-only input.
        if len(exact) == 1:

            # Extract the sole player record.
            chosen = exact[0]

            # Save the association under the original Tennis-Data spelling.
            name_map[td_name] = chosen["player_id"]

            # Append the decision and its scores to the review report.
            review_rows.append(
                {
                    **row_base,
                    "sackmann_name": chosen["full_name"],
                    "player_id": chosen["player_id"],
                    "match_method": "exact_surname_initial",
                    "score": 100,
                    "second_score": None,
                }
            )

            # This name is handled; advance to the next Tennis-Data name.
            continue

        # ------------------------------------------------------------
        # 2. Exact surname, ma iniziale ambigua
        # ------------------------------------------------------------

        # Refuse multiple exact candidates rather than guessing an identity.
        if len(exact) > 1:

            # Append the decision and its scores to the review report.
            review_rows.append(
                {
                    **row_base,
                    "sackmann_name": None,
                    "player_id": None,
                    "match_method": "ambiguous_surname_initial",
                    "score": 100,
                    "second_score": None,
                }
            )

            # This name is handled; advance to the next Tennis-Data name.
            continue

        # ------------------------------------------------------------
        # 3. Exact surname senza iniziale
        # ------------------------------------------------------------

        # This fallback is currently redundant: a unique surname without an
        # initial is already accepted by the len(exact) == 1 branch above.
        if len(candidates) == 1:

            # Select the only player with this exact surname.
            chosen = candidates[0]

            # Se Tennis-Data non ha iniziale, un singolo cognome
            # è sufficientemente sicuro.
            if not initial:

                # Store the accepted association using the original input spelling.
                name_map[td_name] = chosen["player_id"]

                # Record the surname-only decision with a fixed exact-match score.
                review_rows.append(
                    {
                        **row_base,
                        "sackmann_name": chosen["full_name"],
                        "player_id": chosen["player_id"],
                        "match_method": "exact_surname_only",
                        "score": 100,
                        "second_score": None,
                    }
                )

                # Matching succeeded; process the next input name.
                continue

        # ------------------------------------------------------------
        # 4. Fuzzy surname
        # ------------------------------------------------------------

        # Rank up to three surnames by character similarity (0 to 100).
        # Results contain (surname, score, index in surname_list).
        fuzzy_results = rf_process.extract(
            surname,
            surname_list,
            scorer=rf_fuzz.ratio,
            limit=3,
        )

        # A nonempty result list provides at least one surname to consider.
        if fuzzy_results:

            # Extract the top surname and score; discard its list index.
            best_surname, best_score, _ = fuzzy_results[0]

            # Keep the runner-up score to detect close surname matches.
            second_score = (
                fuzzy_results[1][1]
                if len(fuzzy_results) > 1
                else None
            )

            # Consider only players under the best surname, filtering by initial.
            # Lower-ranked surnames are not tried if this yields no player.
            candidates = [
                c
                for c in by_surname.get(
                    best_surname,
                    [],
                )
                if not initial
                    or c["initial"] == initial
            ]

            # Require a sufficient lead over the runner-up, if one exists.
            margin_ok = (
                    second_score is None
                    or best_score - second_score >= fuzzy_margin
            )

            # Accept only a strong surname score, a sufficient lead, and one player.
            if (
                    best_score >= fuzzy_threshold
                    and margin_ok
                    and len(candidates) == 1
            ):

                # Extract the unique player that passed all fuzzy checks.
                chosen = candidates[0]

                # Store the accepted association using the original input spelling.
                name_map[td_name] = chosen["player_id"]

                # Record the accepted match and the two surname similarity scores.
                review_rows.append(
                    {
                        **row_base,
                        "sackmann_name": chosen["full_name"],
                        "player_id": chosen["player_id"],
                        "match_method": "fuzzy_surname",
                        "score": best_score,
                        "second_score": second_score,
                    }
                )

                # Matching succeeded; process the next input name.
                continue

            # Append the decision and its scores to the review report.
            review_rows.append(
                {
                    **row_base,
                    "sackmann_name": None,
                    "player_id": None,
                    "match_method": "fuzzy_ambiguous_or_low_score",
                    "score": best_score,
                    "second_score": second_score,
                }
            )

            # This name is handled; advance to the next Tennis-Data name.
            continue

        # ------------------------------------------------------------
        # 5. Unmatched
        # ------------------------------------------------------------

        # No surname suggestions were available; report the name as unmatched.
        review_rows.append(
            {
                **row_base,
                "sackmann_name": None,
                "player_id": None,
                "match_method": "unmatched",
                "score": None,
                "second_score": None,
            }
        )

    # Convert the decision dictionaries into a table, one row per processed name.
    review_df = pd.DataFrame(
        review_rows
    )

    # Return accepted mappings separately from the full decision report.
    return name_map, review_df


# ============================================================================
# PRE-MATCH SERVICE FEATURES
# ============================================================================

def build_pre_match_serve_features(
        matches: pd.DataFrame,
) -> pd.DataFrame:

    """
    Costruisce feature cumulative per giocatore.

    IMPORTANTE:
    ogni riga contiene esclusivamente lo storico precedente
    al match rappresentato da match_id.

    Per i match nella stessa data, le statistiche non vengono
    propagate da un match all'altro della stessa giornata.
    """

    matches = matches.copy()

    matches = matches.sort_values(
        [
            "tourney_date",
            "match_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Normalizzazione numerica
    # ------------------------------------------------------------

    numeric_columns = [
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced",
    ]

    for col in numeric_columns:

        if col not in matches.columns:
            matches[col] = np.nan

        matches[col] = pd.to_numeric(
            matches[col],
            errors="coerce",
        )

    # ------------------------------------------------------------
    # Long format
    # ------------------------------------------------------------

    winner = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "tourney_date": matches["tourney_date"],
            "player_id": matches["winner_id"],
            "ace": matches["w_ace"],
            "df": matches["w_df"],
            "svpt": matches["w_svpt"],
            "first_in": matches["w_1stIn"],
            "first_won": matches["w_1stWon"],
            "second_won": matches["w_2ndWon"],
            "sv_games": matches["w_SvGms"],
            "bp_saved": matches["w_bpSaved"],
            "bp_faced": matches["w_bpFaced"],
        }
    )

    loser = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "tourney_date": matches["tourney_date"],
            "player_id": matches["loser_id"],
            "ace": matches["l_ace"],
            "df": matches["l_df"],
            "svpt": matches["l_svpt"],
            "first_in": matches["l_1stIn"],
            "first_won": matches["l_1stWon"],
            "second_won": matches["l_2ndWon"],
            "sv_games": matches["l_SvGms"],
            "bp_saved": matches["l_bpSaved"],
            "bp_faced": matches["l_bpFaced"],
        }
    )

    long_df = pd.concat(
        [
            winner,
            loser,
        ],
        ignore_index=True,
    )

    long_df = long_df.sort_values(
        [
            "player_id",
            "tourney_date",
            "match_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Cumulative history
    # ------------------------------------------------------------

    history_columns = [
        "ace",
        "df",
        "svpt",
        "first_in",
        "first_won",
        "second_won",
        "sv_games",
        "bp_saved",
        "bp_faced",
    ]

    grouped = long_df.groupby(
        "player_id",
        sort=False,
    )

    for col in history_columns:

        # shift BEFORE the current match:
        # cumulative value from previous observations only.
        long_df[f"{col}_cum_before"] = (
            grouped[col]
            .cumsum()
            .groupby(long_df["player_id"])
            .shift(1)
        )

    # ------------------------------------------------------------
    # Denominators
    # ------------------------------------------------------------

    svpt_before = (
        long_df["svpt_cum_before"]
        .replace(0, np.nan)
    )

    first_in_before = (
        long_df["first_in_cum_before"]
        .replace(0, np.nan)
    )

    second_balls_before = (
            svpt_before
            - long_df["first_in_cum_before"]
    ).replace(
        0,
        np.nan,
    )

    bp_faced_before = (
        long_df["bp_faced_cum_before"]
        .replace(0, np.nan)
    )

    sv_games_before = (
        long_df["sv_games_cum_before"]
        .replace(0, np.nan)
    )

    # ------------------------------------------------------------
    # Rates
    # ------------------------------------------------------------

    long_df["ace_rate_before"] = (
            long_df["ace_cum_before"]
            / svpt_before
    )

    long_df["df_rate_before"] = (
            long_df["df_cum_before"]
            / svpt_before
    )

    long_df["first_in_rate_before"] = (
            long_df["first_in_cum_before"]
            / svpt_before
    )

    long_df["first_won_rate_before"] = (
            long_df["first_won_cum_before"]
            / first_in_before
    )

    long_df["second_won_rate_before"] = (
            long_df["second_won_cum_before"]
            / second_balls_before
    )

    long_df["bp_saved_rate_before"] = (
            long_df["bp_saved_cum_before"]
            / bp_faced_before
    )

    long_df["bp_faced_per_sv_game_before"] = (
            long_df["bp_faced_cum_before"]
            / sv_games_before
    )

    feature_cols = [
        "ace_rate_before",
        "df_rate_before",
        "first_in_rate_before",
        "first_won_rate_before",
        "second_won_rate_before",
        "bp_saved_rate_before",
        "bp_faced_per_sv_game_before",
    ]

    result = long_df[
        [
            "match_id",
            "tourney_date",
            "player_id",
            *feature_cols,
        ]
    ].copy()

    return result


# ============================================================================
# PRE-MATCH LOOKUP
# ============================================================================

def attach_latest_pre_match_features(
        cleaned: pd.DataFrame,
        player_ids: pd.Series,
        pre_match_features: pd.DataFrame,
        prefix: str,
) -> pd.DataFrame:

    """
    Per ogni riga di cleaned trova l'ultima osservazione storica
    del player_id con data STRICTLY precedente alla data del match.

    Usa merge_asof invece di una scansione completa per ogni match.
    """

    feature_cols = [
        "ace_rate_before",
        "df_rate_before",
        "first_in_rate_before",
        "first_won_rate_before",
        "second_won_rate_before",
        "bp_saved_rate_before",
        "bp_faced_per_sv_game_before",
    ]

    base = cleaned[
        [
            "_cleaned_row_id",
            "date",
        ]
    ].copy()

    base["player_id"] = player_ids.values

    base = base.dropna(
        subset=["player_id", "date"]
    ).copy()

    base["player_id"] = (
        base["player_id"]
        .astype(str)
    )

    history = pre_match_features[
        [
            "player_id",
            "tourney_date",
            *feature_cols,
        ]
    ].copy()

    history = history.dropna(
        subset=[
            "player_id",
            "tourney_date",
        ]
    )

    history["player_id"] = (
        history["player_id"]
        .astype(str)
    )

    # merge_asof richiede ordinamento globale per la chiave temporale.
    base = base.sort_values(
        [
            "date",
            "player_id",
        ],
        kind="mergesort",
    )

    history = history.sort_values(
        [
            "tourney_date",
            "player_id",
        ],
        kind="mergesort",
    )

    # "backward" + allow_exact_matches=False:
    # usa solo date < match date.
    merged = pd.merge_asof(
        base,
        history,
        left_on="date",
        right_on="tourney_date",
        by="player_id",
        direction="backward",
        allow_exact_matches=False,
    )

    merged = merged.sort_values(
        "_cleaned_row_id"
    )

    renamed = {
        col: f"{prefix}_{col}"
        for col in feature_cols
    }

    merged = merged.rename(
        columns=renamed
    )

    return merged[
        [
            "_cleaned_row_id",
            *renamed.values(),
        ]
    ]


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    args = parse_args()

    print("=" * 78)
    print("ENRICH WITH SACKMANN")
    print("=" * 78)

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    if not args.cleaned_input.exists():
        raise FileNotFoundError(
            f"Input non trovato: {args.cleaned_input}"
        )

    cleaned = pd.read_csv(
        args.cleaned_input,
        low_memory=False,
    )

    required_cleaned = {
        "date",
        "source_year",
        "winner_name",
        "loser_name",
    }

    missing_cleaned = (
            required_cleaned
            - set(cleaned.columns)
    )

    if missing_cleaned:
        raise ValueError(
            "Dataset cleaned: colonne mancanti: "
            f"{sorted(missing_cleaned)}"
        )

    cleaned["date"] = pd.to_datetime(
        cleaned["date"],
        errors="coerce",
    )

    cleaned = cleaned.reset_index(
        drop=True
    )

    cleaned["_cleaned_row_id"] = np.arange(
        len(cleaned),
        dtype=np.int64,
    )

    if cleaned["date"].isna().any():
        raise ValueError("Il dataset pulito contiene date mancanti o non valide.")
    print("Sackmann: solo anagrafica; statistiche di servizio con date non verificate escluse.")

    # ------------------------------------------------------------
    # Load Sackmann
    # ------------------------------------------------------------

    players = load_sackmann_players(
        args.sackmann_dir
    )

    # ------------------------------------------------------------
    # Name matching
    # ------------------------------------------------------------

    print()
    print("Costruisco la mappa dei nomi...")

    all_names = pd.concat(
        [
            cleaned["winner_name"],
            cleaned["loser_name"],
        ],
        ignore_index=True,
    )

    name_map, review_df = build_name_map(
        all_names,
        players,
        fuzzy_threshold=args.fuzzy_threshold,
        fuzzy_margin=args.fuzzy_margin,
    )

    args.unmatched_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_df.to_csv(
        args.unmatched_report,
        index=False,
    )

    unresolved_methods = {
        "unmatched",
        "ambiguous_surname_initial",
        "fuzzy_ambiguous_or_low_score",
        "invalid_name",
    }

    unresolved = review_df[
        review_df["match_method"].isin(
            unresolved_methods
        )
    ]

    print(
        f"Nomi Tennis-Data distinti: "
        f"{len(review_df):,}"
    )

    print(
        f"Nomi risolti: "
        f"{len(review_df) - len(unresolved):,}"
    )

    print(
        f"Nomi non risolti/ambigui: "
        f"{len(unresolved):,}"
    )

    print(
        f"Report matching: "
        f"{args.unmatched_report}"
    )

    # ------------------------------------------------------------
    # IDs
    # ------------------------------------------------------------

    cleaned["winner_sackmann_id"] = (
        cleaned["winner_name"]
        .map(name_map)
    )

    cleaned["loser_sackmann_id"] = (
        cleaned["loser_name"]
        .map(name_map)
    )

    # ------------------------------------------------------------
    # Bio
    # ------------------------------------------------------------

    bio = players.set_index(
        "player_id"
    )

    def get_bio(
            player_id: Any,
            field: str,
    ):
        if pd.isna(player_id):
            return np.nan

        player_id = str(player_id)

        if player_id not in bio.index:
            return np.nan

        return bio.loc[
            player_id,
            field,
        ]

    cleaned["winner_height"] = (
        cleaned["winner_sackmann_id"]
        .map(
            lambda x: get_bio(x, "height")
        )
    )

    cleaned["loser_height"] = (
        cleaned["loser_sackmann_id"]
        .map(
            lambda x: get_bio(x, "height")
        )
    )

    cleaned["winner_hand"] = (
        cleaned["winner_sackmann_id"]
        .map(
            lambda x: get_bio(x, "hand")
        )
    )

    cleaned["loser_hand"] = (
        cleaned["loser_sackmann_id"]
        .map(
            lambda x: get_bio(x, "hand")
        )
    )

    winner_dob = (
        cleaned["winner_sackmann_id"]
        .map(
            lambda x: get_bio(x, "birth_date")
        )
    )

    loser_dob = (
        cleaned["loser_sackmann_id"]
        .map(
            lambda x: get_bio(x, "birth_date")
        )
    )

    cleaned["winner_age_years"] = (
            (
                    cleaned["date"]
                    - pd.to_datetime(
                winner_dob,
                errors="coerce",
            )
            ).dt.days
            / 365.25
    )

    cleaned["loser_age_years"] = (
            (
                    cleaned["date"]
                    - pd.to_datetime(
                loser_dob,
                errors="coerce",
            )
            ).dt.days
            / 365.25
    )

    # Tournament start dates cannot establish when service statistics became available.
    cleaned["height_diff"] = cleaned["winner_height"] - cleaned["loser_height"]
    cleaned["age_diff"] = cleaned["winner_age_years"] - cleaned["loser_age_years"]

    # ------------------------------------------------------------
    # Restore original order
    # ------------------------------------------------------------

    cleaned = cleaned.sort_values(
        "_cleaned_row_id"
    ).reset_index(
        drop=True
    )

    cleaned = cleaned.drop(
        columns=[
            "_cleaned_row_id",
        ]
    )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_csv(
        args.output,
        index=False,
    )

    # ------------------------------------------------------------
    # Coverage report
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("COVERAGE")
    print("=" * 78)

    coverage_columns = [
        "winner_sackmann_id",
        "loser_sackmann_id",
        "winner_height",
        "loser_height",
        "winner_age_years",
        "loser_age_years",
        "winner_ace_rate_before",
        "loser_ace_rate_before",
        "winner_df_rate_before",
        "loser_df_rate_before",
        "winner_first_in_rate_before",
        "loser_first_in_rate_before",
        "winner_first_won_rate_before",
        "loser_first_won_rate_before",
        "winner_second_won_rate_before",
        "loser_second_won_rate_before",
        "winner_bp_saved_rate_before",
        "loser_bp_saved_rate_before",
        "winner_bp_faced_per_sv_game_before",
        "loser_bp_faced_per_sv_game_before",
    ]

    for col in coverage_columns:

        if col not in cleaned.columns:
            continue

        pct = (
                cleaned[col]
                .notna()
                .mean()
                * 100
        )

        print(
            f"{col:<42} {pct:6.2f}%"
        )

    # ------------------------------------------------------------
    # Leakage sanity checks
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)

    bad_age = (
            (cleaned["winner_age_years"] < 12)
            | (cleaned["winner_age_years"] > 60)
    ).sum()

    bad_age += (
            (cleaned["loser_age_years"] < 12)
            | (cleaned["loser_age_years"] > 60)
    ).sum()

    print(
        f"Età palesemente anomale: {bad_age}"
    )

    rate_columns = [
        "winner_ace_rate_before",
        "loser_ace_rate_before",
        "winner_df_rate_before",
        "loser_df_rate_before",
        "winner_first_in_rate_before",
        "loser_first_in_rate_before",
        "winner_first_won_rate_before",
        "loser_first_won_rate_before",
        "winner_second_won_rate_before",
        "loser_second_won_rate_before",
        "winner_bp_saved_rate_before",
        "loser_bp_saved_rate_before",
    ]

    out_of_range = 0

    for col in rate_columns:

        if col not in cleaned.columns:
            continue

        invalid = (
                cleaned[col].notna()
                & (
                        (cleaned[col] < 0)
                        | (cleaned[col] > 1)
                )
        )

        out_of_range += int(
            invalid.sum()
        )

    print(
        f"Rate fuori [0,1]: {out_of_range}"
    )

    print()
    print(
        f"Dataset finale: "
        f"{cleaned.shape[0]:,} righe × "
        f"{cleaned.shape[1]:,} colonne"
    )

    print(
        f"Output: {args.output}"
    )

    print()
    print(
        "ENRICHMENT COMPLETATO."
    )

    print(
        "Il file può essere passato a build_features.py."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
