from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "interim" / "tennis_matches_cleaned.data"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "tennis_matches_enriched.data"
DEFAULT_METADATA = REPO_ROOT / "data" / "processed" / "tennis_matches_enriched.metadata.json"
ELO_K_FACTOR = 32.0
ELO_BASE = 1500.0
ROUND_STAGES = {
    "1st Round": 1,
    "2nd Round": 2,
    "3rd Round": 3,
    "4th Round": 4,
    "Quarterfinals": 5,
    "Semifinals": 6,
    "Final": 7,
}
REQUIRED_COLUMNS = {
    "date",
    "surface",
    "court",
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
}


@dataclass
class PlayerState:
    matches: int = 0
    wins: int = 0
    elo: float = ELO_BASE
    last_date: pd.Timestamp | None = None
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    surface_stats: dict[str, list[int]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an enriched feature dataset from cleaned tennis matches."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def as_float(value: Any) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return float("nan")
    return float(parsed)


def safe_win_rate(wins: int, matches: int) -> float:
    if matches <= 0:
        return float("nan")
    return wins / matches


def recent_rate(results: deque[int]) -> float:
    if not results:
        return float("nan")
    return float(sum(results) / len(results))


def expected_elo(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def canonical_pair(player_a: str, player_b: str) -> tuple[str, str]:
    if player_a <= player_b:
        return player_a, player_b
    return player_b, player_a


def read_round_stage(value: Any) -> int:
    if pd.isna(value):
        return 0
    return ROUND_STAGES.get(str(value), 0)


def pick_match_odds(
    row: pd.Series,
    winner_column: str,
    loser_column: str,
) -> tuple[float, float]:
    sources = [
        ("avg_winner", "avg_loser"),
        ("max_winner", "max_loser"),
        ("ps_winner", "ps_loser"),
        ("b365_winner", "b365_loser"),
        ("bfe_winner", "bfe_loser"),
    ]

    for winner_source, loser_source in sources:
        winner_odds = as_float(row[winner_source])
        loser_odds = as_float(row[loser_source])
        if winner_odds > 1.0 and loser_odds > 1.0:
            if winner_column == "winner":
                return winner_odds, loser_odds
            return loser_odds, winner_odds

    return float("nan"), float("nan")


def update_surface_stats(state: PlayerState, surface: str, won: bool) -> None:
    wins, matches = state.surface_stats.get(surface, [0, 0])
    matches += 1
    if won:
        wins += 1
    state.surface_stats[surface] = [wins, matches]


def get_surface_rate(state: PlayerState, surface: str) -> float:
    stats = state.surface_stats.get(surface)
    if not stats:
        return float("nan")
    wins, matches = stats
    return safe_win_rate(wins, matches)


def create_enriched_dataset(cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_columns(cleaned_df)

    df = cleaned_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "winner_name", "loser_name"]).copy()
    df = df.sort_values(["date", "source_year", "atp", "tournament"], kind="mergesort").reset_index(drop=True)

    player_states: dict[str, PlayerState] = {}
    head_to_head: dict[tuple[str, str], dict[str, int]] = {}
    enriched_rows: list[dict[str, Any]] = []
    skipped_same_player = 0

    for match_id, row in df.iterrows():
        winner = str(row["winner_name"])
        loser = str(row["loser_name"])

        if winner == loser:
            skipped_same_player += 1
            continue

        winner_state = player_states.setdefault(winner, PlayerState())
        loser_state = player_states.setdefault(loser, PlayerState())

        player_a, player_b = canonical_pair(winner, loser)
        player_a_won = int(player_a == winner)

        pair_key = canonical_pair(player_a, player_b)
        h2h = head_to_head.setdefault(pair_key, {pair_key[0]: 0, pair_key[1]: 0})

        if player_a_won == 1:
            player_a_rank = as_float(row["winner_rank"])
            player_b_rank = as_float(row["loser_rank"])
            player_a_points = as_float(row["winner_points"])
            player_b_points = as_float(row["loser_points"])
            player_a_state = winner_state
            player_b_state = loser_state
            player_a_odds, player_b_odds = pick_match_odds(row, "winner", "loser")
        else:
            player_a_rank = as_float(row["loser_rank"])
            player_b_rank = as_float(row["winner_rank"])
            player_a_points = as_float(row["loser_points"])
            player_b_points = as_float(row["winner_points"])
            player_a_state = loser_state
            player_b_state = winner_state
            player_a_odds, player_b_odds = pick_match_odds(row, "loser", "winner")

        date_value = pd.Timestamp(row["date"])
        surface = str(row["surface"]) if not pd.isna(row["surface"]) else "Unknown"

        player_a_days_since = (
            float((date_value - player_a_state.last_date).days)
            if player_a_state.last_date is not None
            else float("nan")
        )
        player_b_days_since = (
            float((date_value - player_b_state.last_date).days)
            if player_b_state.last_date is not None
            else float("nan")
        )

        implied_a = 1.0 / player_a_odds if player_a_odds > 1.0 else float("nan")
        implied_b = 1.0 / player_b_odds if player_b_odds > 1.0 else float("nan")
        overround = implied_a + implied_b if np.isfinite(implied_a) and np.isfinite(implied_b) else float("nan")
        implied_a_normalized = implied_a / overround if np.isfinite(overround) and overround > 0 else float("nan")
        implied_b_normalized = implied_b / overround if np.isfinite(overround) and overround > 0 else float("nan")

        enriched_rows.append(
            {
                "match_id": int(match_id),
                "match_date": date_value.strftime("%Y-%m-%d"),
                "source_year": row.get("source_year"),
                "atp": row.get("atp"),
                "tournament": row.get("tournament"),
                "surface": surface,
                "court": row.get("court"),
                "series": row.get("series"),
                "round": row.get("round"),
                "round_stage": read_round_stage(row.get("round")),
                "best_of": row.get("best_of"),
                "player_a_name": player_a,
                "player_b_name": player_b,
                "player_a_win": player_a_won,
                "player_a_rank": player_a_rank,
                "player_b_rank": player_b_rank,
                "player_a_points": player_a_points,
                "player_b_points": player_b_points,
                "rank_diff_player_a": player_b_rank - player_a_rank,
                "points_diff_player_a": player_a_points - player_b_points,
                "player_a_matches_before": player_a_state.matches,
                "player_b_matches_before": player_b_state.matches,
                "player_a_wins_before": player_a_state.wins,
                "player_b_wins_before": player_b_state.wins,
                "player_a_win_rate_before": safe_win_rate(player_a_state.wins, player_a_state.matches),
                "player_b_win_rate_before": safe_win_rate(player_b_state.wins, player_b_state.matches),
                "player_a_recent5_win_rate_before": recent_rate(player_a_state.recent_results),
                "player_b_recent5_win_rate_before": recent_rate(player_b_state.recent_results),
                "player_a_surface_win_rate_before": get_surface_rate(player_a_state, surface),
                "player_b_surface_win_rate_before": get_surface_rate(player_b_state, surface),
                "player_a_elo_before": player_a_state.elo,
                "player_b_elo_before": player_b_state.elo,
                "elo_diff_player_a": player_a_state.elo - player_b_state.elo,
                "head_to_head_matches_before": h2h[player_a] + h2h[player_b],
                "head_to_head_player_a_wins_before": h2h[player_a],
                "head_to_head_player_b_wins_before": h2h[player_b],
                "player_a_days_since_last_match": player_a_days_since,
                "player_b_days_since_last_match": player_b_days_since,
                "player_a_decimal_odds": player_a_odds,
                "player_b_decimal_odds": player_b_odds,
                "market_overround": overround,
                "player_a_market_implied_prob": implied_a,
                "player_b_market_implied_prob": implied_b,
                "player_a_market_prob_normalized": implied_a_normalized,
                "player_b_market_prob_normalized": implied_b_normalized,
            }
        )

        winner_expected = expected_elo(winner_state.elo, loser_state.elo)
        loser_expected = 1.0 - winner_expected
        winner_state.elo += ELO_K_FACTOR * (1.0 - winner_expected)
        loser_state.elo += ELO_K_FACTOR * (0.0 - loser_expected)

        winner_state.matches += 1
        winner_state.wins += 1
        winner_state.recent_results.append(1)
        winner_state.last_date = date_value
        update_surface_stats(winner_state, surface, won=True)

        loser_state.matches += 1
        loser_state.recent_results.append(0)
        loser_state.last_date = date_value
        update_surface_stats(loser_state, surface, won=False)

        h2h[winner] += 1

    enriched_df = pd.DataFrame(enriched_rows)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": int(len(cleaned_df)),
        "output_rows": int(len(enriched_df)),
        "output_columns": list(enriched_df.columns),
        "skipped_same_player_matches": int(skipped_same_player),
        "elo_k_factor": ELO_K_FACTOR,
        "feature_groups": [
            "contextual_match_features",
            "rank_and_points_deltas",
            "historical_player_form",
            "surface_form",
            "elo_strength",
            "head_to_head_history",
            "rest_days",
            "betting_market_features",
        ],
    }
    return enriched_df, metadata


def main() -> None:
    args = parse_args()
    cleaned_df = pd.read_csv(args.input)
    enriched_df, metadata = create_enriched_dataset(cleaned_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)

    enriched_df.to_csv(args.output, index=False)
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Created enriched dataset: {args.output}")
    print(f"Created metadata: {args.metadata_output}")
    print(f"Rows: {metadata['input_rows']:,} -> {metadata['output_rows']:,}")


if __name__ == "__main__":
    main()
