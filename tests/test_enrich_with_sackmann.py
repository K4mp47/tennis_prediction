import sys
from pathlib import Path

import pandas as pd
import pytest


CLEANER_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data_cleaner"
sys.path.insert(0, str(CLEANER_DIR))

from enrich_with_sackmann import build_name_map  # noqa: E402


@pytest.mark.parametrize(
    ("name", "full_names", "expected_map", "method"),
    [
        (
            "Djokovc N.",
            ["Novak Djokovic"],
            {"Djokovc N.": "0"},
            "fuzzy_surname",
        ),
        (
            "Djokovc N.",
            ["Novak Djokovic", "Nikola Djokovoc"],
            {},
            "fuzzy_ambiguous_or_low_score",
        ),
        (
            "Unknown N.",
            ["Novak Djokovic"],
            {},
            "fuzzy_ambiguous_or_low_score",
        ),
    ],
)
def test_fuzzy_matching_requires_a_confident_unique_candidate(
    name: str,
    full_names: list[str],
    expected_map: dict[str, str],
    method: str,
) -> None:
    players = pd.DataFrame(
        {
            "player_id": [str(i) for i in range(len(full_names))],
            "full_name": full_names,
        }
    )

    name_map, review = build_name_map(pd.Series([name]), players, 92, 5)

    assert name_map == expected_map
    assert review["match_method"].tolist() == [method]
