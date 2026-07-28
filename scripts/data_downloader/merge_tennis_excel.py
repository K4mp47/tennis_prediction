"""Merge yearly Tennis-Data Excel workbooks into one canonical .data file.

The .data extension has no special binary format here: the output is a normal,
UTF-8, comma-separated text file with a header.

Run from the project root:

    uv add pandas openpyxl
    uv run python scripts/merge_tennis_excel.py
"""

from pathlib import Path
import re

import pandas as pd


RAW_DIR = Path("../../data/raw")
OUTPUT_FILE = Path("../../data/interim/tennis_matches_raw.data")

CANONICAL_COLUMNS = [
    "ATP", "Location", "Tournament", "Date", "Series", "Court", "Surface",
    "Round", "Best of", "Winner", "Loser", "WRank", "LRank", "WPts", "LPts",
    "W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5",
    "Wsets", "Lsets", "Comment", "B365W", "B365L", "PSW", "PSL",
    "MaxW", "MaxL", "AvgW", "AvgL", "BFEW", "BFEL",
]


def extract_year(path: Path) -> int:
    match = re.search(r"(20\d{2})", path.stem)
    if not match:
        raise ValueError(f"Cannot determine year from {path.name}")
    return int(match.group(1))


def load_and_normalize(path: Path) -> pd.DataFrame:
    year = extract_year(path)
    frame = pd.read_excel(path)

    # Remove accidental empty columns such as "Unnamed: 36".
    frame = frame.dropna(axis="columns", how="all")
    frame.columns = frame.columns.astype(str).str.strip()

    # Preserve the union schema. Columns unavailable in a year become NA.
    frame = frame.reindex(columns=CANONICAL_COLUMNS)

    frame.insert(0, "source_year", year)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return frame


def main() -> None:
    files = sorted(RAW_DIR.glob("20*.xlsx"), key=extract_year)
    if not files:
        raise FileNotFoundError(f"No yearly .xlsx files found in {RAW_DIR}")

    frames = [load_and_normalize(path) for path in files]
    merged = pd.concat(frames, ignore_index=True)

    merged = merged.sort_values(
        ["Date", "ATP", "Tournament", "Winner", "Loser"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Created {OUTPUT_FILE}")
    print(f"Rows: {len(merged):,}")
    print(f"Columns: {len(merged.columns)}")
    print(merged.groupby("source_year").size().to_string())


if __name__ == "__main__":
    main()
