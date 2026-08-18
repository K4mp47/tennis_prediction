Tennis match prediction

This project aims to predict the outcomes of tennis matches using historical data and machine learning techniques.

Data Sources

The primary data source for this project is the yearly Excel files from Tennis-Data. The data includes match statistics, player information, and other relevant features.

Enrichment data comes from Jeff Sackmann's tennis_atp repository (player biographical data and match-level serve/return statistics). Like the raw Tennis-Data files, this data is local-only and not committed to Git — see Download the enrichment data below.

Installation

To run this project locally, you need Python 3.12 or newer and uv for project management.

bash
curl -LsSf https://astral.sh/uv/install.sh | sh

Clone the repository, then install the project dependencies:

bash
uv sync
Download the data

Raw Tennis-Data Excel files are downloaded into data/raw/. The data files are local-only and are not committed to Git. Download a range of yearly files with:

bash
uv run python scripts/download_data.py --start-year 2015 --end-year 2026

The download script also regenerates data/raw/manifest.json. The manifest records the source URL, local path, file size, SHA-256 checksum, and immutable raw-data policy for each downloaded file. If you already have the Excel files locally and only want to rebuild the manifest, run:

bash
uv run python scripts/create_manifest.py --start-year 2015 --end-year 2026

run the merge_tennis_excel.py script to merge the yearly Excel files into a single CSV file:

bash
uv run python scripts/merge_tennis_excel.py
Data preparation

Data preparation is performed in the classification.ipynb notebook. This notebook reads the merged CSV file, cleans the data, and prepares it for machine learning models. The notebook is available in the docs directory and can be run in a Jupyter environment.

Feature engineering

Feature engineering is a separate, independent pipeline from the classification.ipynb notebook above: it is a multi-step set of scripts in scripts/data_cleaner/, run from the project root. Every script writes its output to data/interim/ (also local-only, not committed to Git).

1. Clean the raw match data

Reads the Tennis-Data raw file, removes incomplete/retired matches and invalid values, and normalizes column names and categorical values:

bash
uv run python scripts/data_cleaner/normalize_data.py \
  --input data/interim/tennis_matches_raw.data \
  --output data/interim/tennis_matches_cleaned.data

2. Enrich with Jeff Sackmann's tennis_atp data

Adds player biographical data (height, dominant hand, age) and pre-match serve/return statistics (ace rate, break points saved, etc.), matched to Tennis-Data players by name. See Download the enrichment data for how to obtain external/tennis_atp/.

bash
uv add rapidfuzz  # optional but recommended, improves player name matching

uv run python scripts/data_cleaner/enrich_with_sackmann.py \
  --cleaned-input data/interim/tennis_matches_cleaned.data \
  --sackmann-dir external/tennis_atp \
  --output data/interim/tennis_matches_enriched.data

This also writes data/interim/sackmann_name_matching_review.csv, a report of every player-name match decision (exact, fuzzy, or unresolved) — worth reviewing manually if match coverage looks low.

3. Build model-ready features

Computes pre-match player history, head-to-head, and market-odds features, builds a symmetric (player-order-independent) dataset, and runs a Random Forest baseline with RFECV feature selection:

bash
uv run python scripts/data_cleaner/build_features.py \
  --input data/interim/tennis_matches_enriched.data \
  --rare-threshold 50 \
  --rfecv-step 15 \
  --rfecv-min-features 15

Produces data/interim/tennis_matches_features.data and data/interim/tennis_matches_features_metadata.json (feature list, RFECV results, and baseline CV metrics).

Download the enrichment data

Clone Jeff Sackmann's repository locally (not committed to Git, same policy as the raw Tennis-Data files):

bash
git clone https://github.com/JeffSackmann/tennis_atp.git external/tennis_atp

Licensed under CC BY-NC-SA 4.0 — non-commercial use with attribution.

Run locally

Run the package entry point, still to implement:

bash
uv run tennis-prediction

At the moment this prints a placeholder message while the prediction pipeline is being built.

Development checks

Run the current test suite, still to implement:

bash
uv run pytest

Run the linter:

bash
uv run ruff check .
