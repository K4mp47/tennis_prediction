# Tennis Match Prediction

This project aims to predict the outcomes of ATP tennis matches using historical match data and machine learning techniques. The project includes data collection and cleaning, feature engineering, integration of additional player statistics, and predictive modeling.

## Data Sources

The primary data source is the yearly Excel files from Tennis-Data, containing historical ATP match results and related match statistics.

Additional player and match statistics are obtained from the [Jeff Sackmann Tennis ATP dataset](https://github.com/JeffSackmann/tennis_atp). These data are used to enrich the dataset with player characteristics and historical service and return statistics.

Raw Tennis-Data files and the external Sackmann repository are local inputs and are not committed to Git.

## Installation

To run the project locally, Python 3.12 or newer and `uv` are required.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install the project dependencies:

```bash
uv sync
```

## Data Pipeline

The reproducible pipeline is:

1. Download yearly Tennis-Data workbooks.
2. Merge the workbooks into one canonical CSV file.
3. Normalize and clean the match data.
4. Enrich the cleaned data with pre-match Sackmann features.
5. Build the final feature dataset and run baseline modeling and RFECV.

### Download Tennis-Data

Raw Tennis-Data Excel files are downloaded into `data/raw/`. These files are local-only and are not committed to Git.

To download a range of yearly files:

```bash
uv run python scripts/data_downloader/download_data.py --start-year 2015 --end-year 2026
```

The download script also generates `data/raw/manifest.json`, which records the source URL, local path, file size, SHA-256 checksum, and raw-data policy for each downloaded file.

If the Excel files are already available locally and only the manifest needs to be regenerated:

```bash
uv run python scripts/data_downloader/create_manifest.py --start-year 2015 --end-year 2026
```

The yearly Excel files can then be merged into `data/interim/tennis_matches_raw.data` using:

```bash
uv run --directory scripts/data_downloader python merge_tennis_excel.py
```

### Normalize the data

Normalize the merged data before adding historical features:

```bash
uv run python scripts/data_cleaner/normalize_data.py
```

This step removes duplicate and incomplete matches, drops post-match score columns, normalizes dates and categories, and treats invalid ranks, points, and odds as missing. It writes `data/interim/tennis_matches_cleaned.data`.

The `docs/classification.ipynb` notebook remains available for exploratory analysis.

### Enrich with Sackmann data

Clone the external dataset, or place an equivalent local checkout containing `atp_players.csv` and the required `atp_matches_YYYY.csv` files:

```bash
git clone --depth 1 https://github.com/JeffSackmann/tennis_atp.git external/tennis_atp
```

Run the enrichment step:

```bash
uv run python scripts/data_cleaner/enrich_with_sackmann.py \
  --sackmann-dir external/tennis_atp
```

The script writes `data/interim/tennis_matches_enriched.data` and the name-resolution report `data/interim/sackmann_name_matching_review.csv`. It adds player height, hand, age, and historical ace, double-fault, first-serve, second-serve, and break-point statistics. Historical values are matched using only data before the current match; matches played on the same date are excluded. Ambiguous or unresolved player names are retained in the report rather than being guessed.

## Feature Engineering and Modeling

Build features from the enriched dataset and run the baseline model and RFECV feature selection:

```bash
uv run python scripts/data_cleaner/build_features.py
```

The command accepts `--input`, `--output-features`, and `--output-metadata` path overrides. The main tuning options are `--rare-threshold`, `--rfecv-step`, and `--rfecv-min-features`.

The feature-building pipeline uses only information available before each match, in order to avoid data leakage. It creates:

- Global player experience and win rate.
- Recent five-match form.
- Surface-specific history and rest time.
- Head-to-head history.
- Ranking, ranking/points differences, and market-implied probabilities.
- Tournament and location category buckets.
- Sackmann player biography and historical serve/return statistics.

Each real match is represented twice in the model data: once in the original orientation and once with the players swapped. The target column is `player_a_win`, which removes dependence on the arbitrary winner/loser orientation.

The resulting dataset is used to train a `RandomForestClassifier`. Evaluation uses `TimeSeriesSplit` cross-validation, with the baseline cross-validation score used as the primary generalization estimate. RFECV reports a diagnostic best score and selected encoded features.

The command produces:

- `data/interim/tennis_matches_features.data`
- `data/interim/tennis_matches_features_metadata.json`

The metadata file records the input and output schema, feature lists, pipeline parameters, baseline fold scores, RFECV scores, and selected features.

## Run locally

The main package entry point is currently under development:

```bash
uv run tennis-prediction
```

At the moment, this command provides a placeholder while the final prediction pipeline is being integrated into the package.

## Development Checks

Run the test suite with:

```bash
uv run pytest
```

Run the linter with:

```bash
uv run ruff check .
```
