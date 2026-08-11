# Tennis Match Prediction

This project aims to predict the outcomes of ATP tennis matches using historical match data and machine learning techniques. The project includes data collection and cleaning, feature engineering, integration of additional player statistics, and predictive modeling.

## Data Sources

The primary data source is the yearly Excel files from Tennis-Data, containing historical ATP match results and related match statistics.

Additional player and match statistics are obtained from the Jeff Sackmann Tennis ATP dataset. These data are used to enrich the dataset with player characteristics and historical service and return statistics.

## Installation

To run the project locally, Python 3.12 or newer and `uv` are required.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install the project dependencies:

```bash
uv sync
```

## Download the data

Raw Tennis-Data Excel files are downloaded into `data/raw/`. These files are local-only and are not committed to Git.

To download a range of yearly files:

```bash
uv run python scripts/download_data.py --start-year 2015 --end-year 2026
```

The download script also generates `data/raw/manifest.json`, which records the source URL, local path, file size, SHA-256 checksum, and raw-data policy for each downloaded file.

If the Excel files are already available locally and only the manifest needs to be regenerated:

```bash
uv run python scripts/create_manifest.py --start-year 2015 --end-year 2026
```

The yearly Excel files can then be merged into a single CSV file using:

```bash
uv run python scripts/merge_tennis_excel.py
```

## Data Preparation

The initial data cleaning and preparation are performed in `classification.ipynb`, located in the `docs` directory. This step includes cleaning the raw Tennis-Data data and preparing the dataset for the subsequent analysis.

## Feature Engineering and Modeling

The feature engineering and modeling pipeline builds pre-match features using only information available before each match, in order to avoid data leakage.

The features include player ranking and experience, recent form, surface-specific performance, rest time, head-to-head statistics, market odds, and historical service and return statistics obtained from the Jeff Sackmann dataset.

The dataset is also symmetrized by generating a second observation for each match with the players' roles reversed. This makes the model independent of the arbitrary order in which the players are presented.

The resulting dataset is used to train and evaluate machine learning models using time-based cross-validation.

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
