# Tennis match prediction

This project aims to predict the outcomes of tennis matches using historical data and machine learning techniques.

## Data Sources

The primary data source for this project is the yearly Excel files from Tennis-Data. The data includes match statistics, player information, and other relevant features.

## Installation

To run this project locally, you need Python 3.12 or newer and `uv` for project management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository, then install the project dependencies:

```bash
uv sync
```

## Download the data

Raw Tennis-Data Excel files are downloaded into `data/raw/`. The data files are local-only and are not committed to Git.

Download a range of yearly files with:

```bash
uv run python scripts/download_data.py --start-year 2025 --end-year 2026
```

The download script also regenerates `data/raw/manifest.json`. The manifest records the source URL, local path, file size, SHA-256 checksum, and immutable raw-data policy for each downloaded file.

If you already have the Excel files locally and only want to rebuild the manifest, run:

```bash
uv run python scripts/create_manifest.py --start-year 2025 --end-year 2026
```

run the `merge_tennis_excel.py` script to merge the yearly Excel files into a single CSV file:

```bash
uv run python scripts/merge_tennis_excel.py
```

## Data preparation

Data preparation is performed in the `classification.ipynb` notebook. This notebook reads the merged CSV file, cleans the data, and prepares it for machine learning models. The notebook is available in the `docs` directory and can be run in a Jupyter environment.

## Run locally

Run the package entry point, **still to implement**:

```bash
uv run tennis-prediction
```

At the moment this prints a placeholder message while the prediction pipeline is being built.

## Development checks

Run the current test suite, **still to implement**:

```bash
uv run pytest
```

Run the linter:

```bash
uv run ruff check .
```
