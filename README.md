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
6. Train and evaluate the machine learning models.

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

The `docs/classification.ipynb` notebook remains available for exploratory analysis and represents a separate data-analysis path from the normalization/enrichment pipeline described above.

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

The script writes `data/interim/tennis_matches_enriched.data` and the name-resolution report `data/interim/sackmann_name_matching_review.csv`.

It adds player height, hand, age, and historical ace, double-fault, first-serve, second-serve, and break-point statistics. Historical values are matched using only data before the current match; matches played on the same date are excluded. Ambiguous or unresolved player names are retained in the report rather than being guessed.

## Feature Engineering

Build the final feature dataset from the enriched data:

```bash
uv run python scripts/data_cleaner/build_features.py \
  --input data/interim/tennis_matches_enriched.data \
  --rare-threshold 50 \
  --rfecv-step 15 \
  --rfecv-min-features 15
```

The command accepts `--input`, `--output-features`, and `--output-metadata` path overrides. The main tuning options are `--rare-threshold`, `--rfecv-step`, and `--rfecv-min-features`.

The feature-building pipeline uses only information available before each match, in order to avoid data leakage. It creates:

* Global player experience and win rate.
* Recent five-match form.
* Surface-specific history and rest time.
* Head-to-head history.
* Ranking, ranking/points differences, and market-implied probabilities.
* Tournament and location category buckets.
* Sackmann player biography and historical serve/return statistics.

Each real match is represented twice in the model data: once in the original orientation and once with the players swapped. The target column is `player_a_win`, which removes dependence on the arbitrary winner/loser orientation.

For the current dataset, the feature-building pipeline produced 56,292 rows, corresponding to 28,146 real matches and their mirrored representations. RFECV reduced the encoded feature space from 270 features to 135 selected features, with a best RFECV cross-validation accuracy of 0.6809.

The resulting dataset is used as input for the machine learning models.

The command produces:

* `data/interim/tennis_matches_features.data`
* `data/interim/tennis_matches_features_metadata.json`

The metadata file records the input and output schema, feature lists, pipeline parameters, baseline fold scores, RFECV scores, and selected features.

## Machine Learning Models

### Random Forest

The feature dataset is used to train a `RandomForestClassifier` as one of the project's predictive models.

Evaluation uses chronological `TimeSeriesSplit` cross-validation. The temporal ordering prevents future matches from being used to train models evaluated on earlier data.

The Random Forest is intended as a higher-capacity ensemble model and provides a reference point for comparison with the single Decision Tree.

### Decision Tree with Gini criterion

A second predictive model is implemented using a `DecisionTreeClassifier` with the Gini impurity criterion.

Train the model with:

```bash
uv run python scripts/models/train_decision_tree.py \
  --input data/interim/tennis_matches_features.data \
  --metadata data/interim/tennis_matches_features_metadata.json
```

The Decision Tree training pipeline:

* Uses `criterion="gini"`.
* Reserves the last 15% of unique dates as a chronological holdout set.
* Uses five cross-validation folds based on unique dates.
* Keeps the two mirrored observations of the same match in the same fold.
* Performs grid search over tree depth, minimum leaf size, minimum split size, and cost-complexity pruning.
* Evaluates the selected model on the final chronological holdout, which is not used during hyperparameter tuning.
* Reports feature importances and tree structure.

For the current dataset, the best model using all features selected:

```text
max_depth = 4
min_samples_leaf = 1
min_samples_split = 2
ccp_alpha = 0.001
```

The resulting performance was:

| Model                               | CV accuracy | Holdout accuracy |
| ----------------------------------- | ----------: | ---------------: |
| Decision Tree — with market odds    |      0.6822 |           0.6864 |
| Decision Tree — without market odds |      0.6390 |           0.6492 |

### Analysis of market odds

The Decision Tree using all features assigns essentially all feature importance to:

```text
implied_prob_diff_avg
```

with an importance of 1.0000. This indicates that the market-implied probability difference is highly predictive and that the tree can obtain most of its predictive power from this single feature.

To evaluate the independent contribution of the engineered player and historical features, a second experiment was performed excluding:

```text
winner_implied_prob_avg
loser_implied_prob_avg
implied_prob_diff_avg
has_any_market_odds
```

Without market-odds information, the Decision Tree achieved 0.6492 accuracy on the chronological holdout. The most important features were:

```text
rank_diff                  0.6488
points_diff                0.2262
surface_win_rate_diff      0.0978
```

This demonstrates that the feature-engineering pipeline provides predictive information independently of betting-market probabilities, although market odds provide a substantial additional improvement.

The comparison also provides an interpretable distinction between two sources of predictive information: the market's aggregated pre-match assessment and player-level historical/statistical features.

## Model Outputs

The Decision Tree training script produces a JSON file containing:

* Best hyperparameters.
* Cross-validation accuracy.
* Chronological holdout accuracy.
* Confusion matrix.
* Final tree depth and number of leaves.
* Top feature importances.
* A textual representation of the tree.

The current experiments are stored as:

```text
data/interim/decision_tree_metrics.json
data/interim/decision_tree_no_odds.json
```

The first file corresponds to the model using all features, including market odds. The second corresponds to the experiment excluding market-odds features.

The tree visualization is optional and requires `matplotlib`.

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
