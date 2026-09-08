# Tennis Match Prediction

This project aims to predict the outcomes of ATP tennis matches using historical match data and machine learning techniques. The project includes data collection and cleaning, feature engineering, integration of additional player statistics, and predictive modeling.

## Data Sources

The primary data source is the yearly Excel files from Tennis-Data, containing historical ATP match results and related match statistics.

Additional player and match statistics come from Jeff Sackmann's ATP dataset.
The original upstream repository is currently unavailable, so the reproducible
workflow uses the [June 2026 archival mirror](https://github.com/Aneeshers/tennis-sackmann-archive).
The data remains attributed to Jeff Sackmann and licensed CC BY-NC-SA 4.0;
commercial use is not permitted by that license. These data enrich the project
with player characteristics and historical service and return statistics.

Raw Tennis-Data files and the external Sackmann repository are local inputs and are not committed to Git.

Enrichment data comes from [Jeff Sackmann's `tennis_atp` repository](https://github.com/JeffSackmann/tennis_atp) (player biographical data and match-level serve/return statistics). Like the raw Tennis-Data files, this data is local-only and not committed to Git — see [Download the enrichment data](#download-the-enrichment-data) below.
## Installation

To run the project locally, Python 3.12 or newer and `uv` are required.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install the project dependencies:

```bash
uv sync --all-groups
```

`--all-groups` installs the notebook execution tools in addition to the runtime
dependencies.

## CatBoost: complete run from scratch

Run these commands from the project root. Each command consumes the artifact
created by the previous command.

```bash
# 1. Install runtime, test, and notebook dependencies.
uv sync --all-groups

# 2. Download Tennis-Data workbooks and create their manifest.
uv run python scripts/data_downloader/download_data.py \
  --start-year 2015 \
  --end-year 2026

# 3. Merge the yearly workbooks.
uv run --directory scripts/data_downloader python merge_tennis_excel.py

# 4. Normalize matches and remove post-match/leaking columns.
uv run python scripts/data_cleaner/normalize_data.py

# 5. Download the June 2026 Sackmann archival mirror.
git clone --depth 1 \
  https://github.com/Aneeshers/tennis-sackmann-archive.git \
  external/tennis-sackmann-archive

# 6. Add point-in-time biography, service, and return features.
uv run python scripts/data_cleaner/enrich_with_sackmann.py \
  --sackmann-dir external/tennis-sackmann-archive/atp

# 7. Build the symmetric, leakage-safe model table.
# CatBoost uses fixed parameters, so the unrelated
# Random-Forest/RFECV diagnostic is skipped here.
uv run python scripts/data_cleaner/build_features.py \
  --input data/interim/tennis_matches_enriched.data \
  --rare-threshold 50 \
  --skip-model-analysis

# 8. Train the fixed-parameter model and evaluate the newest 15% of dates.
uv run python scripts/models/train_catboost.py

# 9. Recreate and execute the complete analysis notebook.
uv run python scripts/models/create_catboost_notebook.py --execute

# Optional Italian-language copy.
uv run python scripts/models/create_catboost_notebook.py --language it --execute
```

The training step creates:

```text
data/interim/catboost_model.cbm
data/interim/catboost_metrics.json
```

The notebook step creates `docs/catboost_training_analysis.ipynb`; the Italian
command creates `docs/catboost_training_analysis_it.ipynb`.

The trainer uses 300 trees, depth 8, learning rate 0.03 and L2 regularization 3.
It does not run a parameter search or cross-validation.

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

Reads the Tennis-Data raw file, removes incomplete/retired matches and invalid values, and normalizes column names and categorical values:
```bash
uv run python scripts/data_cleaner/normalize_data.py
```

This step removes duplicate and incomplete matches, drops post-match score columns, normalizes dates and categories, and treats invalid ranks, points, and odds as missing. It writes `data/interim/tennis_matches_cleaned.data`.

The `docs/classification.ipynb` notebook remains available for exploratory analysis and represents a separate data-analysis path from the normalization/enrichment pipeline described above.

### Enrich with Sackmann data

Clone the archival dataset, or provide an equivalent local directory containing
`atp_players.csv` and the required `atp_matches_YYYY.csv` files:

Adds player biographical data (height, dominant hand, age) and pre-match serve/return statistics (ace rate, break points saved, etc.), matched to Tennis-Data players by name. See [Download the enrichment data](#download-the-enrichment-data) for how to obtain `external/tennis_atp/`.
```bash
git clone --depth 1 \
  https://github.com/Aneeshers/tennis-sackmann-archive.git \
  external/tennis-sackmann-archive
```
This also writes `data/interim/sackmann_name_matching_review.csv`, a report of every player-name match decision (exact, fuzzy, or unresolved) — worth reviewing manually if match coverage looks low.

Run the enrichment step:

Computes pre-match player history, head-to-head, and market-odds features, builds a symmetric (player-order-independent) dataset, and runs a Random Forest baseline with RFECV feature selection:
```bash
uv run python scripts/data_cleaner/enrich_with_sackmann.py \
  --sackmann-dir external/tennis-sackmann-archive/atp
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

The Random Forest baseline and RFECV are optional diagnostics. For CatBoost,
building the same feature artifacts without that separate and expensive model
analysis is recommended:

```bash
uv run python scripts/data_cleaner/build_features.py \
  --input data/interim/tennis_matches_enriched.data \
  --rare-threshold 50 \
  --skip-model-analysis
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

For the CatBoost run on 2026-09-08, the available feature table contains 30,952
rows, corresponding to 15,476 real matches from 2020-01-06 to 2026-07-12
and their mirrored representations. The executed CatBoost workflow uses all
72 numeric and five categorical
features; it does not reuse feature selection fitted for a different model.

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

### CatBoost

CatBoost uses categorical columns directly, without one-hot encoding. The
current trainer fits one model with fixed parameters and reserves the newest
15% of unique dates for evaluation. Both orientations of each match remain
in the same partition. This run has no parameter search or cross-validation.

```bash
uv run python scripts/models/train_catboost.py
```

The fixed parameters are:

```text
depth = 8
iterations = 300
learning_rate = 0.03
l2_leaf_reg = 3
random_seed = 42
```

The completed run on **2026-09-08** used 72 numeric and five categorical features:

| Partition | Rows | First date | Last date |
| --- | ---: | --- | --- |
| Training | 26,472 | 2020-01-06 | 2025-08-07 |
| Holdout | 4,480 | 2025-08-08 | 2026-07-12 |

| Evaluation | Accuracy | ROC AUC | Log loss | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Naive majority baseline | 0.5000 | — | — | — |
| CatBoost chronological holdout | 0.6946 | 0.7559 | 0.5834 | 0.2003 |
| Normalized market baseline | 0.6915 | 0.7578 | 0.5818 | 0.1996 |

Training accuracy was 0.7062. Market probabilities are available for all holdout
rows. CatBoost has slightly higher accuracy in this run, while the market has
better ROC AUC, log loss and Brier score. This small accuracy difference does
not establish a reliable advantage over the market. These are raw row-level
predictions; the notebooks separately average the two orientations for
match-level error analysis (accuracy 0.6915).

The trainer saves `data/interim/catboost_model.cbm` and
`data/interim/catboost_metrics.json`. The JSON contains the algorithm, feature
schema, holdout accuracy and ROC AUC. The notebooks calculate the additional
metrics above from the saved model. Model and data artifacts remain local,
as configured by `.gitignore`; executed notebook outputs are committed.

This rerun reuses an existing holdout rather than a fresh independent test.
Older Decision Tree experiments above used a different dataset and are not a
controlled comparison with this run.

### Executed CatBoost analysis notebooks

The English and Italian notebooks inspect the dataset, reproduce the trainer's
chronological split, load the saved model, verify its accuracy and ROC AUC
against the metrics JSON, and show baselines, confusion matrix, classification
report, ROC curve, feature importance and match-level error diagnostics.

```bash
uv run python scripts/models/create_catboost_notebook.py --execute
uv run python scripts/models/create_catboost_notebook.py --language it --execute
```
Produces `data/interim/tennis_matches_features.data` and `data/interim/tennis_matches_features_metadata.json` (feature list, RFECV results, and baseline CV metrics).

They create `docs/catboost_training_analysis.ipynb` and
`docs/catboost_training_analysis_it.ipynb`. Run the trainer first; **Run All**
evaluates the saved model without repeating training. The separate
`docs/classification.ipynb` remains the exploratory analysis notebook.

### Use the trained CatBoost model

Score any CSV containing the exact engineered feature schema recorded in
`catboost_metrics.json`:

```bash
uv run python scripts/models/predict_catboost.py \
  --input data/interim/tennis_matches_features.data \
  --model data/interim/catboost_model.cbm \
  --metrics data/interim/catboost_metrics.json \
  --output data/interim/catboost_predictions.csv
```
Licensed under CC BY-NC-SA 4.0 — non-commercial use with attribution.

The output adds:

```text
player_a_win_probability
player_b_win_probability
predicted_player_a_win
predicted_winner_side
predicted_winner_name
```

The example above demonstrates batch scoring on the historical engineered
table. For a future match, the input must first be built from the same
point-in-time feature definitions; player names alone are not enough. After
symmetrization, legacy columns named `winner_*` and `loser_*` represent oriented
player A and player B. `player_a_win` and the prediction columns use that
orientation.

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

CatBoost outputs are:

```text
data/interim/catboost_model.cbm
data/interim/catboost_metrics.json
data/interim/catboost_predictions.csv
docs/catboost_training_analysis.ipynb
```

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
