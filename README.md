# Tennis match prediction

Predict ATP tennis match winners from rankings, historical results, market odds,
and player biography. The current CatBoost workflow rebuilds local inputs,
reserves the newest 15% of unique dates for evaluation, and verifies the same
saved model in English and Italian analysis notebooks.

## Local data sources

- Tennis-Data yearly workbooks in `data/raw/2015.xlsx` through `2026.xlsx` contain
  match results, rankings, and decimal odds. The local 2026 workbook ends on
  2026-07-19. Raw score columns are discarded before feature engineering.
- Sackmann player biography comes from the local
  [June 2026 archival mirror](https://github.com/Aneeshers/tennis-sackmann-archive),
  in `external/tennis-sackmann-archive/atp/atp_players.csv`. The original
  [Jeff Sackmann repository](https://github.com/JeffSackmann/tennis_atp) is the
  upstream attribution; its availability has not been rechecked for this run.
- Source files and derived datasets remain local and are excluded from Git.
  This realignment uses existing archives and does not refresh downloads.
  Earlier artifacts are preserved under `data/archive/before_realignment_*/`.

The Sackmann data is attributed to Jeff Sackmann and licensed CC BY-NC-SA 4.0.
Commercial use is not permitted by that license. Check the source terms before
redistributing data or using derived artifacts.

## Reproduce the current run

Use Python 3.12 or newer and `uv`. Run these commands from the project root,
using the existing local source files. Feature and model outputs are replaced,
so copy any previous experiment you need to retain before rerunning.

```bash
uv sync --all-groups
uv run pytest -q

# Merge existing yearly workbooks. This legacy script uses its own working directory.
uv run --directory scripts/data_downloader python merge_tennis_excel.py

# Keep completed matches, normalize fields, and remove post-match score columns.
uv run python scripts/data_cleaner/normalize_data.py

# Add matched player IDs and snapshot biography only.
uv run python scripts/data_cleaner/enrich_with_sackmann.py \
  --sackmann-dir external/tennis-sackmann-archive/atp

# Build strict prior-date histories and both player orientations.
uv run python scripts/data_cleaner/build_features.py \
  --input data/interim/tennis_matches_enriched.data \
  --skip-model-analysis

# Train one fixed-parameter model and save metrics and provenance.
uv run python scripts/models/train_catboost.py

# Execute both analyses and refresh the measured-results section below.
uv run python scripts/models/create_catboost_notebook.py --execute
uv run python scripts/models/create_catboost_notebook.py --language it --execute

# Score the complete historical table.
uv run python scripts/models/predict_catboost.py \
  --input data/interim/tennis_matches_features.data \
  --output data/interim/catboost_predictions.csv
```

Notebook execution updates only the marked current-results section of this
README, using its verified evaluation summary. Generation without `--execute`
creates unexecuted cells and leaves results unchanged. After changing data or
features, rebuild and retrain before executing the notebooks. Old model,
metadata, and data files cannot be mixed with the corrected artifacts.

## Data transformations and assumptions

`normalize_data.py` removes exact raw duplicates, keeps only `Completed`
matches, removes missing player names and invalid dates, and converts invalid
ranking, points, and odds values to missing. Retirements and walkovers are
excluded. This is basic normalization, not certification of the external source.

`enrich_with_sackmann.py` maps abbreviated Tennis-Data names to Sackmann player
IDs and adds height, hand, and age at the match date. It writes all accepted and
rejected matching decisions to `data/interim/sackmann_name_matching_review.csv`.
Unresolved players remain in the table with missing enrichment values.
Biography is from a snapshot rather than a complete historical record.

Sackmann match service statistics are excluded from this workflow because
`tourney_date` is a tournament start date, not verified availability of each
match result. Historical helper functions remain in the script for separate
analysis, but `main()` does not use them. Its `--start-year` and `--end-year`
arguments remain accepted for compatibility and do not select biography rows.

`build_features.py` calculates player experience, win rate, last-five form,
surface history, rest days, and head-to-head records using only **strictly earlier
dates**. No result from the current day contributes to that day's features.
Within a previous shared date, last-five windows use deterministic match-ID
order because match times are unavailable. Histories may include earlier dates
of the test period, as in daily sequential prediction. The trained model itself
remains fixed.

Tournament and location are raw categorical inputs. Their frequencies are not
fitted on the entire dataset. The legacy `--rare-threshold` argument remains
accepted but no longer changes category values. Names, IDs, dates, and the target
are not model features. `match_id_internal` is retained as a diagnostic key;
it is deterministic for the same ordered input, not a global source identifier.

Each real match has two oriented rows with complementary `player_a_win` labels.
After swapping, legacy `winner_*` columns mean player A and `loser_*` columns
mean player B, including names, rankings, biography, historical features, and
odds. Difference features change sign. This orientation does not require knowing
the eventual winner when scoring an already prepared future-match row.

The market baseline normalizes both reciprocal average odds:

```text
P(A) = (1 / odds_A) / [(1 / odds_A) + (1 / odds_B)]
P(B) = 1 - P(A)
```

Both odds must exceed 1. These normalized probabilities and their difference
are diagnostic columns, not additional model inputs. The model retains the
existing unnormalized implied-probability features and the odds-availability
flag. Odds observation times are not verified, so this run does not establish
that prices were obtainable at a particular betting decision time.

## Model and artifact contract

The fixed CatBoost parameters are 300 iterations, depth 8, learning rate 0.03,
L2 leaf regularization 3, Logloss, and random seed 42. There is no parameter
search, cross-validation, early stopping, or final refit on the holdout.
CatBoost handles categorical strings directly. Invalid numeric values become
missing and missing categories use `__MISSING__`.

The trainer rejects malformed schemas, missing declared features, invalid
dates, fractional targets, identifiers included as inputs, and partitions
without both classes. The newest 15% of **distinct dates** form the holdout;
the row fraction can differ. Both orientations of a match stay together.

The main artifacts are:

| Artifact | Purpose |
| --- | --- |
| `data/interim/tennis_matches_features.data` | Oriented features and diagnostic columns |
| `data/interim/tennis_matches_features_metadata.json` | Ordered feature schema and transformation policies |
| `data/interim/catboost_model.cbm` | Fitted CatBoost model |
| `data/interim/catboost_metrics.json` | Parameters, versions, hashes, split, counts, and four training/holdout metrics |
| `data/interim/catboost_analysis_summary.json` | Verified notebook evaluation and market comparison |
| `docs/catboost_training_analysis.ipynb` | Executed English analysis |
| `docs/catboost_training_analysis_it.ipynb` | Executed Italian analysis |

Prediction verifies the model hash, ordered feature names, categorical positions,
and binary classes. Notebook evaluation also verifies the original dataset and
metadata hashes, reconstructs the saved cutoff, and checks all four metrics.
Arbitrary prediction inputs need the saved feature schema, not the training-data
hash. Existing artifacts without this provenance contract must be regenerated.

## Current verified results

<!-- catboost-results:start -->

Training run UTC: `2026-11-08T15:16:15.103007+00:00`.
Dataset: **56,450 oriented rows**, **28,225 real matches**, 83 columns, from 2015-01-05 to 2026-07-19.
The model uses **43 numeric** and **5 categorical** features.

| Partition | Rows | First date | Last date |
| --- | ---: | --- | --- |
| Training | 48,158 | 2015-01-05 | 2024-11-14 |
| Holdout | 8,292 | 2024-11-15 | 2026-07-19 |

| Evaluation | Accuracy | ROC AUC | Log loss | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Training | 0.699863 | 0.779656 | 0.560833 | 0.191089 |
| Naive holdout baseline | 0.500000 | 0.500000 | 18.021827 | 0.500000 |
| CatBoost full holdout | 0.686204 | 0.750947 | 0.588525 | 0.202385 |
| CatBoost market subset | 0.686204 | 0.750947 | 0.588525 | 0.202385 |
| Market on identical rows | 0.687651 | 0.750927 | 0.588263 | 0.202278 |

Market coverage: **100.0%**, 8,292 oriented holdout rows.
Match-level accuracy after averaging orientations: **0.688133** on 4,146 matches. Exact ties: 0; ties receive half credit.

Compare CatBoost and the market on the identical-row subset. These historical results do not establish betting profit or an advantage on a fresh independent test.

<!-- catboost-results:end -->

## Interpret the evaluation

Accuracy measures correct class decisions at probability 0.5; ROC AUC measures
ranking. Log loss and Brier score assess probabilities and are better when
lower. Full-holdout metrics count the original oriented rows. Diagnostic
match-level accuracy averages the probability assigned to the actual winner
across both orientations before counting each match once. Exact ties receive
half credit and no predicted winner.

The market comparison uses exactly the same market-covered rows for CatBoost
and the market and reports coverage. Missing market coverage is not filled with
fabricated odds. AUC is undefined on one-class subsets; metrics are undefined
when a comparison subset is empty. Notebook JSON summaries use `null` there.
Feature importance and differences between correct and wrong predictions do not
establish causality. No accuracy or profitability target is promised.

This historical holdout has already been inspected in previous experiments.
A corrected rerun is not a fresh independent test. Future tuning must use
training-only temporal folds and a subsequently untouched evaluation period.

## Use predictions and other experiments

`predict_catboost.py` appends `player_a_win_probability`,
`player_b_win_probability`, `predicted_player_a_win`, and
`predicted_winner_side`. If both name columns are present, it also appends
`predicted_winner_name`. Use `--threshold` to change the decision threshold
from 0.5 to a value strictly between 0 and 1.

Whole-table historical scoring includes training rows. It is not an independent
accuracy measurement. Future-match scoring requires the same point-in-time
feature definitions; names alone are insufficient. This correction does not
add an input builder for future fixtures.

Random Forest/RFECV diagnostics in the feature builder remain optional and are
skipped for this CatBoost workflow. The Decision Tree trainer remains available:

```bash
uv run python scripts/models/train_decision_tree.py
```

Its experiments and `docs/classification.ipynb` are separate exploratory paths.
They were not rerun in this realignment and are not controlled comparisons with
the current CatBoost result. Previously published numerical results from other
schemas are not treated as current measurements.
