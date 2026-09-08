# CatBoost for Tennis Match Prediction

## Goal

The model solves a **binary classification** problem:

- `player_a_win = 1`: player A wins;
- `player_a_win = 0`: player B wins.

The main file is:

```text
scripts/models/train_catboost.py
```

The implementation is intentionally kept simple so that every step can be explained during an oral presentation.

## Program flow

Training follows six steps:

```text
Dataset
   ↓
Features X and target y
   ↓
Chronological Train/Test split
   ↓
CatBoostClassifier
   ↓
model.fit()
   ↓
Prediction + Accuracy + ROC AUC
   ↓
Save model
```

## 1. Loading the data

The final feature dataset is read from:

```text
data/interim/tennis_matches_features.data
```

The metadata file contains the numeric and categorical feature lists:

```text
data/interim/tennis_matches_features_metadata.json
```

The feature matrix is called `X`, while the match outcome is called `y`:

```python
X = data[feature_names]
y = data["player_a_win"]
```

## 2. Categorical features

CatBoost can directly use categorical features. In this project, examples include surface, court, and tournament information.

Categorical columns are converted to strings and missing values are replaced with:

```text
__MISSING__
```

Therefore, manual One-Hot Encoding is not required before training CatBoost.

## 3. Chronological split

For sports prediction, randomly mixing past and future matches is not appropriate.

The project therefore uses:

- the oldest 85% of unique dates for training;
- the newest 15% of unique dates for testing.

```text
past                                         future
|---------------- TRAIN ----------------|------ TEST ------|
```

The test set therefore simulates the real use case: the model learns from matches that have already happened and predicts later matches.

## 4. CatBoost model

The classifier uses a small fixed set of hyperparameters:

```python
model = CatBoostClassifier(
    iterations=300,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=3,
    loss_function="Logloss",
    cat_features=categorical_features,
    random_seed=42,
    verbose=False,
)
```

### `iterations = 300`

This is the number of trees built by boosting.

CatBoost does not rely on a single Decision Tree. It builds multiple trees sequentially, and every new tree tries to correct errors made by the current ensemble.

A simplified representation is:

\[
F_M(x) = F_0(x) + \eta h_1(x) + \eta h_2(x) + \dots + \eta h_M(x)
\]

### `depth = 8`

This controls tree depth. Deeper trees can represent more complex relationships between features.

### `learning_rate = 0.03`

This controls the contribution of each new tree to the current model.

A small learning rate makes the correction more gradual.

### `l2_leaf_reg = 3`

This applies L2 regularization to leaf values and helps reduce overfitting.

### `loss_function = "Logloss"`

For binary classification, CatBoost minimizes Log Loss.

If `p` is the predicted probability that player A wins, the loss strongly penalizes confident but incorrect predictions.

## 5. Training

Training is performed with:

```python
model.fit(X_train, y_train)
```

During this operation CatBoost sequentially builds the trees that form the final classifier.

## 6. Prediction

The predicted class is obtained with:

```python
predictions = model.predict(X_test)
```

The probability that player A wins is obtained with:

```python
probabilities = model.predict_proba(X_test)[:, 1]
```

Example:

```text
P(Player A wins) = 0.72
```

means that the model assigns player A a 72% probability of winning.

## Metrics

### Accuracy

\[
Accuracy = \frac{correct\ predictions}{total\ predictions}
\]

It measures the percentage of matches classified correctly.

### ROC AUC

ROC AUC measures how well the model ranks positive and negative examples using predicted probabilities.

Simple interpretation:

- `0.5`: similar to random ranking;
- `1.0`: perfect separation between the two classes.

## Outputs

Training creates:

```text
data/interim/catboost_model.cbm
data/interim/catboost_metrics.json
```

The first file contains the trained model.

The second contains only the information needed to interpret and reuse it:

- algorithm;
- numeric features;
- categorical features;
- test Accuracy;
- test ROC AUC.

## Executed results — 2026-09-08

The local feature table contains 30,952 rows (15,476 matches), with 72 numeric
and five categorical features. Training uses 26,472 rows dated 2020-01-06 to
2025-08-07; testing uses 4,480 rows dated 2025-08-08 to 2026-07-12.

| Evaluation | Accuracy | ROC AUC | Log loss | Brier score |
| --- | ---: | ---: | ---: | ---: |
| CatBoost | 0.6946 | 0.7559 | 0.5834 | 0.2003 |
| Normalized market baseline | 0.6915 | 0.7578 | 0.5818 | 0.1996 |
| Majority baseline | 0.5000 | — | — | — |

Training accuracy is 0.7062. This run uses fixed parameters, with no tuning or
cross-validation. It reuses an existing holdout, not a fresh independent test.
The small accuracy improvement over the market does not establish a reliable
advantage; the market performs better on the other three metrics.

Both executed notebooks load the saved model and assert that accuracy and ROC
AUC match the training JSON. Additional metrics are computed in the notebooks.
Main metrics use raw predictions for both orientations; the separate match-level
diagnostic averages orientations and obtains 0.6915 accuracy.

```bash
uv run python scripts/models/create_catboost_notebook.py --execute
uv run python scripts/models/create_catboost_notebook.py --language it --execute
```

Model/data artifacts stay local under `data/interim/`; executed outputs are in
[the English notebook](catboost_training_analysis.ipynb) and
[the Italian notebook](catboost_training_analysis_it.ipynb).

## Running the model

After the feature dataset has been generated:

```bash
uv run python scripts/models/train_catboost.py
```

Run the tests with:

```bash
uv run pytest
```

## Short oral explanation

A concise presentation can be:

> CatBoost is a Gradient Boosting algorithm based on Decision Trees. Instead of using a single tree, it builds several trees sequentially, and every new tree tries to correct the errors of the current ensemble. In this project I use it for binary classification, where the target indicates whether player A wins the match. CatBoost is also useful because it can directly process categorical features. To simulate real prediction, I use a chronological split: the model is trained on older matches and evaluated on more recent ones. The main evaluation metrics are Accuracy and ROC AUC.

## Essential code to remember

The conceptual core of the implementation is:

```python
X = data[feature_names]
y = data["player_a_win"]

model = CatBoostClassifier(
    iterations=300,
    depth=8,
    learning_rate=0.03,
    cat_features=categorical_features,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

Everything else in the training file only loads the data, preserves temporal ordering, and saves the trained model.
