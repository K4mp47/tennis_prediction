"""Create and optionally execute the teacher-style CatBoost notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "docs" / "catboost_training_analysis.ipynb",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute every cell and embed the outputs.",
    )
    return parser.parse_args()


def build_notebook(metrics: dict[str, object]) -> nbformat.NotebookNode:
    best_cv_accuracy = metrics["best_cv_accuracy"]
    holdout_accuracy = metrics["holdout_accuracy"]

    cells = [
        new_markdown_cell(
            """# Tennis Match Winner Prediction with CatBoost

## Goal

The task is a **binary classification problem**:

- class `1`: player A wins;
- class `0`: player B wins.

The objective is to train a CatBoost classifier and evaluate whether it can
generalize to matches played after those used for training."""
        ),
        new_markdown_cell(
            """## Why CatBoost?

CatBoost is an **ensemble method based on boosting**. As discussed for
AdaBoost, boosting combines several decision trees sequentially: every new
tree tries to reduce the errors made by the current ensemble.

The final prediction is obtained by combining the contribution of all trees.
CatBoost is useful here because the dataset contains both numerical and
categorical attributes, such as surface, court and tournament."""
        ),
        new_markdown_cell("## Setup"),
        new_code_cell(
            """from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from IPython.display import display
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

MODELS_DIR = PROJECT_ROOT / "scripts" / "models"
sys.path.insert(0, str(MODELS_DIR))

from train_catboost import prepare_catboost_features
from training_utils import chronological_holdout_split

FEATURES_PATH = PROJECT_ROOT / "data/interim/tennis_matches_features.data"
METADATA_PATH = PROJECT_ROOT / "data/interim/tennis_matches_features_metadata.json"
METRICS_PATH = PROJECT_ROOT / "data/interim/catboost_metrics.json"

plt.style.use("seaborn-v0_8-whitegrid")"""
        ),
        new_markdown_cell("## 1. Load the dataset"),
        new_code_cell(
            """dataset = pd.read_csv(FEATURES_PATH)
dataset["date"] = pd.to_datetime(dataset["date"], errors="raise")

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
training_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

columns_to_show = [
    "date",
    "surface",
    "winner_rank",
    "loser_rank",
    "rank_diff",
    "win_rate_diff",
    "implied_prob_diff_avg",
    "player_a_win",
]

print("Dataset shape:", dataset.shape)
display(dataset[columns_to_show].head())
display(dataset["player_a_win"].value_counts().sort_index().rename("instances").to_frame())"""
        ),
        new_markdown_cell(
            """The dataset contains two rows for every real match: one for each
player orientation. This produces a balanced target and avoids making the
model depend on the original winner/loser column order.

Only pre-match information is used: rankings, previous results, surface form,
head-to-head history, player statistics and market probabilities."""
        ),
        new_markdown_cell("## 2. Prepare features and create the train/test split"),
        new_markdown_cell(
            """A random split is not appropriate for this problem because it could
train the model on future matches and test it on older matches.

The most recent 15% of unique dates is therefore used as the **test set**. The
remaining older matches form the **training set**. Both orientations of a
match have the same date and remain in the same partition."""
        ),
        new_code_cell(
            """numeric_features = tuple(metadata["numeric_features"])
categorical_features = tuple(metadata["categorical_features"])
feature_names = numeric_features + categorical_features

X = prepare_catboost_features(
    dataset.loc[:, feature_names],
    numeric_features,
    categorical_features,
)
y = dataset["player_a_win"].astype(int)

train_index, test_index = chronological_holdout_split(dataset["date"], 0.15)
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

split = pd.DataFrame(
    {
        "rows": [len(X_train), len(X_test)],
        "first date": [
            dataset.iloc[train_index]["date"].min().date(),
            dataset.iloc[test_index]["date"].min().date(),
        ],
        "last date": [
            dataset.iloc[train_index]["date"].max().date(),
            dataset.iloc[test_index]["date"].max().date(),
        ],
    },
    index=["training", "test"],
)
display(split)"""
        ),
        new_markdown_cell("## 3. Naive baseline"),
        new_markdown_cell(
            """A classifier must be compared with a simple baseline. Following the
course notes, the naive classifier always predicts the majority class.

Because the symmetric dataset is balanced, the expected baseline accuracy is
approximately 50%."""
        ),
        new_code_cell(
            """baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))

print(f"Naive baseline accuracy: {baseline_accuracy:.4f}")"""
        ),
        new_markdown_cell("## 4. Hyper-parameter tuning"),
        new_markdown_cell(
            """The parameters were selected with `GridSearchCV`, as in the course
example. Every parameter combination was evaluated using five chronological
cross-validation folds. The final test set was not used during this selection.

The grid controls:

- `iterations`: number of trees in the ensemble;
- `depth`: maximum depth of every tree;
- `learning_rate`: contribution of each new tree;
- `l2_leaf_reg`: regularization used to limit over-fitting."""
        ),
        new_code_cell(
            """print("Parameter grid:")
display(pd.Series(training_metrics["parameter_grid"], name="tested values").to_frame())

print(f"Best cross-validation accuracy: {training_metrics['best_cv_accuracy']:.4f}")
print("Best parameters:", training_metrics["best_params"])"""
        ),
        new_markdown_cell("## 5. Train the final model"),
        new_code_cell(
            """model = CatBoostClassifier(
    **training_metrics["best_params"],
    loss_function="Logloss",
    eval_metric="Accuracy",
    cat_features=categorical_features,
    random_seed=42,
    allow_writing_files=False,
    verbose=False,
    thread_count=1,
)

model.fit(X_train, y_train)
print("Training completed.")"""
        ),
        new_markdown_cell("## 6. Evaluate the classifier"),
        new_code_cell(
            """train_prediction = model.predict(X_train).astype(int).ravel()
test_prediction = model.predict(X_test).astype(int).ravel()
test_probability = model.predict_proba(X_test)[:, 1]

train_accuracy = accuracy_score(y_train, train_prediction)
test_accuracy = accuracy_score(y_test, test_prediction)
test_auc = roc_auc_score(y_test, test_probability)

results = pd.Series(
    {
        "Naive baseline accuracy": baseline_accuracy,
        "Training accuracy": train_accuracy,
        "Cross-validation accuracy": training_metrics["best_cv_accuracy"],
        "Test accuracy": test_accuracy,
        "Test ROC AUC": test_auc,
    },
    name="score",
)
display(results.to_frame())"""
        ),
        new_markdown_cell("### Confusion matrix"),
        new_code_cell(
            """ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_prediction,
    display_labels=["Player B wins", "Player A wins"],
    cmap="Blues",
)
plt.title("CatBoost confusion matrix - test set")
plt.grid(False)
plt.show()"""
        ),
        new_markdown_cell("### Precision, recall and F1-score"),
        new_code_cell(
            """print(
    classification_report(
        y_test,
        test_prediction,
        target_names=["Player B wins", "Player A wins"],
        digits=4,
    )
)"""
        ),
        new_markdown_cell("### ROC curve"),
        new_code_cell(
            """RocCurveDisplay.from_predictions(
    y_test,
    test_probability,
    name=f"CatBoost (AUC = {test_auc:.4f})",
)
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
plt.title("ROC curve - test set")
plt.legend()
plt.show()"""
        ),
        new_markdown_cell("## 7. Feature importance"),
        new_markdown_cell(
            """Tree-based models expose a feature-importance score. The score
indicates how much each feature contributes to the decisions of the ensemble.
As discussed in the course, importance is useful for interpretation, but it
does not prove that a feature causes the prediction."""
        ),
        new_code_cell(
            """feature_importance = (
    pd.Series(
        np.asarray(model.feature_importances_, dtype=float),
        index=feature_names,
        name="importance",
    )
    .sort_values(ascending=False)
    .head(10)
)

display(feature_importance.to_frame())

ax = feature_importance.sort_values().plot(
    kind="barh",
    figsize=(8, 4.5),
    color="#2a9d8f",
)
ax.set_title("Ten most important features")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.show()"""
        ),
        new_markdown_cell(
            f"""## Conclusion

- The naive baseline obtains about **50% accuracy** because the classes are
  balanced.
- CatBoost obtains **{best_cv_accuracy:.4f} cross-validation accuracy** and
  **{holdout_accuracy:.4f} test accuracy**, so the result is stable on future
  matches.
- The model improves clearly over the naive baseline.
- Market-implied probabilities are the most important features. Therefore,
  part of the predictive power comes from information already contained in
  betting odds.
- The difference between training and test accuracy should be monitored: a
  much larger training score would indicate over-fitting.

The final test set is used only once for evaluation. Any future change to the
features or CatBoost parameters must be selected using the training
cross-validation folds, not this test result."""
        ),
    ]

    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    metrics_path = project_root / "data/interim/catboost_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    notebook = build_notebook(metrics)

    if args.execute:
        client = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(project_root)}},
        )
        client.execute()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"Notebook: {args.output}")
    print(f"Executed: {args.execute}")


if __name__ == "__main__":
    main()
