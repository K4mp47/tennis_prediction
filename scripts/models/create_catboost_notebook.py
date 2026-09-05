"""Create and optionally execute the teacher-style CatBoost notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def translate_to_italian(notebook: nbformat.NotebookNode) -> None:
    """Translate reader-facing notebook prose while preserving executable code."""

    italian_markdown = [
        """# Predizione del vincitore di una partita di tennis con CatBoost

## Obiettivo

Il problema e' una **classificazione binaria**:

- classe `1`: vince il giocatore A;
- classe `0`: vince il giocatore B.

L'obiettivo e' addestrare un classificatore CatBoost e verificare se riesce a
generalizzare sulle partite disputate dopo quelle usate per l'addestramento.""",
        """## Perche' CatBoost?

CatBoost e' un **metodo ensemble basato sul boosting**. Come in AdaBoost, il
boosting combina in sequenza diversi alberi decisionali: ogni nuovo albero
cerca di ridurre gli errori dell'ensemble corrente.

La previsione finale combina il contributo di tutti gli alberi. CatBoost e'
adatto perche' il dataset contiene attributi numerici e categorici, come
superficie, campo e torneo.""",
        "## Configurazione",
        "## 1. Caricamento del dataset",
        """Il dataset contiene due righe per ogni partita reale, una per ciascun
orientamento dei giocatori. In questo modo la variabile obiettivo e' bilanciata
e il modello non dipende dall'ordine originale vincitore/sconfitto.

Sono usate solo informazioni disponibili prima della partita: ranking,
risultati precedenti, rendimento sulla superficie, scontri diretti,
statistiche dei giocatori e probabilita' di mercato.""",
        "## 2. Preparazione delle feature e divisione train/test",
        """Una divisione casuale non e' appropriata: potrebbe addestrare il modello
su partite future e valutarlo su partite precedenti.

Il 15% piu' recente delle date uniche e' quindi riservato come **test set**.
Le partite precedenti costituiscono il **training set**. I due orientamenti
della stessa partita condividono la data e restano nella stessa partizione.""",
        "## 3. Baseline ingenua",
        """Il classificatore va confrontato con una baseline semplice, che predice
sempre la classe maggioritaria.

Poiche' il dataset simmetrico e' bilanciato, l'accuratezza attesa della baseline
e' circa il 50%.""",
        "## 4. Ricerca degli iperparametri",
        """I parametri sono stati scelti con `RandomizedSearchCV`. Dodici
combinazioni campionate sono state valutate con cinque fold cronologici di
cross-validation. Il test set finale non e' stato usato durante la selezione.
L'ultimo fold CV ha poi scelto il numero di alberi con early stopping.

La griglia controlla:

- `iterations`: numero di alberi nell'ensemble;
- `depth`: profondita' massima di ogni albero;
- `learning_rate`: contributo di ogni nuovo albero;
- `l2_leaf_reg`: regolarizzazione per limitare l'overfitting;
- `random_strength`: casualita' applicata alla scelta delle divisioni;
- `bagging_temperature`: intensita' del bootstrap bayesiano.""",
        "## 5. Addestramento del modello finale",
        "## 6. Valutazione del classificatore",
        "### Matrice di confusione",
        "### Precisione, richiamo e F1-score",
        "### Curva ROC",
        "## 7. Importanza delle feature",
        """I modelli basati su alberi espongono un punteggio di importanza delle
feature, che indica quanto ciascuna contribuisce alle decisioni dell'ensemble.
L'importanza aiuta a interpretare il modello, ma non dimostra un rapporto
causale.""",
        "## 8. Analisi delle partite piu' facili e piu' difficili",
        """Le righe sono abbinate tramite `match_id_internal`, quindi ogni partita
reale viene contata una sola volta. La confidenza assegnata al vincitore reale
e' mediata tra i due orientamenti. Le previsioni piu' corrette assegnano la
probabilita' maggiore al vincitore reale; quelle piu' errate la minore.""",
        "### Previsioni corrette con maggiore confidenza",
        "### Previsioni errate con maggiore confidenza",
        "### Distribuzioni numeriche: previsioni corrette ed errate",
        """Per ogni feature numerica, la tabella confronta mediane, tassi di valori
mancanti e differenza tra le medie espressa in deviazioni standard complessive.
Le differenze assolute maggiori mostrano dove gli errori occupano una parte
diversa della distribuzione, senza implicare causalita'.""",
        "### Distribuzioni delle feature categoriche",
        "### Quali peculiarita' emergono?",
    ]
    markdown_cells = [
        cell for cell in notebook.cells if cell.cell_type == "markdown"
    ]
    if len(markdown_cells) != len(italian_markdown) + 1:
        raise ValueError("Unexpected notebook structure for Italian translation.")
    for cell, source in zip(
        markdown_cells[:-1], italian_markdown, strict=True
    ):
        cell.source = source

    conclusion = markdown_cells[-1].source
    conclusion_replacements = {
        "## Conclusion": "## Conclusioni",
        "The naive baseline obtains about": "La baseline ingenua ottiene circa il",
        "% accuracy**": "% di accuratezza**",
        "because the classes are": "perche' le classi sono",
        "balanced.": "bilanciate.",
        "CatBoost obtains": "CatBoost ottiene",
        "in cross-validation and": "in cross-validation e",
        "test accuracy**, so the result is stable on future": "di accuratezza sul test**, quindi il risultato e' stabile sulle",
        "matches.": "partite future.",
        "The model improves clearly over the naive baseline.": "Il modello migliora chiaramente rispetto alla baseline ingenua.",
        "Market-implied probabilities are the most important features. Therefore,": "Le probabilita' implicite di mercato sono le feature piu' importanti. Quindi",
        "part of the predictive power comes from information already contained in": "parte del potere predittivo deriva da informazioni gia' contenute nelle",
        "betting odds.": "quote delle scommesse.",
        "The most confident mistakes are mainly upsets in which rankings, Elo, form": "Gli errori piu' sicuri sono soprattutto sorprese in cui ranking, Elo, forma",
        "and betting odds all favored the eventual loser. There is no similarly large": "e quote favorivano tutti il giocatore poi sconfitto. Non emerge un'anomalia",
        "missing-value anomaly separating correct and wrong predictions.": "altrettanto grande nei valori mancanti tra previsioni corrette ed errate.",
        "The difference between training and test accuracy should be monitored: a": "La differenza tra accuratezza di training e test va monitorata: un punteggio",
        "much larger training score would indicate over-fitting.": "di training molto maggiore indicherebbe overfitting.",
        "The final test set is used only once for evaluation. Any future change to the": "Il test set finale e' usato una sola volta per la valutazione. Ogni modifica futura a",
        "features or CatBoost parameters must be selected using the training": "feature o parametri CatBoost deve essere scelta usando i fold di",
        "cross-validation folds, not this test result.": "cross-validation del training, non questo risultato di test.",
    }
    for english, italian in conclusion_replacements.items():
        conclusion = conclusion.replace(english, italian)
    markdown_cells[-1].source = conclusion

    code_replacements = {
        "Dataset shape:": "Dimensioni del dataset:",
        'rename("instances")': 'rename("istanze")',
        '"first date"': '"prima data"',
        '"last date"': '"ultima data"',
        'index=["training", "test"]': 'index=["addestramento", "test"]',
        "Naive baseline accuracy:": "Accuratezza della baseline ingenua:",
        "Parameter grid:": "Griglia dei parametri:",
        'name="tested values"': 'name="valori provati"',
        "Best cross-validation": "Migliore risultato di cross-validation per",
        "Best parameters:": "Parametri migliori:",
        "Training completed.": "Addestramento completato.",
        '"Naive baseline accuracy"': '"Accuratezza baseline ingenua"',
        '"Training accuracy"': '"Accuratezza training"',
        '"Test accuracy"': '"Accuratezza test"',
        '"Test ROC AUC"': '"ROC AUC test"',
        'display_labels=["Player B wins", "Player A wins"]': 'display_labels=["Vince il giocatore B", "Vince il giocatore A"]',
        "CatBoost confusion matrix - test set": "Matrice di confusione CatBoost - test set",
        'target_names=["Player B wins", "Player A wins"]': 'target_names=["Vince il giocatore B", "Vince il giocatore A"]',
        "Random classifier": "Classificatore casuale",
        "ROC curve - test set": "Curva ROC - test set",
        "Ten most important features": "Le dieci feature piu' importanti",
        'ax.set_xlabel("Importance")': 'ax.set_xlabel("Importanza")',
        'ax.set_ylabel("Feature")': 'ax.set_ylabel("Feature")',
        "Each holdout match must have two orientations.": "Ogni partita di holdout deve avere due orientamenti.",
        "Real holdout matches:": "Partite reali nell'holdout:",
        "Match-level accuracy:": "Accuratezza per partita:",
        'axis.set_xlabel("Prediction correct")': 'axis.set_xlabel("Previsione corretta")',
        'axis.set_ylabel("Value")': 'axis.set_ylabel("Valore")',
        "Largest numeric distribution shifts": "Maggiori differenze nelle distribuzioni numeriche",
        "Importance assigned to six odds features:": "Importanza assegnata alle sei feature delle quote:",
        "Wrong predictions where the actual winner was the market underdog:": "Errori in cui il vincitore reale era sfavorito dal mercato:",
        "Largest correct/wrong missing-rate gap:": "Massima differenza nel tasso di valori mancanti:",
        "Interpretation: errors are mostly genuine upsets. Ranking, Elo, form and ": "Interpretazione: gli errori sono soprattutto vere sorprese. Ranking, Elo, forma e ",
        "market-probability differences reverse direction in the wrong cohort. ": "differenze nelle probabilita' di mercato cambiano segno nel gruppo degli errori. ",
        "No comparably large missing-data anomaly is visible. Category-level ": "Non emerge un'anomalia altrettanto grande nei dati mancanti. Le differenze tra ",
        "differences should be treated cautiously because many groups are small.": "categorie vanno interpretate con cautela perche' molti gruppi sono piccoli.",
    }
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for english, italian in code_replacements.items():
                cell.source = cell.source.replace(english, italian)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--language", choices=("en", "it"), default="en")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute every cell and embed the outputs.",
    )
    return parser.parse_args()


def build_notebook(
    metrics: dict[str, object], language: str = "en"
) -> nbformat.NotebookNode:
    best_cv_accuracy = metrics.get("best_cv_accuracy")
    if best_cv_accuracy is None:
        cv_result = (
            f"{metrics.get('selection_metric', 'validation score')} "
            f"{metrics['best_cv_score']:.4f}"
        )
    else:
        cv_result = f"accuracy {best_cv_accuracy:.4f}"
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

from train_catboost import (  # noqa: E402
    prepare_catboost_features,
    symmetrize_pair_probabilities,
)
from training_utils import chronological_holdout_split  # noqa: E402

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
        """The parameters were selected with `RandomizedSearchCV`. Twelve sampled
combinations were evaluated using five chronological cross-validation folds.
The final test set was not used during this selection. The last CV fold then
selected the number of trees with early stopping.

The grid controls:

- `iterations`: number of trees in the ensemble;
- `depth`: maximum depth of every tree;
- `learning_rate`: contribution of each new tree;
- `l2_leaf_reg`: regularization used to limit over-fitting;
- `random_strength`: randomness applied when choosing tree splits;
- `bagging_temperature`: intensity of Bayesian bootstrap sampling."""
        ),
        new_code_cell(
            """print("Parameter grid:")
display(pd.Series(training_metrics["parameter_grid"], name="tested values").to_frame())

cv_metric = training_metrics.get("selection_metric", "accuracy")
cv_score = training_metrics.get("best_cv_accuracy")
if cv_score is None:
    cv_score = training_metrics["best_cv_score"]
print(f"Best cross-validation {cv_metric}: {cv_score:.4f}")
print("Best parameters:", training_metrics["best_params"])"""
        ),
        new_markdown_cell("## 5. Train the final model"),
        new_code_cell(
            """model = CatBoostClassifier(
    **training_metrics["best_params"],
    loss_function="Logloss",
    eval_metric="Logloss",
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
raw_test_probability = model.predict_proba(X_test)[:, 1]
test_probability, test_actual_winner_probability = symmetrize_pair_probabilities(
    dataset.iloc[test_index]["match_id_internal"],
    y_test,
    raw_test_probability,
)
test_prediction = (test_probability >= 0.5).astype(int)

train_accuracy = accuracy_score(y_train, train_prediction)
test_accuracy = accuracy_score(y_test, test_prediction)
test_auc = roc_auc_score(y_test, test_probability)
cv_metric = training_metrics.get("selection_metric", "accuracy")
cv_score = training_metrics.get("best_cv_accuracy")
if cv_score is None:
    cv_score = training_metrics["best_cv_score"]

results = pd.Series(
    {
        "Naive baseline accuracy": baseline_accuracy,
        "Training accuracy": train_accuracy,
        f"Cross-validation {cv_metric}": cv_score,
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
)

display(feature_importance.head(10).to_frame())

ax = feature_importance.head(10).sort_values().plot(
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
        new_markdown_cell("## 8. Inspect the easiest and hardest matches"),
        new_markdown_cell(
            """Rows are paired by `match_id_internal`, so every real match is
counted once. Confidence in the real winner is averaged across the two player
orientations. The most correct matches have the highest probability assigned
to the actual winner; the most wrong matches have the lowest."""
        ),
        new_code_cell(
            """holdout = dataset.iloc[test_index].copy()
holdout["actual_winner_probability"] = test_actual_winner_probability

pair_sizes = holdout.groupby("match_id_internal").size()
assert pair_sizes.eq(2).all(), "Each holdout match must have two orientations."

match_diagnostics = (
    holdout.loc[holdout["player_a_win"].eq(1)]
    .set_index("match_id_internal")
)
match_diagnostics["correct"] = (
    match_diagnostics["actual_winner_probability"] >= 0.5
)
match_diagnostics["predicted_winner"] = np.where(
    match_diagnostics["correct"],
    match_diagnostics["winner_name"],
    match_diagnostics["loser_name"],
)

assert len(match_diagnostics) * 2 == len(holdout)
print(f"Real holdout matches: {len(match_diagnostics):,}")
print(f"Match-level accuracy: {match_diagnostics['correct'].mean():.4f}")"""
        ),
        new_markdown_cell("### Most confident correct predictions"),
        new_code_cell(
            """instance_columns = [
    "date",
    "winner_name",
    "loser_name",
    "predicted_winner",
    "surface",
    "round",
    "series",
    "winner_rank",
    "loser_rank",
    "normalized_market_prob_diff",
    "actual_winner_probability",
]

most_correct = match_diagnostics.nlargest(
    10, "actual_winner_probability"
)[instance_columns]
display(most_correct)"""
        ),
        new_markdown_cell("### Most confident wrong predictions"),
        new_code_cell(
            """most_wrong = match_diagnostics.nsmallest(
    10, "actual_winner_probability"
)[instance_columns]
display(most_wrong)"""
        ),
        new_markdown_cell("### Numeric feature distributions: correct vs wrong"),
        new_markdown_cell(
            """For each numeric feature, the table compares cohort medians,
missing-value rates and the mean shift measured in overall standard deviations.
Large absolute shifts identify where wrong predictions occupy a different part
of the feature distribution; they do not establish causality."""
        ),
        new_code_cell(
            """distribution_rows = []
for feature in numeric_features:
    correct_values = pd.to_numeric(
        match_diagnostics.loc[match_diagnostics["correct"], feature],
        errors="coerce",
    )
    wrong_values = pd.to_numeric(
        match_diagnostics.loc[~match_diagnostics["correct"], feature],
        errors="coerce",
    )
    overall_std = pd.to_numeric(
        match_diagnostics[feature], errors="coerce"
    ).std()
    standardized_shift = (
        (wrong_values.mean() - correct_values.mean()) / overall_std
        if overall_std > 0
        else np.nan
    )
    distribution_rows.append(
        {
            "feature": feature,
            "wrong_minus_correct_sd": standardized_shift,
            "correct_median": correct_values.median(),
            "wrong_median": wrong_values.median(),
            "correct_missing_rate": correct_values.isna().mean(),
            "wrong_missing_rate": wrong_values.isna().mean(),
        }
    )

numeric_distribution_shift = (
    pd.DataFrame(distribution_rows)
    .assign(
        absolute_shift=lambda frame: frame["wrong_minus_correct_sd"].abs(),
        missing_rate_delta=lambda frame: (
            frame["wrong_missing_rate"] - frame["correct_missing_rate"]
        ),
    )
    .sort_values("absolute_shift", ascending=False)
)
display(numeric_distribution_shift.head(15))

top_shift_features = numeric_distribution_shift.head(4)["feature"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for feature, axis in zip(top_shift_features, axes.ravel(), strict=True):
    match_diagnostics.boxplot(
        column=feature,
        by="correct",
        ax=axis,
        showfliers=False,
        grid=False,
    )
    axis.set_title(feature)
    axis.set_xlabel("Prediction correct")
    axis.set_ylabel("Value")
plt.suptitle("Largest numeric distribution shifts")
plt.tight_layout()
plt.show()"""
        ),
        new_markdown_cell("### Categorical feature distributions"),
        new_code_cell(
            """overall_error_rate = 1 - match_diagnostics["correct"].mean()
categorical_rows = []
for feature in categorical_features:
    grouped = match_diagnostics.groupby(feature, dropna=False)["correct"].agg(
        matches="size",
        error_rate=lambda values: 1 - values.mean(),
    )
    grouped = grouped.loc[grouped["matches"] >= 20].reset_index()
    grouped.insert(0, "feature", feature)
    categorical_rows.append(grouped.rename(columns={feature: "value"}))

categorical_error_rates = pd.concat(categorical_rows, ignore_index=True)
categorical_error_rates["difference_from_overall"] = (
    categorical_error_rates["error_rate"] - overall_error_rate
)
categorical_error_rates["absolute_difference"] = categorical_error_rates[
    "difference_from_overall"
].abs()
categorical_error_rates = categorical_error_rates.sort_values(
    "absolute_difference", ascending=False
)
display(categorical_error_rates.head(15))"""
        ),
        new_markdown_cell("### What is peculiar?"),
        new_code_cell(
            """odds_importance_share = feature_importance.loc[
    [
        "winner_implied_prob_avg",
        "loser_implied_prob_avg",
        "implied_prob_diff_avg",
        "winner_market_prob_normalized",
        "loser_market_prob_normalized",
        "normalized_market_prob_diff",
    ]
].sum() / feature_importance.sum()
wrong_matches = match_diagnostics.loc[~match_diagnostics["correct"]]
upset_share = wrong_matches["normalized_market_prob_diff"].lt(0).mean()
largest_missing_gap = numeric_distribution_shift["missing_rate_delta"].abs().max()

print(f"Importance assigned to six odds features: {odds_importance_share:.1%}")
print(f"Wrong predictions where the actual winner was the market underdog: {upset_share:.1%}")
print(f"Largest correct/wrong missing-rate gap: {largest_missing_gap:.1%}")
print(
    "Interpretation: errors are mostly genuine upsets. Ranking, Elo, form and "
    "market-probability differences reverse direction in the wrong cohort. "
    "No comparably large missing-data anomaly is visible. Category-level "
    "differences should be treated cautiously because many groups are small."
)"""
        ),
        new_markdown_cell(
            f"""## Conclusion

- The naive baseline obtains about **50% accuracy** because the classes are
  balanced.
- CatBoost obtains **{cv_result}** in cross-validation and
  **{holdout_accuracy:.4f} test accuracy**, so the result is stable on future
  matches.
- The model improves clearly over the naive baseline.
- Market-implied probabilities are the most important features. Therefore,
  part of the predictive power comes from information already contained in
  betting odds.
- The most confident mistakes are mainly upsets in which rankings, Elo, form
  and betting odds all favored the eventual loser. There is no similarly large
  missing-value anomaly separating correct and wrong predictions.
- The difference between training and test accuracy should be monitored: a
  much larger training score would indicate over-fitting.

The final test set is used only once for evaluation. Any future change to the
features or CatBoost parameters must be selected using the training
cross-validation folds, not this test result."""
        ),
    ]

    notebook = new_notebook(
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
    if language == "it":
        translate_to_italian(notebook)
    return notebook


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    metrics_path = project_root / "data/interim/catboost_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    notebook = build_notebook(metrics, args.language)

    if args.execute:
        client = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(project_root)}},
        )
        client.execute()

    output = args.output or project_root / "docs" / (
        "catboost_training_analysis_it.ipynb"
        if args.language == "it"
        else "catboost_training_analysis.ipynb"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Notebook: {output}")
    print(f"Executed: {args.execute}")


if __name__ == "__main__":
    main()
