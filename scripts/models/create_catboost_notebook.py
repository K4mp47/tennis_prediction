"""Create and optionally execute a source-verified CatBoost analysis notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--language", choices=("en", "it"), default="en")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def build_notebook(language: str = "en") -> nbformat.NotebookNode:
    """Build both languages from the same evaluation code and artifact contract."""
    if language not in {"en", "it"}:
        raise ValueError("Unsupported notebook language.")
    italian = language == "it"
    labels = {
        "dataset": "Dimensioni del dataset" if italian else "Dataset shape",
        "train": "Addestramento" if italian else "Training",
        "test": "Test completo" if italian else "Full holdout",
        "naive": "Baseline ingenua" if italian else "Naive baseline",
        "subset": "CatBoost sulle righe con quote"
        if italian
        else "CatBoost on market rows",
        "market": "Mercato sulle stesse righe"
        if italian
        else "Market on identical rows",
        "coverage": "Copertura del mercato" if italian else "Market coverage",
        "verified": "Provenienza, split e metriche verificati"
        if italian
        else "Provenance, split and metrics verified",
        "player_b": "Vince B" if italian else "B wins",
        "player_a": "Vince A" if italian else "A wins",
        "confusion": "Matrice di confusione sul test"
        if italian
        else "Holdout confusion matrix",
        "importance": "Importanza delle feature" if italian else "Feature importance",
        "matches": "Partite reali nel test" if italian else "Real holdout matches",
        "match_accuracy": "Accuracy per partita" if italian else "Match-level accuracy",
        "missing": "Differenza massima nei tassi di valori mancanti"
        if italian
        else "Largest missing-rate gap",
        "odds": "Quota di importanza delle feature derivate dalle quote"
        if italian
        else "Importance share of odds features",
        "upsets": "Errori con vincitore sfavorito, tra quelli con quote"
        if italian
        else "Errors whose actual winner was the underdog, among errors with odds",
        "gap": "Differenza accuracy training meno test"
        if italian
        else "Training minus holdout accuracy",
        "delta": "Differenza accuracy CatBoost meno mercato sulle stesse righe"
        if italian
        else "CatBoost minus market accuracy on identical rows",
    }

    def md(english: str, italian_text: str):
        return new_markdown_cell(italian_text if italian else english)

    cells = [
        md(
            "# CatBoost tennis prediction analysis\n\nClass 1 means player A wins; class 0 means player B wins. This notebook loads the saved model, verifies its inputs and reproduces its evaluation without retraining.",
            "# Analisi della predizione del tennis con CatBoost\n\nLa classe 1 indica una vittoria di A; la classe 0 una vittoria di B. Il notebook carica il modello salvato, verifica gli input e riproduce la valutazione senza riaddestrarlo.",
        ),
        md(
            "CatBoost combines boosted decision trees and handles numeric and categorical features. This run uses 300 trees, depth 8, learning rate 0.03, L2 regularization 3 and seed 42, with no tuning or early stopping.",
            "CatBoost combina alberi decisionali tramite boosting e gestisce feature numeriche e categoriche. Questa esecuzione usa 300 alberi, profondità 8, learning rate 0.03, regolarizzazione L2 3 e seed 42, senza tuning o early stopping.",
        ),
        new_code_cell(
            f"TEXT = {labels!r}\n"
            + """import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from IPython.display import display
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts/models"))
from train_catboost import (
    classification_metrics, evaluate_market, load_verified_model, prepare_catboost_features,
)

FEATURES_PATH = PROJECT_ROOT / "data/interim/tennis_matches_features.data"
METADATA_PATH = PROJECT_ROOT / "data/interim/tennis_matches_features_metadata.json"
METRICS_PATH = PROJECT_ROOT / "data/interim/catboost_metrics.json"
MODEL_PATH = PROJECT_ROOT / "data/interim/catboost_model.cbm"
training_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
model = load_verified_model(
    MODEL_PATH, training_metrics, input_path=FEATURES_PATH, metadata_path=METADATA_PATH,
)
plt.style.use("seaborn-v0_8-whitegrid")
display(pd.Series(training_metrics["params"], name="parameters").to_frame())
display(pd.Series(training_metrics["versions"], name="version").to_frame())
print(training_metrics["created_at_utc"])
print(training_metrics["sha256"])"""
        ),
        md("## 1. Data and provenance", "## 1. Dati e provenienza"),
        md(
            "Each real match has two orientations with complementary targets. Legacy winner/loser columns mean player A/B after swapping. Tennis-Data histories use strictly earlier dates. Sackmann supplies snapshot biography only; service statistics with unverified availability are excluded. Odds observation times are not verified. Tournament and location are raw categories, with no frequencies fitted on the test set.",
            "Ogni partita ha due orientamenti con target complementari. Dopo lo scambio, le colonne legacy winner/loser rappresentano A/B. Gli storici Tennis-Data usano solo date precedenti. Sackmann fornisce soltanto biografia da snapshot; le statistiche di servizio con disponibilità non verificata sono escluse. Gli orari di osservazione delle quote non sono verificati. Torneo e località sono categorie originali, senza frequenze apprese dal test.",
        ),
        new_code_cell("""dataset = pd.read_csv(FEATURES_PATH)
dataset["date"] = pd.to_datetime(dataset["date"], errors="raise")
assert dataset["date"].notna().all()
pairs = dataset.groupby("match_id_internal")
assert pairs.size().eq(2).all()
assert pairs["player_a_win"].agg(lambda values: sorted(values) == [0, 1]).all()
assert pairs["date"].nunique().eq(1).all()
print(TEXT["dataset"], dataset.shape)
display(dataset[["date", "surface", "winner_name", "loser_name", "rank_diff", "win_rate_diff", "player_a_win"]].head())
display(dataset["player_a_win"].value_counts().sort_index().to_frame())"""),
        md("## 2. Saved chronological split", "## 2. Separazione cronologica salvata"),
        md(
            "The trainer reserves the newest 15% of distinct dates. This notebook uses its saved cutoff, not a new split. Both orientations stay together. Histories may update with earlier test dates, as in sequential daily prediction; the model remains fixed. Matches on a previous shared date use deterministic match-ID order in the last-five window because times are unavailable.",
            "Il trainer riserva il 15% più recente delle date distinte. Il notebook usa la soglia salvata, senza creare un nuovo split. I due orientamenti restano insieme. Gli storici possono includere date precedenti del test, come in una previsione giornaliera sequenziale; il modello resta fisso. Nella finestra delle ultime cinque partite, gli incontri di una stessa data precedente seguono l'ordine deterministico degli ID, perché mancano gli orari.",
        ),
        new_code_cell("""numeric_features = training_metrics["numeric_features"]
categorical_features = training_metrics["categorical_features"]
feature_names = numeric_features + categorical_features
X = prepare_catboost_features(dataset[feature_names], numeric_features, categorical_features)
y = dataset["player_a_win"].astype(int)
cutoff = pd.Timestamp(training_metrics["split_date"])
train_mask, test_mask = dataset["date"].lt(cutoff), dataset["date"].ge(cutoff)
X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]
assert len(X_train) == training_metrics["train_rows"]
assert len(X_test) == training_metrics["holdout_rows"]
for mask, key in [(train_mask, "train_date_range"), (test_mask, "holdout_date_range")]:
    dates = dataset.loc[mask, "date"]
    assert [dates.min().date().isoformat(), dates.max().date().isoformat()] == training_metrics[key]
split = pd.DataFrame({
    "rows": [len(X_train), len(X_test)],
    "first_date": [dataset.loc[train_mask, "date"].min(), dataset.loc[test_mask, "date"].min()],
    "last_date": [dataset.loc[train_mask, "date"].max(), dataset.loc[test_mask, "date"].max()],
}, index=[TEXT["train"], TEXT["test"]])
display(split)"""),
        md(
            "## 3. Evaluation and fair market comparison",
            "## 3. Valutazione e confronto omogeneo col mercato",
        ),
        md(
            "Accuracy and ROC AUC are better when higher; log loss and Brier score are better when lower. The full holdout and the shared market subset are shown separately. AUC is undefined for one-class subsets; all metrics are undefined when market coverage is zero.",
            "Accuracy e ROC AUC sono migliori quando crescono; log loss e Brier score quando diminuiscono. Il test completo e il sottoinsieme comune col mercato sono distinti. L'AUC non è definita su una sola classe; con copertura del mercato nulla tutte le metriche del confronto sono mancanti.",
        ),
        new_code_cell("""train_probability = model.predict_proba(X_train)[:, 1]
test_probability = model.predict_proba(X_test)[:, 1]
test_prediction = (test_probability >= 0.5).astype(int)
train_metrics = classification_metrics(y_train, train_probability)
test_metrics = classification_metrics(y_test, test_probability)
for actual, saved in [(train_metrics, "train_metrics"), (test_metrics, "holdout_metrics")]:
    for metric, value in actual.items():
        assert np.isclose(value, training_metrics[saved][metric], rtol=1e-10, atol=1e-12)
assert np.isclose(test_metrics["accuracy"], training_metrics["holdout_accuracy"])
assert np.isclose(test_metrics["roc_auc"], training_metrics["holdout_roc_auc"])
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
naive_metrics = classification_metrics(y_test, baseline.predict_proba(X_test)[:, 1])
market_probability = dataset.loc[test_mask, "winner_market_prob_normalized"]
market_comparison = evaluate_market(y_test, test_probability, market_probability)
results = pd.DataFrame.from_dict({
    TEXT["train"]: train_metrics,
    TEXT["naive"]: naive_metrics,
    TEXT["test"]: test_metrics,
    TEXT["subset"]: market_comparison["catboost"],
    TEXT["market"]: market_comparison["market"],
}, orient="index")
display(results)
print(TEXT["coverage"], f"{market_comparison['coverage']:.1%}", market_comparison["rows"])
print(TEXT["gap"], f"{train_metrics['accuracy'] - test_metrics['accuracy']:.4f}")
print(TEXT["verified"])"""),
        new_code_cell("""ConfusionMatrixDisplay.from_predictions(
    y_test, test_prediction, display_labels=[TEXT["player_b"], TEXT["player_a"]], cmap="Blues",
)
plt.title(TEXT["confusion"])
plt.grid(False)
plt.show()
print(classification_report(y_test, test_prediction, target_names=[TEXT["player_b"], TEXT["player_a"]], digits=4))
RocCurveDisplay.from_predictions(y_test, test_probability, name="CatBoost")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.show()"""),
        md("## 4. Feature importance", "## 4. Importanza delle feature"),
        md(
            "Importance describes the fitted model and does not establish causality. Only actual model inputs are included.",
            "L'importanza descrive il modello addestrato e non dimostra causalità. Sono incluse soltanto le feature effettivamente usate.",
        ),
        new_code_cell("""feature_importance = pd.Series(
    model.get_feature_importance(Pool(X_train, y_train, cat_features=categorical_features)),
    index=feature_names, name="importance",
).sort_values(ascending=False)
assert np.isfinite(feature_importance).all() and feature_importance.sum() > 0
display(feature_importance.head(10).to_frame())
feature_importance.head(10).sort_values().plot.barh(figsize=(8, 4.5), title=TEXT["importance"])
plt.tight_layout()
plt.show()"""),
        md(
            "## 5. Match-level successes and errors",
            "## 5. Successi ed errori per partita",
        ),
        md(
            "For this diagnostic only, average the probability assigned to the actual winner across both orientations, then count each real match once. Ties receive half credit and no predicted winner. Main evaluation above uses the original row predictions.",
            "Solo per questa diagnostica, si media la probabilità assegnata al vincitore reale nei due orientamenti e si conta ogni partita una volta. Le parità ricevono mezzo punto e nessun vincitore previsto. La valutazione principale usa le predizioni originali per riga.",
        ),
        new_code_cell("""holdout = dataset.loc[test_mask].copy()
holdout["actual_winner_probability"] = np.where(y_test.eq(1), test_probability, 1 - test_probability)
holdout["actual_winner_probability"] = holdout.groupby("match_id_internal")["actual_winner_probability"].transform("mean")
match_diagnostics = holdout.loc[holdout["player_a_win"].eq(1)].set_index("match_id_internal").copy()
assert len(match_diagnostics) * 2 == len(holdout)
match_diagnostics["tied"] = match_diagnostics["actual_winner_probability"].eq(0.5)
match_diagnostics["correct"] = match_diagnostics["actual_winner_probability"].gt(0.5)
match_diagnostics["accuracy_credit"] = match_diagnostics["correct"].astype(float) + 0.5 * match_diagnostics["tied"]
match_diagnostics["predicted_winner"] = np.where(
    match_diagnostics["correct"], match_diagnostics["winner_name"], match_diagnostics["loser_name"],
)
match_diagnostics.loc[match_diagnostics["tied"], "predicted_winner"] = "?"
print("Ties / Parità", int(match_diagnostics["tied"].sum()))
print(TEXT["matches"], len(match_diagnostics))
print(TEXT["match_accuracy"], f"{match_diagnostics['accuracy_credit'].mean():.4f}")
instance_columns = ["date", "winner_name", "loser_name", "predicted_winner", "surface", "round", "normalized_market_prob_diff", "actual_winner_probability"]
display(match_diagnostics.loc[match_diagnostics["correct"]].nlargest(10, "actual_winner_probability")[instance_columns])
display(match_diagnostics.loc[~match_diagnostics["correct"] & ~match_diagnostics["tied"]].nsmallest(10, "actual_winner_probability")[instance_columns])"""),
        md("## 6. Error distributions", "## 6. Distribuzione degli errori"),
        md(
            "Compare numeric distributions and missingness between correct and wrong predictions. Standardized differences are descriptive, not causal. Category error rates include only groups with at least 20 matches.",
            "Si confrontano distribuzioni numeriche e valori mancanti tra previsioni corrette ed errate. Le differenze standardizzate sono descrittive, non causali. I tassi di errore per categoria includono gruppi di almeno 20 partite.",
        ),
        new_code_cell("""distribution_rows = []
for feature in numeric_features:
    values = pd.to_numeric(match_diagnostics[feature], errors="coerce")
    correct = values.loc[match_diagnostics["correct"]]
    wrong = values.loc[~match_diagnostics["correct"] & ~match_diagnostics["tied"]]
    std = values.std()
    distribution_rows.append({
        "feature": feature,
        "wrong_minus_correct_sd": (wrong.mean() - correct.mean()) / std if std > 0 else np.nan,
        "correct_median": correct.median(), "wrong_median": wrong.median(),
        "correct_missing_rate": correct.isna().mean(), "wrong_missing_rate": wrong.isna().mean(),
    })
numeric_distribution_shift = pd.DataFrame(distribution_rows)
numeric_distribution_shift["absolute_shift"] = numeric_distribution_shift["wrong_minus_correct_sd"].abs()
numeric_distribution_shift["missing_rate_delta"] = numeric_distribution_shift["wrong_missing_rate"] - numeric_distribution_shift["correct_missing_rate"]
numeric_distribution_shift = numeric_distribution_shift.sort_values("absolute_shift", ascending=False)
display(numeric_distribution_shift.head(15))
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for feature, axis in zip(numeric_distribution_shift.head(4)["feature"], axes.ravel()):
    for correct, color in [(True, "#2a9d8f"), (False, "#e76f51")]:
        values = match_diagnostics.loc[match_diagnostics["correct"].eq(correct), feature].dropna()
        if len(values):
            axis.hist(values, bins=20, alpha=0.5, label=str(correct), color=color, density=True)
    axis.set_title(feature)
    axis.legend(title="correct")
plt.tight_layout()
plt.show()
categorical_rows = []
for feature in categorical_features:
    grouped = match_diagnostics.groupby(feature, dropna=False)["accuracy_credit"].agg(matches="size", accuracy="mean")
    grouped = grouped.loc[grouped["matches"].ge(20)].reset_index().rename(columns={feature: "value"})
    grouped["feature"] = feature
    grouped["error_rate"] = 1 - grouped["accuracy"]
    categorical_rows.append(grouped)
categorical_error_rates = pd.concat(categorical_rows, ignore_index=True)
display(categorical_error_rates.sort_values("error_rate", ascending=False).head(15))"""),
        md("## 7. Measured conclusions", "## 7. Conclusioni misurate"),
        new_code_cell("""odds_features = [name for name in feature_names if "implied_prob" in name or name == "has_any_market_odds"]
odds_importance_share = float(feature_importance.loc[odds_features].sum() / feature_importance.sum())
wrong_matches = match_diagnostics.loc[~match_diagnostics["correct"] & ~match_diagnostics["tied"]]
wrong_with_market = wrong_matches.dropna(subset=["normalized_market_prob_diff"])
upset_share = float(wrong_with_market["normalized_market_prob_diff"].lt(0).mean())
accuracy_delta = market_comparison["catboost"]["accuracy"] - market_comparison["market"]["accuracy"]
print(TEXT["delta"], f"{accuracy_delta:.4f}")
print(TEXT["odds"], len(odds_features), f"{odds_importance_share:.1%}")
print(TEXT["upsets"], f"{upset_share:.1%}", len(wrong_with_market))
print(TEXT["missing"], f"{numeric_distribution_shift['missing_rate_delta'].abs().max():.1%}")
summary = {
    "sha256": training_metrics["sha256"], "split_date": training_metrics["split_date"],
    "train_rows": len(X_train), "holdout_rows": len(X_test),
    "train_metrics": train_metrics, "holdout_metrics": test_metrics,
    "naive_metrics": naive_metrics, "market_comparison": market_comparison,
    "match_accuracy": float(match_diagnostics["accuracy_credit"].mean()),
    "holdout_matches": len(match_diagnostics), "match_ties": int(match_diagnostics["tied"].sum()),
    "dataset_shape": list(dataset.shape), "odds_importance_share": odds_importance_share,
    "upset_share": upset_share, "errors_with_market": len(wrong_with_market),
}
def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return None if isinstance(value, float) and not np.isfinite(value) else value
(PROJECT_ROOT / "data/interim/catboost_analysis_summary.json").write_text(
    json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\\n", encoding="utf-8",
)"""),
        md(
            "The tables and computed differences describe this run without assuming that CatBoost beats the market. Feature importance and error cohorts do not demonstrate causality or betting profit. This historical holdout has already been inspected and is not a fresh independent test. Future tuning must use training-only temporal folds and a subsequently untouched test. Snapshot biography and unverified odds timestamps limit point-in-time claims.",
            "Le tabelle e le differenze calcolate descrivono questa esecuzione senza assumere che CatBoost superi il mercato. Importanza delle feature e gruppi di errori non dimostrano causalità o profitto nelle scommesse. Questo test storico è già stato esaminato e non è un nuovo test indipendente. Il tuning futuro deve usare fold temporali del solo training e un successivo test mai esaminato. Biografia da snapshot e orari delle quote non verificati limitano le affermazioni sulla disponibilità storica.",
        ),
    ]
    for index, cell in enumerate(cells):
        cell.id = f"catboost-{index:02d}"
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )


def update_readme_results(readme_path: Path, metrics: dict, summary: dict) -> None:
    """Replace only the marked current-run section with verified measured results."""
    if summary["sha256"] != metrics["sha256"]:
        raise ValueError("Analysis summary does not match training artifacts.")
    start, end = "<!-- catboost-results:start -->", "<!-- catboost-results:end -->"
    readme = readme_path.read_text(encoding="utf-8")
    if (
        readme.count(start) != 1
        or readme.count(end) != 1
        or readme.index(start) >= readme.index(end)
    ):
        raise ValueError(
            "README must contain one ordered CatBoost results marker pair."
        )
    rows, columns = summary["dataset_shape"]
    lines = [
        f"Training run UTC: `{metrics['created_at_utc']}`.",
        f"Dataset: **{rows:,} oriented rows**, **{rows // 2:,} real matches**, {columns} columns, "
        f"from {metrics['train_date_range'][0]} to {metrics['holdout_date_range'][1]}.",
        f"The model uses **{len(metrics['numeric_features'])} numeric** and "
        f"**{len(metrics['categorical_features'])} categorical** features.",
        "",
        "| Partition | Rows | First date | Last date |",
        "| --- | ---: | --- | --- |",
        f"| Training | {metrics['train_rows']:,} | {' | '.join(metrics['train_date_range'])} |",
        f"| Holdout | {metrics['holdout_rows']:,} | {' | '.join(metrics['holdout_date_range'])} |",
        "",
        "| Evaluation | Accuracy | ROC AUC | Log loss | Brier score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    market = summary["market_comparison"]
    for label, values in [
        ("Training", summary["train_metrics"]),
        ("Naive holdout baseline", summary["naive_metrics"]),
        ("CatBoost full holdout", summary["holdout_metrics"]),
        ("CatBoost market subset", market["catboost"]),
        ("Market on identical rows", market["market"]),
    ]:
        formatted = [
            f"{values[key]:.6f}" if values[key] is not None else "undefined"
            for key in ("accuracy", "roc_auc", "log_loss", "brier_score")
        ]
        lines.append(f"| {label} | {' | '.join(formatted)} |")
    lines.extend(
        [
            "",
            f"Market coverage: **{market['coverage']:.1%}**, {market['rows']:,} oriented holdout rows.",
            f"Match-level accuracy after averaging orientations: **{summary['match_accuracy']:.6f}** "
            f"on {summary['holdout_matches']:,} matches. Exact ties: {summary['match_ties']}; ties receive half credit.",
            "",
            "Compare CatBoost and the market on the identical-row subset. These historical results do not "
            "establish betting profit or an advantage on a fresh independent test.",
        ]
    )
    prefix, rest = readme.split(start)
    _, suffix = rest.split(end)
    readme_path.write_text(
        prefix + start + "\n\n" + "\n".join(lines) + "\n\n" + end + suffix,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    notebook = build_notebook(args.language)
    if args.execute:
        NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        ).execute()
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    suffix = "_it" if args.language == "it" else ""
    output = args.output or root / "docs" / f"catboost_training_analysis{suffix}.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    if args.execute:
        metrics = json.loads(
            (root / "data/interim/catboost_metrics.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (root / "data/interim/catboost_analysis_summary.json").read_text(
                encoding="utf-8"
            )
        )
        update_readme_results(root / "README.md", metrics, summary)
    print(f"Notebook: {output}")
    print(f"Executed: {args.execute}")


if __name__ == "__main__":
    main()
