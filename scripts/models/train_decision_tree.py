from __future__ import annotations

"""
train_decision_tree.py

Addestra un DecisionTreeClassifier (criterion="gini") sul dataset prodotto da
build_features.py, con:

  - grid search su max_depth / min_samples_leaf / min_samples_split / ccp_alpha
  - fold di cross-validation costruiti su DATE UNICHE (non su indice di riga),
    cosi' le due righe mirror dello stesso match (dataset simmetrico, stessa
    data) restano sempre nello stesso fold e non si separano tra train e
    validation come poteva succedere con TimeSeriesSplit "grezzo"
  - un holdout finale cronologico (ultima porzione di date), mai usato durante
    il tuning, per una stima onesta della generalizzazione
  - output interpretabile: feature importances, albero in formato testuale,
    plot dei primi livelli

Uso tipico (dalla root del progetto):

    uv run python scripts/models/train_decision_tree.py \
        --input data/interim/tennis_matches_features.data \
        --metadata data/interim/tennis_matches_features_metadata.json \
        --holdout-fraction 0.15 \
        --cv-folds 5
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Gini-criterion Decision Tree on the engineered tennis dataset."
    )
    default_root = Path(__file__).resolve().parents[2]

    parser.add_argument(
        "--input",
        type=Path,
        default=default_root / "data" / "interim" / "tennis_matches_features.data",
        help="Output di build_features.py",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=default_root / "data" / "interim" / "tennis_matches_features_metadata.json",
        help="Metadata JSON prodotto da build_features.py (elenco numeric/categorical features)",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.15,
        help="Frazione (per data, non per riga) tenuta da parte come test set finale",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Numero di fold basati sulle date per la grid search",
    )
    parser.add_argument(
        "--exclude-features",
        type=str,
        default="",
        help=(
            "Lista di feature da escludere, separate da virgola "
            "(es. per allenare il modello senza le quote di mercato)"
        ),
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=default_root / "data" / "interim" / "decision_tree_metrics.json",
    )
    parser.add_argument(
        "--output-tree-plot",
        type=Path,
        default=default_root / "data" / "interim" / "decision_tree_plot.png",
    )
    parser.add_argument(
        "--plot-max-depth",
        type=int,
        default=3,
        help="Profondita' massima mostrata nel plot (l'albero addestrato puo' essere piu' profondo)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Fold basati su date uniche (non su indice di riga)
# ---------------------------------------------------------------------------

def make_date_based_folds(dates: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Espande la finestra di training un chunk di date alla volta, come
    TimeSeriesSplit, ma i confini dei fold cadono sempre TRA una data e la
    successiva, mai in mezzo a una giornata di partite: cosi' le due righe
    mirror dello stesso match (dataset simmetrico) restano sempre nello
    stesso fold, sia in training sia in validation.
    """
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < n_splits + 1:
        raise ValueError(
            f"Troppe poche date uniche ({len(unique_dates)}) per {n_splits} fold."
        )

    date_chunks = np.array_split(unique_dates, n_splits + 1)
    dates_values = dates.to_numpy()

    folds = []
    train_dates = set(date_chunks[0].tolist())
    for chunk in date_chunks[1:]:
        chunk_set = set(chunk.tolist())
        train_idx = np.where(np.isin(dates_values, list(train_dates)))[0]
        test_idx = np.where(np.isin(dates_values, list(chunk_set)))[0]
        folds.append((train_idx, test_idx))
        train_dates |= chunk_set

    return folds


def chronological_holdout_split(dates: pd.Series, holdout_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    unique_dates = np.sort(dates.unique())
    cutoff_index = int(len(unique_dates) * (1 - holdout_fraction))
    cutoff_date = unique_dates[cutoff_index]

    dates_values = dates.to_numpy()
    train_idx = np.where(dates_values < cutoff_date)[0]
    holdout_idx = np.where(dates_values >= cutoff_date)[0]

    return train_idx, holdout_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("TRAIN DECISION TREE (Gini)")
    print("=" * 70)

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    numeric_features = [c for c in metadata["numeric_features"] if c in df.columns]
    categorical_features = [c for c in metadata["categorical_features"] if c in df.columns]

    # Feature escluse esplicitamente dalla riga di comando
    excluded_features = {
        f.strip()
        for f in args.exclude_features.split(",")
        if f.strip()
    }

    if excluded_features:
        numeric_features = [
            c for c in numeric_features
            if c not in excluded_features
        ]
        categorical_features = [
            c for c in categorical_features
            if c not in excluded_features
        ]

    print(f"Feature escluse: {sorted(excluded_features)}")
    print(f"Righe: {len(df):,}")
    print(f"Feature numeriche: {len(numeric_features)}")
    print(f"Feature categoriche: {len(categorical_features)}")

    X = df[numeric_features + categorical_features].copy()
    y = df["player_a_win"].astype(int)
    dates = df["date"]

    # ------------------------------------------------------------------
    # Holdout cronologico finale (mai visto durante il tuning)
    # ------------------------------------------------------------------

    train_idx, holdout_idx = chronological_holdout_split(dates, args.holdout_fraction)
    X_train, y_train, dates_train = X.iloc[train_idx], y.iloc[train_idx], dates.iloc[train_idx]
    X_holdout, y_holdout = X.iloc[holdout_idx], y.iloc[holdout_idx]

    print()
    print(f"Training set: {len(X_train):,} righe")
    print(f"Holdout finale: {len(X_holdout):,} righe (ultimo {args.holdout_fraction:.0%} delle date)")

    # ------------------------------------------------------------------
    # Fold per la grid search, basati su date
    # ------------------------------------------------------------------

    cv_folds = make_date_based_folds(dates_train.reset_index(drop=True), args.cv_folds)
    print(f"Fold di cross-validation (basati su date): {len(cv_folds)}")

    # ------------------------------------------------------------------
    # Pipeline: imputazione + one-hot (l'albero non richiede scaling)
    # ------------------------------------------------------------------

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    tree_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", DecisionTreeClassifier(criterion="gini", random_state=42)),
        ]
    )

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    param_grid = {
        "model__max_depth": [4, 6, 8, 10, 12, None],
        "model__min_samples_leaf": [1, 5, 20, 50],
        "model__min_samples_split": [2, 20, 50],
        "model__ccp_alpha": [0.0, 0.0005, 0.001, 0.005],
    }

    print()
    print("Grid search in corso (puo' richiedere qualche minuto)...")

    search = GridSearchCV(
        tree_pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )
    search.fit(X_train, y_train)

    print(f"Migliori parametri: {search.best_params_}")
    print(f"Miglior accuratezza CV: {search.best_score_:.4f}")

    best_pipeline = search.best_estimator_

    # ------------------------------------------------------------------
    # Valutazione finale su holdout mai visto
    # ------------------------------------------------------------------

    holdout_predictions = best_pipeline.predict(X_holdout)
    holdout_accuracy = accuracy_score(y_holdout, holdout_predictions)

    print()
    print(f"Accuratezza su holdout finale: {holdout_accuracy:.4f}")
    print()
    print(classification_report(y_holdout, holdout_predictions))

    # ------------------------------------------------------------------
    # Interpretabilita'
    # ------------------------------------------------------------------

    fitted_preprocess = best_pipeline.named_steps["preprocess"]
    fitted_tree = best_pipeline.named_steps["model"]
    feature_names = fitted_preprocess.get_feature_names_out().tolist()

    importances = sorted(
        zip(feature_names, fitted_tree.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top_importances = [{"feature": name, "importance": float(score)} for name, score in importances[:20]]

    print()
    print("Top 10 feature per importanza:")
    for item in top_importances[:10]:
        print(f"  {item['feature']:<40} {item['importance']:.4f}")

    tree_text = export_text(fitted_tree, feature_names=feature_names, max_depth=args.plot_max_depth)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            fitted_tree,
            feature_names=feature_names,
            class_names=["player_a_loses", "player_a_wins"],
            filled=True,
            max_depth=args.plot_max_depth,
            fontsize=8,
            ax=ax,
        )
        args.output_tree_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_tree_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nPlot albero (primi {args.plot_max_depth} livelli): {args.output_tree_plot}")
    except ImportError:
        print("\nmatplotlib non disponibile: salto il plot dell'albero (solo export testuale).")

    # ------------------------------------------------------------------
    # Salvataggio metriche
    # ------------------------------------------------------------------

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_holdout)),
        "cv_folds": len(cv_folds),
        "best_params": search.best_params_,
        "best_cv_accuracy": float(search.best_score_),
        "holdout_accuracy": float(holdout_accuracy),
        "confusion_matrix_holdout": confusion_matrix(y_holdout, holdout_predictions).tolist(),
        "tree_depth": int(fitted_tree.get_depth()),
        "tree_leaf_count": int(fitted_tree.get_n_leaves()),
        "top_feature_importances": top_importances,
        "tree_text_preview": tree_text,
    }

    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print(f"Metriche salvate in: {args.output_metrics}")
    print("Pipeline completata.")


if __name__ == "__main__":
    main()