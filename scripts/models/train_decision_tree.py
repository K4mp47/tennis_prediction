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

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from training_utils import (
    chronological_holdout_split,
    load_feature_dataset,
    make_date_based_folds,
    parse_excluded_features,
)


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
        default=default_root
        / "data"
        / "interim"
        / "tennis_matches_features_metadata.json",
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
    parser.add_argument( #percorso immagine dell'albero
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

# Main--------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("TRAIN DECISION TREE (Gini)")
    print("=" * 70)

    excluded_features = parse_excluded_features(args.exclude_features)
    dataset = load_feature_dataset(args.input, args.metadata, excluded_features)
    numeric_features = list(dataset.numeric_features)
    categorical_features = list(dataset.categorical_features)

    print(f"Feature escluse: {sorted(excluded_features)}")
    print(f"Righe: {len(dataset.features):,}")
    print(f"Feature numeriche: {len(numeric_features)}")
    print(f"Feature categoriche: {len(categorical_features)}")
    #separa gli input X dal target y
    X = dataset.features
    y = dataset.target
    dates = dataset.dates

    # Holdout cronologico finale (mai visto durante il tuning)------------------------------------------------------------
    train_idx, holdout_idx = chronological_holdout_split(dates, args.holdout_fraction)
    X_train, y_train, dates_train = ( #Training: partite più vecchie, Holdout: partite più recenti
        X.iloc[train_idx], #informazioni utilizzate addestramento
        y.iloc[train_idx],
        dates.iloc[train_idx],
    )
    X_holdout, y_holdout = X.iloc[holdout_idx], y.iloc[holdout_idx] #informazioni fuori da tuning

    print()
    print(f"Training set: {len(X_train):,} righe")
    print(
        f"Holdout finale: {len(X_holdout):,} righe (ultimo {args.holdout_fraction:.0%} delle date)"
    )

    # Fold per la grid search, basati su date----------------------------------------------------------
    cv_folds = make_date_based_folds(dates_train.reset_index(drop=True), args.cv_folds)
    print(f"Fold di cross-validation (basati su date): {len(cv_folds)}")

    # Pipeline: imputazione + one-hot (l'albero non richiede scaling)---------------------------------------------------------
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"), #valori mancanti -> mediana
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")), #valori mancanti -> +frequente
                        ("onehot", OneHotEncoder(handle_unknown="ignore")), #categorie -> col numeriche
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    tree_pipeline = Pipeline( #Unisce preprocessing + modello
        steps=[
            ("preprocess", preprocess),
            ("model", DecisionTreeClassifier(criterion="gini", random_state=42)), #random_state=42->riproducibili i risultati quando componenti casuali
        ]
    )

    # Grid search-------------------------------------------------------------
    param_grid = {
        "model__max_depth": [4, 6, 8, 10, 12, None],
        "model__min_samples_leaf": [1, 5, 20, 50], #Minimo numero di osservazioni in 1 foglia
        "model__min_samples_split": [2, 20, 50], #Minimo numero di osservazioni per dividere 1 nodo
        "model__ccp_alpha": [0.0, 0.0005, 0.001, 0.005], #	Intensità potatura
    }

    print()
    print("Grid search in corso (puo' richiedere qualche minuto)...")

    search = GridSearchCV( #prova diverse configurazioni Decision Tree, seleziona migliori
        tree_pipeline, #preprocessing + Decision Tree
        param_grid=param_grid, #combinazioni parametri
        cv=cv_folds, #suddivisione training set in fold
        scoring="accuracy", #accuracy per valutare configurazioni
        n_jobs=-1, #utilizza tutti i core CPU disponibili
        refit=True, #riaddestra automaticamente su tutto il training set
        verbose=2, #stampa informazioni dettagliate durante l'esecuzione
    )
    search.fit(X_train, y_train) #avvia addestramento

    print(f"Migliori parametri: {search.best_params_}")
    print(f"Miglior accuratezza CV: {search.best_score_:.4f}")

    best_pipeline = search.best_estimator_ #Recupera pipeline migliore

    # Valutazione finale su holdout mai visto--------------------------------------------------------------
    holdout_predictions = best_pipeline.predict(X_holdout) #risultati partite holdout
    holdout_accuracy = accuracy_score(y_holdout, holdout_predictions) #percentuale previsioni corrette

    print()
    print(f"Accuratezza su holdout finale: {holdout_accuracy:.4f}")
    print()
    print(classification_report(y_holdout, holdout_predictions))

    # Interpretabilita'---------------------------------------------------------------
    fitted_preprocess = best_pipeline.named_steps["preprocess"] #Recupera preprocessing già addestrato
    fitted_tree = best_pipeline.named_steps["model"] #Recupera albero già addestrato
    feature_names = fitted_preprocess.get_feature_names_out().tolist() #nomi feature dopo preprocessing e OneHotEncoding

    importances = sorted( #ordina da più importante a meno
        zip(feature_names, fitted_tree.feature_importances_), #Restituisce array. zip: combina nomi
        key=lambda pair: pair[1], #secondo elemento coppia= l'importanza, come criterio ordinamento
        reverse=True, #decrescente
    )
    top_importances = [ #Conserva prime 20
        {"feature": name, "importance": float(score)}
        for name, score in importances[:20]
    ]

    print()
    print("Top 10 feature per importanza:")
    for item in top_importances[:10]:
        print(f"  {item['feature']:<40} {item['importance']:.4f}")

    tree_text = export_text( #rappresentazione testuale dell'albero
        fitted_tree, feature_names=feature_names, max_depth=args.plot_max_depth
    ) #tree addestrato, feature, limita profondità

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree( #disegna tree in png
            fitted_tree, #modello da visualizzare
            feature_names=feature_names,
            class_names=["player_a_loses", "player_a_wins"],
            filled=True, #colora nodi
            max_depth=args.plot_max_depth,
            fontsize=8,
            ax=ax, #x disegnare
        )
        args.output_tree_plot.parent.mkdir(parents=True, exist_ok=True) #create path
        fig.savefig(args.output_tree_plot, dpi=150, bbox_inches="tight") #150 risoluzione, tight -spazi vuoti
        plt.close(fig)
        print(
            f"\nPlot albero (primi {args.plot_max_depth} livelli): {args.output_tree_plot}"
        )
    except ImportError:
        print(
            "\nmatplotlib non disponibile: salto il plot dell'albero (solo export testuale)."
        )

    #Salvataggio metriche-------------------------------------------------------------
    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input), #percorso dataset
        "train_rows": int(len(X_train)), #number of raw for training
        "holdout_rows": int(len(X_holdout)),#"" for final test
        "cv_folds": len(cv_folds), #number cv folds
        "best_params": search.best_params_, # in grid search
        "best_cv_accuracy": float(search.best_score_),
        "holdout_accuracy": float(holdout_accuracy),
        "confusion_matrix_holdout": confusion_matrix( #confronta classi reali/predette
            y_holdout, holdout_predictions
        ).tolist(),
        "tree_depth": int(fitted_tree.get_depth()),
        "tree_leaf_count": int(fitted_tree.get_n_leaves()),
        "top_feature_importances": top_importances, #top 20
        "tree_text_preview": tree_text, #Rappresentazione testuale tree
    }

    args.output_metrics.parent.mkdir(parents=True, exist_ok=True) #crea cartella
    args.output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8") #save in a file

    print()
    print(f"Metriche salvate in: {args.output_metrics}")
    print("Pipeline completata.")


if __name__ == "__main__":
    main()
