"""Train a simple CatBoost model for tennis match prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


MISSING_CATEGORY = "__MISSING__"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train the CatBoost tennis model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/interim/tennis_matches_features.data",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "data/interim/tennis_matches_features_metadata.json",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=root / "data/interim/catboost_model.cbm",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=root / "data/interim/catboost_metrics.json",
    )
    return parser.parse_args()


def prepare_catboost_features(
    features: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Prepare numeric and categorical columns for CatBoost."""

    features = features.copy()

    for column in numeric_features:
        features[column] = pd.to_numeric(features[column], errors="coerce") #col numeriche -> val numerici

    for column in categorical_features:
        features[column] = (
            features[column]
            .astype("string") #valori in str Pandas
            .fillna(MISSING_CATEGORY)
            .astype(str) #col tipo str Python
        )

    return features


def chronological_split(
    data: pd.DataFrame,
    test_fraction: float = 0.15, #lascia 15% date ai test
) -> tuple[pd.Series, pd.Series]:
    """Use old matches for training and the newest dates for testing."""

    dates = sorted(data["date"].unique()) #ordina date senza duplicati
    split_index = int(len(dates) * (1 - test_fraction)) #Individua data separa training e test
    split_date = dates[split_index]

    train_mask = data["date"] < split_date #partite precedenti data separazione
    test_mask = data["date"] >= split_date #partite poi

    return train_mask, test_mask


def main() -> None:
    args = parse_args()

    # 1. Load data
    data = pd.read_csv(args.input)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]
    feature_names = numeric_features + categorical_features

    data["date"] = pd.to_datetime(data["date"], errors="raise") #col date -> datetime, raise: stop with error

    X = prepare_catboost_features(
        data[feature_names],
        numeric_features,
        categorical_features,
    )
    y = data["player_a_win"].astype(int) #result: player_a_win=1=win

    # 2.Chronological train/test split
    train_mask, test_mask = chronological_split(data)

    X_train = X.loc[train_mask] #feature x addestrare
    X_test = X.loc[test_mask] #feature x valutare
    y_train = y.loc[train_mask] #result x addestrare
    y_test = y.loc[test_mask] #result x valutare

    #3.Create the model.
    model = CatBoostClassifier(
        iterations=300, #numero alberi boosting
        depth=8,
        learning_rate=0.03, #velocità correzione
        l2_leaf_reg=3, #limita overfitting penalizza valori elevati stime nelle foglie
        loss_function="Logloss", #valuta quanto probabilità predette, vicine risultati reali
        cat_features=categorical_features, #Indica colonne categoriche
        random_seed=42, #addestramento riproducibile stessi dati, cerca stesso risultato
        verbose=False, 	#Disattiva stampa
        allow_writing_files=False, #Evita creazione file lavoro aggiuntivi
    )

    #4 Train
    model.fit(X_train, y_train)

    # 5. Predict and evaluate
    predictions = model.predict(X_test).astype(int).ravel() #ravel() array bidimensionale -> array mono
    probabilities = model.predict_proba(X_test)[:, 1] #prob giocatore A vinca

    accuracy = accuracy_score(y_test, predictions) # % previsioni corrette
    roc_auc = roc_auc_score(y_test, probabilities) #quanto bene modello distingue due classi con diverse soglie

    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # 6. Save the model and minimal metadata needed by predict_catboost.py.
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_model))

    metrics = {
        "algorithm": "CatBoostClassifier",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "holdout_accuracy": float(accuracy), #sul test;
        "holdout_roc_auc": float(roc_auc), #sul test;
    }

    args.output_metrics.parent.mkdir(parents=True, exist_ok=True) #save metrics in JSON
    args.output_metrics.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
