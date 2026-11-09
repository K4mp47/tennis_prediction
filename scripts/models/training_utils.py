"""Shared utilities for training tennis match classifiers.

The feature dataset contains two mirrored rows for every real match.  Splits
must therefore be made on complete dates, rather than on row positions, to
keep both orientations of a match in the same partition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True) #no modifications after
class FeatureDataset:
    """Validated model inputs and their feature schema."""

    features: pd.DataFrame
    target: pd.Series #output: player_a_win
    dates: pd.Series
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]


def parse_excluded_features(value: str) -> set[str]: #exclude some features
    """Parse a comma-separated feature list passed on the command line."""

    return {feature.strip() for feature in value.split(",") if feature.strip()} #strip: delete spaces, set: no duplicates


def load_feature_dataset(
    input_path: Path,
    metadata_path: Path,
    excluded_features: set[str] | None = None,
) -> FeatureDataset:
    """Load and validate the output produced by ``build_features.py``."""

    excluded_features = excluded_features or set() #set: no duplicates
    frame = pd.read_csv(input_path)

    required_columns = {"date", "player_a_win"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset privo delle colonne obbligatorie: {sorted(missing_columns)}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in ("numeric_features", "categorical_features"):
        if key not in metadata or not isinstance(metadata[key], list):
            raise ValueError(f"Metadata non validi: '{key}' deve essere una lista.")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce") #converts in DATETIME format77
    frame = frame.dropna(subset=["date"]).reset_index(drop=True)
    if frame.empty: #if dataset
        raise ValueError("Il dataset non contiene righe con una data valida.")

    numeric_features = tuple( #feature selection
        feature
        for feature in metadata["numeric_features"]
        if feature in frame.columns and feature not in excluded_features
    )
    categorical_features = tuple(
        feature
        for feature in metadata["categorical_features"]
        if feature in frame.columns and feature not in excluded_features
    )

    duplicated_features = set(numeric_features).intersection(categorical_features)
    if duplicated_features:
        raise ValueError(
            "Feature presenti sia tra le numeriche sia tra le categoriche: "
            f"{sorted(duplicated_features)}"
        )
    if not numeric_features and not categorical_features:
        raise ValueError("Nessuna feature disponibile per l'addestramento.") #and stop

    target = pd.to_numeric(frame["player_a_win"], errors="raise").astype(int) #target -> numeric -> int
    target_values = set(target.unique().tolist()) #array->list, no duplicates
    if not target_values.issubset({0, 1}) or len(target_values) < 2:
        raise ValueError(
            "La target 'player_a_win' deve contenere entrambe le classi binarie 0 e 1."
        )

    feature_names = list(numeric_features + categorical_features) #what needed to train model
    return FeatureDataset(
        features=frame.loc[:, feature_names].copy(),
        target=target,
        dates=frame["date"].copy(),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def make_date_based_folds(
    dates: pd.Series,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window CV folds whose boundaries fall between dates."""

    if n_splits < 2:
        raise ValueError("Il numero di fold deve essere almeno 2.")

    unique_dates = np.sort(dates.unique()) #orders date chronologically
    if len(unique_dates) < n_splits + 1: #distinct dates < number folds
        raise ValueError(
            f"Troppe poche date uniche ({len(unique_dates)}) per {n_splits} fold."
        )

    date_chunks = np.array_split(unique_dates, n_splits + 1) #divide unique dates in n_splits blocks
    date_values = dates.to_numpy() #dates -> array
    folds: list[tuple[np.ndarray, np.ndarray]] = [] #create list for folds -> index_row_train, index_row_validation
    train_dates = set(date_chunks[0].tolist()) #1° block for training

    for validation_dates in date_chunks[1:]: #for other blocks
        validation_date_set = set(validation_dates.tolist())
        train_indices = np.flatnonzero(np.isin(date_values, list(train_dates))) #checks if raws in date_training
        validation_indices = np.flatnonzero( #identifica righe blocco validation
            np.isin(date_values, list(validation_date_set))
        )
        folds.append((train_indices, validation_indices)) #save fold
        train_dates.update(validation_date_set) #add validation_date to training next folds

    return folds


def chronological_holdout_split( #Separa dataset training e holdout finale
    dates: pd.Series,
    holdout_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve the most recent fraction of unique dates as a final holdout."""

    if not 0 < holdout_fraction < 1:
        raise ValueError("La frazione di holdout deve essere compresa tra 0 e 1.")

    unique_dates = np.sort(dates.unique()) #Calcola data separazione
    cutoff_index = int(len(unique_dates) * (1 - holdout_fraction)) #calcolo soglia validation/training
    if cutoff_index <= 0 or cutoff_index >= len(unique_dates):
        raise ValueError(
            "La frazione di holdout non produce partizioni non vuote con le date "
            "disponibili."
        )

    cutoff_date = unique_dates[cutoff_index]
    date_values = dates.to_numpy() #dates -> array
    train_indices = np.flatnonzero(date_values < cutoff_date) #date precedenti soglia
    holdout_indices = np.flatnonzero(date_values >= cutoff_date) #date successive soglia
    return train_indices, holdout_indices
