import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODELS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "models"
sys.path.insert(0, str(MODELS_DIR))

from training_utils import (  # noqa: E402
    chronological_holdout_split,
    make_date_based_folds,
)


def test_holdout_split_keeps_each_date_in_one_partition() -> None:
    dates = pd.Series(
        pd.to_datetime(
            ["2024-01-01"] * 2
            + ["2024-01-02"] * 2
            + ["2024-01-03"] * 2
            + ["2024-01-04"] * 2
        )
    )

    train_indices, holdout_indices = chronological_holdout_split(dates, 0.25)

    assert set(dates.iloc[train_indices]).isdisjoint(dates.iloc[holdout_indices])
    assert dates.iloc[holdout_indices].nunique() == 1
    assert dates.iloc[holdout_indices].min() > dates.iloc[train_indices].max()


def test_date_folds_are_expanding_and_do_not_split_dates() -> None:
    dates = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=8, freq="D"), 2))

    folds = make_date_based_folds(dates, n_splits=3)

    assert len(folds) == 3
    previous_train_size = 0
    for train_indices, validation_indices in folds:
        train_dates = set(dates.iloc[train_indices])
        validation_dates = set(dates.iloc[validation_indices])
        assert train_dates.isdisjoint(validation_dates)
        assert max(train_dates) < min(validation_dates)
        assert len(train_indices) > previous_train_size
        previous_train_size = len(train_indices)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_holdout_fraction_must_be_between_zero_and_one(fraction: float) -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=5, freq="D"))

    with pytest.raises(ValueError, match="frazione di holdout"):
        chronological_holdout_split(dates, fraction)
