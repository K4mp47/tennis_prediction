import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "models" / "train_catboost.py"
PREDICTION_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "models" / "predict_catboost.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))

from train_catboost import ensure_disjoint_pairs, parse_args  # noqa: E402


def test_default_search_matches_reported_methodology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH)])

    args = parse_args()

    assert args.iterations == [500, 1000, 1500]
    assert args.depth == [5, 6, 7, 8]
    assert args.learning_rate == [0.01, 0.02, 0.03, 0.05]
    assert args.l2_leaf_reg == [3.0, 5.0, 10.0, 15.0]
    assert args.random_strength == [0.5, 1.0, 2.0]
    assert args.bagging_temperature == [0.0, 0.5, 1.0]
    assert args.search_iterations == 12
    assert args.early_stopping_rounds == 100


def test_pair_overlap_is_rejected() -> None:
    match_ids = pd.Series([10, 11, 10, 11])

    with pytest.raises(ValueError, match="separa 2 coppie"):
        ensure_disjoint_pairs(
            match_ids,
            np.array([0, 1]),
            np.array([2, 3]),
            "Fold CV 1",
        )


def test_catboost_training_smoke_run(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for day, date in enumerate(pd.date_range("2024-01-01", periods=12, freq="D")):
        rows.extend(
            [
                {
                    "match_id_internal": day,
                    "date": date.date().isoformat(),
                    "strength_diff": 1.0 + day / 100,
                    "surface": "hard" if day % 2 else "clay",
                    "player_a_win": 1,
                },
                {
                    "match_id_internal": day,
                    "date": date.date().isoformat(),
                    "strength_diff": -1.0 - day / 100,
                    "surface": "hard" if day % 2 else "clay",
                    "player_a_win": 0,
                },
            ]
        )

    input_path = tmp_path / "features.data"
    metadata_path = tmp_path / "metadata.json"
    metrics_path = tmp_path / "metrics.json"
    model_path = tmp_path / "model.cbm"
    predictions_path = tmp_path / "predictions.csv"
    holdout_predictions_path = tmp_path / "holdout_predictions.csv"
    diagnostics_path = tmp_path / "diagnostics.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "numeric_features": ["strength_diff"],
                "categorical_features": ["surface"],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_path),
            "--metadata",
            str(metadata_path),
            "--holdout-fraction",
            "0.25",
            "--cv-folds",
            "2",
            "--iterations",
            "5",
            "--depth",
            "2",
            "--learning-rate",
            "0.1",
            "--l2-leaf-reg",
            "3",
            "--random-strength",
            "1",
            "--bagging-temperature",
            "0",
            "--search-iterations",
            "1",
            "--early-stopping-rounds",
            "2",
            "--jobs",
            "1",
            "--output-metrics",
            str(metrics_path),
            "--output-model",
            str(model_path),
            "--output-holdout-predictions",
            str(holdout_predictions_path),
            "--output-match-diagnostics",
            str(diagnostics_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert model_path.stat().st_size > 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["algorithm"] == "CatBoostClassifier"
    assert metrics["best_search_params"] == {
        "bagging_temperature": 0.0,
        "depth": 2,
        "iterations": 5,
        "l2_leaf_reg": 3.0,
        "learning_rate": 0.1,
        "random_strength": 1.0,
    }
    assert metrics["best_params"]["iterations"] <= 5
    assert metrics["search_strategy"] == "random"
    assert metrics["selection_metric"] == "neg_log_loss"
    assert metrics["best_cv_accuracy"] is None
    assert metrics["pair_symmetry_enforced"] is True
    assert metrics["holdout_accuracy"] == 1.0
    assert metrics["holdout_brier_score"] >= 0
    assert holdout_predictions_path.exists()
    assert diagnostics_path.exists()
    holdout_predictions = pd.read_csv(holdout_predictions_path)
    diagnostics = pd.read_csv(diagnostics_path)
    assert holdout_predictions.groupby("match_pair_id").size().eq(2).all()
    assert len(diagnostics) * 2 == len(holdout_predictions)

    prediction_run = subprocess.run(
        [
            sys.executable,
            str(PREDICTION_SCRIPT_PATH),
            "--input",
            str(input_path),
            "--model",
            str(model_path),
            "--metrics",
            str(metrics_path),
            "--output",
            str(predictions_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert prediction_run.returncode == 0, prediction_run.stderr
    predictions = pd.read_csv(predictions_path)
    assert len(predictions) == len(rows)
    assert predictions["player_a_win_probability"].between(0, 1).all()
    assert set(predictions["predicted_winner_side"]) <= {"player_a", "player_b"}
