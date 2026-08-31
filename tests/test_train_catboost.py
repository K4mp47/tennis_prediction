import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "models" / "train_catboost.py"
PREDICTION_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "models" / "predict_catboost.py"


def test_catboost_training_smoke_run(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for day, date in enumerate(pd.date_range("2024-01-01", periods=12, freq="D")):
        rows.extend(
            [
                {
                    "date": date.date().isoformat(),
                    "strength_diff": 1.0 + day / 100,
                    "surface": "hard" if day % 2 else "clay",
                    "player_a_win": 1,
                },
                {
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
            "--jobs",
            "1",
            "--output-metrics",
            str(metrics_path),
            "--output-model",
            str(model_path),
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
    assert metrics["best_params"] == {
        "depth": 2,
        "iterations": 5,
        "l2_leaf_reg": 3.0,
        "learning_rate": 0.1,
    }
    assert metrics["holdout_accuracy"] == 1.0
    assert metrics["holdout_brier_score"] >= 0

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
