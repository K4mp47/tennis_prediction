import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "models" / "train_catboost.py"
PREDICTION_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "models" / "predict_catboost.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))

from train_catboost import chronological_split  # noqa: E402


def test_chronological_split_keeps_dates_together() -> None:
    data = pd.DataFrame({"date": pd.to_datetime([
        "2024-01-04", "2024-01-01", "2024-01-03", "2024-01-02",
        "2024-01-04", "2024-01-01", "2024-01-03", "2024-01-02",
    ])})
    train, test = chronological_split(data, 0.25)
    assert train.sum() == 6
    assert test.sum() == 2
    assert (train ^ test).all()
    assert data.loc[train, "date"].max() < data.loc[test, "date"].min()


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
    assert metrics["numeric_features"] == ["strength_diff"]
    assert metrics["categorical_features"] == ["surface"]
    assert metrics["holdout_accuracy"] == 1.0
    assert metrics["holdout_roc_auc"] == 1.0

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
