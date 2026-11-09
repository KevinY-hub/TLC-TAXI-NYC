"""Train the agreed scikit-learn RF comparison on the same hourly dataset/splits."""

from __future__ import annotations

import gc
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "output" / "machine_learning" / "dataset"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ROOT = PROJECT_ROOT / "output" / "machine_learning" / "sklearn_rf" / RUN_ID
REPORT_PATH = RUN_ROOT / "reports"
PREDICTION_PATH = RUN_ROOT / "predictions"

LABEL = "trip_count"
CATEGORY_COLS = ["PULocationID", "weather_code"]
NUMERIC_COLS = [
    "pickup_hour_local",
    "day_of_week",
    "pickup_month",
    "weekend_indicator",
    "weather_temperature_2m",
    "weather_precipitation",
    "weather_relative_humidity_2m",
    "weather_wind_speed_10m",
]
KEY_COLS = ["PULocationID", "pickup_hour_utc_epoch", "pickup_month_local"]
MODEL_COLS = list(dict.fromkeys(CATEGORY_COLS + NUMERIC_COLS + [LABEL] + KEY_COLS))
EXPECTED_ROWS = {"training": 718_545, "validation": 186_481, "testing": 180_689}
SPLIT_MONTHS = {
    "training": ["2026-01", "2026-02", "2026-03", "2026-04"],
    "validation": ["2026-05"],
    "testing": ["2026-06"],
}


def available_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def peak_rss_bytes() -> int | None:
    try:
        # Linux ru_maxrss is KiB; macOS reports bytes.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if os.uname().sysname == "Darwin" else peak * 1024)
    except (AttributeError, OSError):
        return None


def read_split(name: str) -> pd.DataFrame:
    dataset = ds.dataset(str(DATASET_PATH), format="parquet", partitioning="hive")
    months = SPLIT_MONTHS[name]
    filt = ds.field("pickup_month_local").isin(months)
    table = dataset.to_table(columns=MODEL_COLS, filter=filt)
    frame = table.to_pandas()
    del table
    gc.collect()
    expected = EXPECTED_ROWS[name]
    if len(frame) != expected:
        raise ValueError(f"Jumlah {name} {len(frame):,}; diharapkan {expected:,}.")
    null_counts = frame[MODEL_COLS].isna().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"Null pada split {name}: {bad.to_dict()}")
    return frame


def save_predictions(frame: pd.DataFrame, predictions: np.ndarray, split: str) -> None:
    output = pd.DataFrame(
        {
            "PULocationID": frame["PULocationID"].to_numpy(),
            "pickup_hour_utc_epoch": frame["pickup_hour_utc_epoch"].to_numpy(),
            "pickup_month_local": frame["pickup_month_local"].to_numpy(),
            "actual_trip_count": frame[LABEL].to_numpy(dtype=np.int64),
            "prediction": predictions.astype(np.float64),
        }
    )
    output["residual"] = output["prediction"] - output["actual_trip_count"]
    destination = PREDICTION_PATH / f"{split}.parquet"
    if destination.exists():
        raise FileExistsError(f"Prediksi tidak ditimpa: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(output, preserve_index=False), str(destination), compression="zstd")


def main() -> None:
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(f"Dataset model tidak ditemukan: {DATASET_PATH}")
    if RUN_ROOT.exists():
        raise FileExistsError(f"Direktori output sudah ada dan tidak ditimpa: {RUN_ROOT}")
    mem_available_before = available_memory_bytes()
    if mem_available_before is not None and mem_available_before < 4 * 1024**3:
        raise MemoryError(
            f"Memori tersedia hanya {mem_available_before / 1024**3:.2f} GiB; "
            "eksperimen dihentikan sebelum memuat split training."
        )

    training = read_split("training")
    category_counts = {
        col: int(training[col].nunique(dropna=True)) for col in CATEGORY_COLS
    }
    transformer = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                CATEGORY_COLS,
            ),
            ("numeric", "passthrough", NUMERIC_COLS),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        n_jobs=2,
    )
    pipeline = Pipeline([("preprocess", transformer), ("rf", model)])

    fit_start = time.perf_counter()
    pipeline.fit(training[CATEGORY_COLS + NUMERIC_COLS], training[LABEL])
    fit_seconds = time.perf_counter() - fit_start
    training_rows = len(training)
    del training
    gc.collect()

    metric_rows: list[dict[str, Any]] = []
    test_frame: pd.DataFrame | None = None
    test_predictions: np.ndarray | None = None
    prediction_seconds: dict[str, float] = {}
    for split in ("validation", "testing"):
        frame = read_split(split)
        predict_start = time.perf_counter()
        pred = pipeline.predict(frame[CATEGORY_COLS + NUMERIC_COLS])
        prediction_seconds[split] = time.perf_counter() - predict_start
        actual = frame[LABEL].to_numpy(dtype=np.float64)
        metric_rows.append(
            {
                "implementation": "scikit-learn RandomForestRegressor",
                "split": split,
                "rows": len(frame),
                "trip_count_sum": int(frame[LABEL].sum()),
                "MAE": float(mean_absolute_error(actual, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
                "R2": float(r2_score(actual, pred)),
                "fit_seconds": fit_seconds,
                "prediction_seconds": prediction_seconds[split],
            }
        )
        save_predictions(frame, pred, split)
        if split == "testing":
            test_frame = frame
            test_predictions = pred
        else:
            del frame, pred
            gc.collect()

    if test_frame is None or test_predictions is None:
        raise RuntimeError("Prediksi test tidak tersedia.")

    transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    raw_importance = pipeline.named_steps["rf"].feature_importances_
    if len(transformed_names) != len(raw_importance):
        raise ValueError("Jumlah feature names dan importance tidak cocok.")
    importance_map: dict[str, float] = {col: 0.0 for col in CATEGORY_COLS + NUMERIC_COLS}
    encoded_rows = []
    for name, importance in zip(transformed_names, raw_importance):
        name = str(name)
        source = name.split("__", 1)[-1]
        if source.startswith("PULocationID_"):
            raw_feature = "PULocationID"
        elif source.startswith("weather_code_"):
            raw_feature = "weather_code"
        else:
            raw_feature = source
        importance_map[raw_feature] = importance_map.get(raw_feature, 0.0) + float(importance)
        encoded_rows.append({"encoded_feature": name, "importance": float(importance)})

    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(REPORT_PATH / "model_metrics.csv", index=False)
    importance_rows = sorted(
        [
            {"feature": feature, "importance": value}
            for feature, value in importance_map.items()
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )
    for rank, row in enumerate(importance_rows, 1):
        row["rank"] = rank
    pd.DataFrame(importance_rows).to_csv(REPORT_PATH / "feature_importance_by_source.csv", index=False)
    pd.DataFrame(encoded_rows).to_csv(REPORT_PATH / "feature_importance_encoded.csv", index=False)

    sample = test_frame[["PULocationID", "pickup_hour_utc_epoch", LABEL]].copy()
    sample["prediction"] = test_predictions
    sample = sample.sample(n=min(5000, len(sample)), random_state=42).sort_values(
        ["pickup_hour_utc_epoch", "PULocationID"]
    )
    sample.to_csv(REPORT_PATH / "test_prediction_plot_sample.csv", index=False)
    visualization_status = "NOT_CREATED"
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Scatter(
                x=sample[LABEL],
                y=sample["prediction"],
                mode="markers",
                name="Deterministic test sample",
                marker={"size": 5, "opacity": 0.5},
            )
        )
        fig.update_layout(
            title="Actual vs predicted trip_count (scikit-learn RF)",
            xaxis_title="Actual trip_count",
            yaxis_title="Predicted trip_count",
            template="plotly_white",
        )
        fig.write_html(str(REPORT_PATH / "test_actual_vs_predicted.html"), include_plotlyjs="cdn")
        importance_fig = go.Figure(
            go.Bar(
                x=[r["importance"] for r in reversed(importance_rows)],
                y=[r["feature"] for r in reversed(importance_rows)],
                orientation="h",
            )
        )
        importance_fig.update_layout(
            title="Random Forest feature importance (scikit-learn, source features)",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_white",
        )
        importance_fig.write_html(str(REPORT_PATH / "feature_importance.html"), include_plotlyjs="cdn")
        visualization_status = "CREATED"
    except Exception as exc:
        visualization_status = f"NOT_CREATED: {type(exc).__name__}: {exc}"

    summary = {
        "status": "VALIDATION_PASSED",
        "model": "sklearn.ensemble.RandomForestRegressor",
        "dataset": str(DATASET_PATH),
        "row_counts": {"training": training_rows, "validation": EXPECTED_ROWS["validation"], "testing": EXPECTED_ROWS["testing"]},
        "target": LABEL,
        "features": CATEGORY_COLS + NUMERIC_COLS,
        "categorical_encoding": "OneHotEncoder(handle_unknown=ignore), fit on training only",
        "training_category_counts": category_counts,
        "split": "Jan-Apr training; May validation; Jun testing",
        "time_basis": "UTC epoch key; local calendar fields in America/New_York; retrospective same-hour estimate using actual weather",
        "parameters": {"n_estimators": 50, "max_depth": 8, "random_state": 42, "n_jobs": 2},
        "implementation_difference": "Spark uses nominal StringIndexer metadata; sklearn uses sparse one-hot encoding. Category variables/features and splits are the same, but categorical split representation differs.",
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "memory": {
            "available_before_training_bytes": mem_available_before,
            "process_peak_rss_bytes": peak_rss_bytes(),
            "spark_comparable_peak_memory_available": False,
        },
        "metrics": metric_rows,
        "visualization_status": visualization_status,
        "model_path": str(RUN_ROOT / "model.joblib"),
        "reports_path": str(REPORT_PATH),
        "predictions_path": str(PREDICTION_PATH),
    }
    (RUN_ROOT / "model.joblib").parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipeline, RUN_ROOT / "model.joblib", compress=3)
    (REPORT_PATH / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Model: {RUN_ROOT / 'model.joblib'}")
    print(f"Training rows: {training_rows:,}; categories: {category_counts}")
    for row in metric_rows:
        print(
            f"{row['split']}: rows={row['rows']:,}, MAE={row['MAE']:.6f}, "
            f"RMSE={row['RMSE']:.6f}, R2={row['R2']:.6f}, "
            f"prediction_seconds={row['prediction_seconds']:.3f}"
        )
    print(f"Peak RSS bytes: {summary['memory']['process_peak_rss_bytes']}")
    print(f"Reports: {REPORT_PATH}; visualization: {visualization_status}")


if __name__ == "__main__":
    main()
