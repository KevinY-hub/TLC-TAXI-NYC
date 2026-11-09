"""Fit and evaluate the agreed Spark MLlib Random Forest baseline."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession, functions as F
from pyspark.storagelevel import StorageLevel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "output" / "machine_learning" / "dataset"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ROOT = PROJECT_ROOT / "output" / "machine_learning" / "spark_rf" / RUN_ID
MODEL_PATH = RUN_ROOT / "model"
REPORT_PATH = RUN_ROOT / "reports"
PREDICTION_PATHS = {
    "validation": RUN_ROOT / "predictions" / "validation",
    "testing": RUN_ROOT / "predictions" / "testing",
}

LABEL_COL = "trip_count"
KEY_COLS = ["PULocationID", "pickup_hour_utc_epoch"]
CATEGORICAL_SOURCE_COLS = ["PULocationID", "weather_code"]
INDEXED_COLS = ["PULocationID_index", "weather_code_index"]
FEATURE_COLS = [
    "PULocationID_index",
    "pickup_hour_local",
    "day_of_week",
    "pickup_month",
    "weekend_indicator",
    "weather_temperature_2m",
    "weather_precipitation",
    "weather_relative_humidity_2m",
    "weather_wind_speed_10m",
    "weather_code_index",
]
EXPECTED_ROWS = {"training": 718_545, "validation": 186_481, "testing": 180_689}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"File output sudah ada; tidak ditimpa: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_absent(paths: list[Path]) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError("Output berikut sudah ada dan tidak akan ditimpa: " + ", ".join(existing))


def main() -> None:
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(f"Dataset ML tidak ditemukan: {DATASET_PATH}")
    ensure_absent([RUN_ROOT])

    spark = (
        SparkSession.builder.appName("HVFHVRandomForestSparkMLlib")
        .master("local[2]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        data = spark.read.parquet(str(DATASET_PATH))
        required = set(KEY_COLS + [LABEL_COL, "pickup_month_local"] + FEATURE_COLS[1:-1] + CATEGORICAL_SOURCE_COLS)
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Kolom dataset model tidak ditemukan: {sorted(missing)}")

        duplicate_keys = (
            data.groupBy(*KEY_COLS).count().filter(F.col("count") > 1).count()
        )
        if duplicate_keys:
            raise ValueError(f"Dataset model memiliki {duplicate_keys:,} kunci zona-jam duplikat.")
        null_condition = (
            F.col(LABEL_COL).isNull()
            | F.col("PULocationID").isNull()
            | F.col("weather_code").isNull()
            | F.col("pickup_hour_local").isNull()
            | F.col("day_of_week").isNull()
            | F.col("pickup_month").isNull()
            | F.col("weekend_indicator").isNull()
            | F.col("weather_temperature_2m").isNull()
            | F.col("weather_precipitation").isNull()
            | F.col("weather_relative_humidity_2m").isNull()
            | F.col("weather_wind_speed_10m").isNull()
        )
        null_rows = data.filter(null_condition).count()
        if null_rows:
            raise ValueError(f"Terdapat {null_rows:,} baris dengan null pada target/fitur model.")

        training = data.filter(F.col("pickup_month_local").between("2026-01", "2026-04"))
        validation = data.filter(F.col("pickup_month_local") == "2026-05")
        testing = data.filter(F.col("pickup_month_local") == "2026-06")
        split_frames = {"training": training, "validation": validation, "testing": testing}
        split_counts = {name: frame.count() for name, frame in split_frames.items()}
        for name, expected in EXPECTED_ROWS.items():
            if split_counts[name] != expected:
                raise ValueError(
                    f"Jumlah {name} berubah: {split_counts[name]:,}; "
                    f"dataset preparation mencatat {expected:,}."
                )

        training_trip_count = training.agg(F.sum(LABEL_COL).alias("n")).first()["n"]
        validation_trip_count = validation.agg(F.sum(LABEL_COL).alias("n")).first()["n"]
        testing_trip_count = testing.agg(F.sum(LABEL_COL).alias("n")).first()["n"]

        location_categories = training.select("PULocationID").distinct().count()
        weather_code_categories = training.select("weather_code").distinct().count()
        max_bins = max(32, location_categories + 1, weather_code_categories + 1)

        indexers = [
            StringIndexer(
                inputCol="PULocationID",
                outputCol="PULocationID_index",
                handleInvalid="keep",
                stringOrderType="alphabetAsc",
            ),
            StringIndexer(
                inputCol="weather_code",
                outputCol="weather_code_index",
                handleInvalid="keep",
                stringOrderType="alphabetAsc",
            ),
        ]
        assembler = VectorAssembler(
            inputCols=FEATURE_COLS,
            outputCol="features",
            handleInvalid="error",
        )
        preprocessing = Pipeline(stages=indexers + [assembler])

        fit_start = time.perf_counter()
        preprocessing_model = preprocessing.fit(training)
        prepared_training = preprocessing_model.transform(training)
        ml_attr = prepared_training.schema["features"].metadata.get("ml_attr", {})
        nominal_attributes = ml_attr.get("attrs", {}).get("nominal", [])
        nominal_names = {attribute.get("name") for attribute in nominal_attributes}
        if not {"PULocationID_index", "weather_code_index"}.issubset(nominal_names):
            raise ValueError(
                "Spark VectorAssembler tidak mempertahankan metadata kategorikal "
                f"untuk kedua StringIndexer. Nominal attrs ditemukan: {sorted(nominal_names)}"
            )

        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol=LABEL_COL,
            predictionCol="prediction",
            numTrees=50,
            maxDepth=8,
            seed=42,
            maxBins=max_bins,
        )
        rf_model = rf.fit(prepared_training)
        total_fit_seconds = time.perf_counter() - fit_start

        full_model = PipelineModel(stages=preprocessing_model.stages + [rf_model])
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        full_model.write().save(str(MODEL_PATH))

        evaluator_rows: list[dict[str, Any]] = []
        prediction_sample = None
        test_predictions = None
        for split_name, frame in (("validation", validation), ("testing", testing)):
            prediction_frame = (
                rf_model.transform(preprocessing_model.transform(frame))
                .select(
                    *KEY_COLS,
                    "pickup_month_local",
                    F.col(LABEL_COL).cast("double").alias("actual_trip_count"),
                    F.col("prediction").cast("double").alias("prediction"),
                )
                .withColumn("residual", F.col("prediction") - F.col("actual_trip_count"))
                .persist(StorageLevel.MEMORY_AND_DISK)
            )
            predict_start = time.perf_counter()
            prediction_rows = prediction_frame.count()
            prediction_seconds = time.perf_counter() - predict_start
            if prediction_rows != split_counts[split_name]:
                raise ValueError(
                    f"Prediksi {split_name} menghasilkan {prediction_rows:,} baris; "
                    f"seharusnya {split_counts[split_name]:,}."
                )

            metric_values = {}
            for metric_name in ("mae", "rmse", "r2"):
                evaluator = RegressionEvaluator(
                    labelCol="actual_trip_count",
                    predictionCol="prediction",
                    metricName=metric_name,
                )
                metric_values[metric_name.upper()] = float(evaluator.evaluate(prediction_frame))
            evaluator_rows.append(
                {
                    "implementation": "Spark MLlib RandomForestRegressor",
                    "split": split_name,
                    "rows": prediction_rows,
                    "trip_count_sum": {
                        "training": training_trip_count,
                        "validation": validation_trip_count,
                        "testing": testing_trip_count,
                    }[split_name],
                    "MAE": metric_values["MAE"],
                    "RMSE": metric_values["RMSE"],
                    "R2": metric_values["R2"],
                    "fit_seconds": total_fit_seconds,
                    "prediction_materialization_seconds": prediction_seconds,
                }
            )

            if split_name == "testing":
                test_predictions = prediction_frame
                prediction_sample = (
                    prediction_frame.orderBy("pickup_hour_utc_epoch", "PULocationID")
                    .limit(20)
                    .toPandas()
                )
            else:
                output_path = PREDICTION_PATHS[split_name]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                prediction_frame.write.mode("errorifexists").parquet(str(output_path))
                prediction_frame.unpersist()

        assert test_predictions is not None
        test_prediction_path = PREDICTION_PATHS["testing"]
        test_prediction_path.parent.mkdir(parents=True, exist_ok=True)
        test_predictions.write.mode("errorifexists").parquet(str(test_prediction_path))

        quantile_probs = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        residual_quantiles = test_predictions.approxQuantile("residual", quantile_probs, 0.001)
        residual_rows = [
            {"quantile": q, "residual": value}
            for q, value in zip(quantile_probs, residual_quantiles)
        ]

        importance_values = rf_model.featureImportances.toArray().tolist()
        if len(importance_values) != len(FEATURE_COLS):
            raise ValueError(
                f"Jumlah feature importance {len(importance_values)} tidak sama "
                f"dengan jumlah fitur {len(FEATURE_COLS)}."
            )
        importance_rows = [
            {"feature": feature, "importance": float(value)}
            for feature, value in zip(FEATURE_COLS, importance_values)
        ]
        importance_rows.sort(key=lambda row: row["importance"], reverse=True)
        for rank, row in enumerate(importance_rows, start=1):
            row["rank"] = rank

        REPORT_PATH.mkdir(parents=True, exist_ok=True)
        write_csv(
            REPORT_PATH / "model_metrics.csv",
            evaluator_rows,
            [
                "implementation",
                "split",
                "rows",
                "trip_count_sum",
                "MAE",
                "RMSE",
                "R2",
                "fit_seconds",
                "prediction_materialization_seconds",
            ],
        )
        write_csv(
            REPORT_PATH / "feature_importance.csv",
            importance_rows,
            ["rank", "feature", "importance"],
        )
        write_csv(
            REPORT_PATH / "test_residual_quantiles.csv",
            residual_rows,
            ["quantile", "residual"],
        )
        if prediction_sample is not None:
            sample_path = REPORT_PATH / "test_prediction_examples.csv"
            if sample_path.exists():
                raise FileExistsError(f"Sample output tidak ditimpa: {sample_path}")
            prediction_sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

        run_summary = {
            "status": "VALIDATION_PASSED",
            "model": "pyspark.ml.regression.RandomForestRegressor",
            "parameters": {
                "numTrees": 50,
                "maxDepth": 8,
                "seed": 42,
                "maxBins": max_bins,
                "maxBins_reason": "Must cover training categorical cardinality plus the StringIndexer handleInvalid keep bucket.",
            },
            "categorical_cardinality_training": {
                "PULocationID": location_categories,
                "weather_code": weather_code_categories,
                "StringIndexer_handleInvalid": "keep",
                "fit_scope": "training only",
            },
            "feature_metadata_nominal_names": sorted(nominal_names),
            "features": FEATURE_COLS,
            "label": LABEL_COL,
            "time_basis": {
                "UTC_key": "pickup_hour_utc_epoch",
                "calendar_features_timezone": "America/New_York",
                "split": "Jan-Apr training; May validation; Jun testing",
                "actual_hour_weather_note": "Retrospective same-hour estimate; not operational future forecasting.",
            },
            "row_counts": split_counts,
            "trip_count_by_split": {
                "training": training_trip_count,
                "validation": validation_trip_count,
                "testing": testing_trip_count,
            },
            "fit_seconds_total": total_fit_seconds,
            "metrics": evaluator_rows,
            "model_path": str(MODEL_PATH),
            "prediction_paths": {k: str(v) for k, v in PREDICTION_PATHS.items()},
            "reports_path": str(REPORT_PATH),
        }
        summary_path = REPORT_PATH / "run_summary.json"
        if summary_path.exists():
            raise FileExistsError(f"Ringkasan run tidak ditimpa: {summary_path}")
        summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

        try:
            import plotly.graph_objects as go

            importance_fig = go.Figure(
                go.Bar(
                    x=[row["importance"] for row in reversed(importance_rows)],
                    y=[row["feature"] for row in reversed(importance_rows)],
                    orientation="h",
                )
            )
            importance_fig.update_layout(
                title="Random Forest feature importance (Spark MLlib)",
                xaxis_title="Importance",
                yaxis_title="Feature",
                template="plotly_white",
            )
            importance_fig.write_html(str(REPORT_PATH / "feature_importance.html"), include_plotlyjs="cdn")

            plot_sample = (
                test_predictions.filter(
                    F.pmod(F.xxhash64(*[F.col(c) for c in KEY_COLS]), F.lit(100)) < 3
                )
                .limit(5000)
                .toPandas()
            )
            actual_pred_fig = go.Figure(
                go.Scatter(
                    x=plot_sample["actual_trip_count"],
                    y=plot_sample["prediction"],
                    mode="markers",
                    name="Test samples",
                    marker={"size": 5, "opacity": 0.55},
                )
            )
            actual_pred_fig.update_layout(
                title="Actual vs predicted trip_count (deterministic test sample)",
                xaxis_title="Actual trip_count",
                yaxis_title="Predicted trip_count",
                template="plotly_white",
            )
            actual_pred_fig.write_html(str(REPORT_PATH / "test_actual_vs_predicted.html"), include_plotlyjs="cdn")
            visualization_status = "CREATED"
        except Exception as exc:  # Optional charts do not invalidate evaluated metrics.
            visualization_status = f"NOT_CREATED: {type(exc).__name__}: {exc}"
        viz_status_path = REPORT_PATH / "visualization_status.txt"
        if viz_status_path.exists():
            raise FileExistsError(f"Status visualisasi tidak ditimpa: {viz_status_path}")
        viz_status_path.write_text(visualization_status + "\n", encoding="utf-8")

        print(f"Model: {MODEL_PATH}")
        print(f"Spark RF fit seconds: {total_fit_seconds:.3f}")
        for row in evaluator_rows:
            print(
                f"{row['split']}: rows={row['rows']:,}, MAE={row['MAE']:.6f}, "
                f"RMSE={row['RMSE']:.6f}, R2={row['R2']:.6f}, "
                f"predict_seconds={row['prediction_materialization_seconds']:.3f}"
            )
        print(f"Feature importance and metrics: {REPORT_PATH}")
        print(f"Visualization status: {visualization_status}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
