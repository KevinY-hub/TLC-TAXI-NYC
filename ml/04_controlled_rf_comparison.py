"""Controlled Spark MLlib vs scikit-learn RF comparison on a shared OHE design."""

from __future__ import annotations

import csv
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
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from pyspark.ml.feature import OneHotEncoder as SparkOneHotEncoder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.pipeline import PipelineModel as SparkPipelineModel
from pyspark.ml.regression import RandomForestRegressor as SparkRandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession, functions as F
from pyspark.storagelevel import StorageLevel


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "output" / "machine_learning" / "dataset"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ROOT = ROOT / "output" / "machine_learning" / "controlled_comparison" / RUN_ID
REPORTS = RUN_ROOT / "reports"
PREDICTIONS = RUN_ROOT / "predictions"
MODELS = RUN_ROOT / "models"

LABEL = "trip_count"
ZONE = "PULocationID"
WEATHER_CODE = "weather_code"
CATS = [ZONE, WEATHER_CODE]
NUMERIC = [
    "pickup_hour_local",
    "day_of_week",
    "pickup_month",
    "weekend_indicator",
    "weather_temperature_2m",
    "weather_precipitation",
    "weather_relative_humidity_2m",
    "weather_wind_speed_10m",
]
KEYS = [ZONE, "pickup_hour_utc_epoch", "pickup_month_local"]
INPUT_COLS = CATS + NUMERIC
READ_COLS = list(dict.fromkeys(INPUT_COLS + [LABEL] + KEYS))
SPLITS = {
    "training": ("2026-01", "2026-02", "2026-03", "2026-04"),
    "validation": ("2026-05",),
    "testing": ("2026-06",),
}
EXPECTED_ROWS = {"training": 718_545, "validation": 186_481, "testing": 180_689}
EXPECTED_TRIPS = {"training": 83_569_154, "validation": 22_045_080, "testing": 20_695_307}
UNKNOWN = "__HVFHS_UNKNOWN_CATEGORY__"
MIN_AVAILABLE_GIB = 6.0


def mem_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def peak_rss_bytes() -> int | None:
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if os.uname().sysname == "Darwin" else value * 1024)
    except (AttributeError, OSError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Tidak menimpa file output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def arrow_split(name: str) -> pd.DataFrame:
    dataset = pads.dataset(str(DATA_PATH), format="parquet", partitioning="hive")
    frame = dataset.to_table(
        columns=READ_COLS,
        filter=pads.field("pickup_month_local").isin(list(SPLITS[name])),
    ).to_pandas()
    if len(frame) != EXPECTED_ROWS[name]:
        raise ValueError(f"{name} rows={len(frame):,}, expected={EXPECTED_ROWS[name]:,}")
    if int(frame[LABEL].sum()) != EXPECTED_TRIPS[name]:
        raise ValueError(f"{name} trip_count sum mismatch")
    if frame[READ_COLS].isna().any().any():
        raise ValueError(f"Null value found in {name} modeling table")
    return frame


def as_category_strings(frame: pd.DataFrame, labels: dict[str, list[str]]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for col in CATS:
        values = frame[col].map(lambda value: str(int(value)) if isinstance(value, (int, np.integer, float, np.floating)) else str(value))
        result[col] = values.where(values.isin(labels[col]), UNKNOWN)
    return result


def sklearn_matrix(
    frame: pd.DataFrame,
    encoder: OneHotEncoder,
    labels: dict[str, list[str]],
) -> sparse.csr_matrix:
    cats = as_category_strings(frame, labels)
    cat_matrix = encoder.transform(cats[CATS])
    numeric_matrix = sparse.csr_matrix(frame[NUMERIC].to_numpy(dtype=np.float64))
    matrix = sparse.hstack([cat_matrix, numeric_matrix], format="csr", dtype=np.float64)
    return matrix


def spark_encoded(frame, index_models, ohe_model, assembler):
    indexed = frame
    for model in index_models:
        indexed = model.transform(indexed)
    encoded = ohe_model.transform(indexed)
    return assembler.transform(encoded)


def compute_sklearn_metrics(split: str, frame: pd.DataFrame, predictions: np.ndarray, seconds: float) -> dict[str, Any]:
    actual = frame[LABEL].to_numpy(dtype=np.float64)
    return {
        "implementation": "Controlled scikit-learn",
        "split": split,
        "rows": len(frame),
        "trip_count_sum": int(frame[LABEL].sum()),
        "MAE": float(mean_absolute_error(actual, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predictions))),
        "R2": float(r2_score(actual, predictions)),
        "prediction_seconds": seconds,
    }


def save_sklearn_predictions(split: str, frame: pd.DataFrame, predictions: np.ndarray) -> None:
    out = pd.DataFrame(
        {
            ZONE: frame[ZONE].to_numpy(),
            "pickup_hour_utc_epoch": frame["pickup_hour_utc_epoch"].to_numpy(),
            "pickup_month_local": frame["pickup_month_local"].to_numpy(),
            "actual_trip_count": frame[LABEL].to_numpy(dtype=np.int64),
            "prediction": predictions.astype(np.float64),
        }
    )
    out["residual"] = out["prediction"] - out["actual_trip_count"]
    path = PREDICTIONS / "sklearn" / f"{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), str(path), compression="zstd")


def main() -> None:
    if not DATA_PATH.is_dir():
        raise FileNotFoundError(f"ML dataset not found: {DATA_PATH}")
    if RUN_ROOT.exists():
        raise FileExistsError(f"Run directory exists; not overwritten: {RUN_ROOT}")
    available_before = mem_available_bytes()
    if available_before is not None and available_before < MIN_AVAILABLE_GIB * 1024**3:
        raise MemoryError(
            f"Available RAM {available_before / 1024**3:.2f} GiB is below "
            f"the {MIN_AVAILABLE_GIB:.1f} GiB safety floor. No training started."
        )

    spark = (
        SparkSession.builder.appName("HVFHSControlledRandomForestComparison")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    spark_metrics: list[dict[str, Any]] = []
    sklearn_metrics: list[dict[str, Any]] = []
    try:
        spark_data = spark.read.parquet(str(DATA_PATH))
        train_spark = spark_data.filter(F.col("pickup_month_local").isin(*SPLITS["training"]))
        for col in CATS:
            train_spark = train_spark.withColumn(col, F.col(col).cast("string"))
        train_spark = train_spark.persist(StorageLevel.MEMORY_AND_DISK)
        train_count = train_spark.count()
        if train_count != EXPECTED_ROWS["training"]:
            raise ValueError(f"Spark training rows={train_count:,}; expected={EXPECTED_ROWS['training']:,}")
        if train_spark.agg(F.sum(LABEL).alias("n")).first()["n"] != EXPECTED_TRIPS["training"]:
            raise ValueError("Spark training target sum mismatch")
        spark_nulls = train_spark.filter(
            F.expr(" OR ".join(f"`{c}` IS NULL" for c in READ_COLS))
        ).count()
        if spark_nulls:
            raise ValueError(f"Spark training rows with nulls: {spark_nulls:,}")

        # All vocabulary is learned exclusively from the training period.
        spark_preprocess_start = time.perf_counter()
        indexer_models = []
        labels: dict[str, list[str]] = {}
        index_cols = []
        for col in CATS:
            index_col = f"{col}_index"
            model = StringIndexer(
                inputCol=col,
                outputCol=index_col,
                handleInvalid="keep",
                stringOrderType="alphabetAsc",
            ).fit(train_spark)
            indexer_models.append(model)
            labels[col] = list(model.labels)
            index_cols.append(index_col)
            if UNKNOWN in labels[col]:
                raise ValueError(f"Reserved unknown token collides with training category {col}")
        expected_categories = {ZONE: 258, WEATHER_CODE: 13}
        actual_categories = {col: len(labels[col]) for col in CATS}
        if actual_categories != expected_categories:
            raise ValueError(f"Training category counts changed: {actual_categories}")

        indexed_train = train_spark
        for indexer_model in indexer_models:
            indexed_train = indexer_model.transform(indexed_train)
        ohe_estimator = SparkOneHotEncoder(
            inputCols=index_cols,
            outputCols=[f"{c}_ohe" for c in CATS],
            handleInvalid="error",
            dropLast=False,
        )
        ohe_model = ohe_estimator.fit(indexed_train)
        spark_preprocessing_fit_seconds = time.perf_counter() - spark_preprocess_start
        sizes = list(ohe_model.categorySizes)
        expected_sizes = [len(labels[col]) + 1 for col in CATS]
        if sizes != expected_sizes:
            raise ValueError(f"Spark OHE sizes {sizes}; expected categories plus unknown bucket {expected_sizes}")
        ohe_cols = [f"{c}_ohe" for c in CATS]
        assembler = VectorAssembler(
            inputCols=ohe_cols + NUMERIC,
            outputCol="features",
            handleInvalid="error",
        )

        # Verify exact category order, unknown handling and vector dimension before RF fitting.
        sample_spark = train_spark.orderBy("pickup_hour_utc_epoch", ZONE).select(*(CATS + NUMERIC)).limit(20)
        unknown_row = sample_spark.limit(1)
        for col in CATS:
            unknown_row = unknown_row.withColumn(col, F.lit(UNKNOWN))
        check_spark = sample_spark.unionByName(unknown_row)
        check_features = spark_encoded(check_spark, indexer_models, ohe_model, assembler)
        spark_vectors = np.vstack([row["features"].toArray() for row in check_features.select("features").collect()])
        sample_pdf = pd.DataFrame([row.asDict() for row in check_spark.collect()])
        categories_for_sklearn = {col: labels[col] + [UNKNOWN] for col in CATS}
        sklearn_encoder = OneHotEncoder(
            categories=[categories_for_sklearn[c] for c in CATS],
            handle_unknown="error",
            drop=None,
            sparse_output=True,
            dtype=np.float64,
        )
        sklearn_encoder.fit(as_category_strings(sample_pdf.iloc[:20], labels)[CATS])
        sklearn_vectors = sklearn_matrix(sample_pdf, sklearn_encoder, labels).toarray()
        if spark_vectors.shape != sklearn_vectors.shape or not np.array_equal(spark_vectors, sklearn_vectors):
            raise ValueError(
                "Spark and scikit-learn encoding sample differs: "
                f"spark={spark_vectors.shape}, sklearn={sklearn_vectors.shape}, "
                f"max_abs_diff={float(np.max(np.abs(spark_vectors - sklearn_vectors)))}"
            )
        feature_names = (
            [f"{ZONE}={v}" for v in categories_for_sklearn[ZONE]]
            + [f"{WEATHER_CODE}={v}" for v in categories_for_sklearn[WEATHER_CODE]]
            + NUMERIC
        )
        feature_dimension = len(feature_names)
        if feature_dimension != 281 or spark_vectors.shape[1] != feature_dimension:
            raise ValueError(f"Unexpected common feature dimension: {feature_dimension}, encoded={spark_vectors.shape}")
        encoding_proof = {
            "status": "IDENTICAL_SAMPLE_ENCODING",
            "training_category_count": actual_categories,
            "unknown_token": UNKNOWN,
            "spark_ohe_sizes": sizes,
            "sklearn_ohe_categories": {c: categories_for_sklearn[c] for c in CATS},
            "feature_order": feature_names,
            "feature_dimension": feature_dimension,
            "checked_rows": int(spark_vectors.shape[0]),
            "includes_unknown_category_row": True,
            "max_absolute_difference": float(np.max(np.abs(spark_vectors - sklearn_vectors))),
            "vocabulary_fit_scope": "training Jan-Apr only",
        }

        rf_spark = SparkRandomForestRegressor(
            featuresCol="features",
            labelCol=LABEL,
            predictionCol="prediction",
            numTrees=50,
            maxDepth=8,
            seed=42,
            featureSubsetStrategy="all",
            impurity="variance",
            maxBins=32,
            minInstancesPerNode=1,
            minInfoGain=0.0,
            minWeightFractionPerNode=0.0,
            subsamplingRate=1.0,
            bootstrap=True,
            maxMemoryInMB=256,
            cacheNodeIds=False,
            checkpointInterval=10,
        )
        fit_start = time.perf_counter()
        rf_model = rf_spark.fit(assembler.transform(ohe_model.transform(indexed_train)))
        spark_fit_seconds = time.perf_counter() - fit_start
        spark_fit_total_seconds = spark_preprocessing_fit_seconds + spark_fit_seconds
        model_spark = SparkPipelineModel(stages=indexer_models + [ohe_model, assembler, rf_model])
        spark_model_path = MODELS / "spark"
        MODELS.mkdir(parents=True, exist_ok=True)
        model_spark.write().save(str(spark_model_path))

        spark_feature_importance = rf_model.featureImportances.toArray().tolist()
        if len(spark_feature_importance) != feature_dimension:
            raise ValueError("Spark importance vector length differs from shared feature dimension")
        spark_importance_rows = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(feature_names, spark_feature_importance)
        ]
        spark_importance_rows.sort(key=lambda row: row["importance"], reverse=True)

        for split in SPLITS:
            frame = spark_data.filter(F.col("pickup_month_local").isin(*SPLITS[split]))
            for col in CATS:
                frame = frame.withColumn(col, F.col(col).cast("string"))
            prediction_start = time.perf_counter()
            transformed = model_spark.transform(frame).select(
                *KEYS,
                F.col(LABEL).cast("double").alias("actual_trip_count"),
                F.col("prediction").cast("double").alias("prediction"),
            ).withColumn("residual", F.col("prediction") - F.col("actual_trip_count")).persist(StorageLevel.MEMORY_AND_DISK)
            materialized = transformed.agg(
                F.count(F.lit(1)).alias("rows"), F.sum("prediction").alias("prediction_sum")
            ).first()
            prediction_seconds = time.perf_counter() - prediction_start
            if materialized["rows"] != EXPECTED_ROWS[split]:
                raise ValueError(f"Spark {split} prediction count mismatch")
            metric_values = {}
            for metric in ("mae", "rmse", "r2"):
                metric_values[metric.upper()] = float(
                    RegressionEvaluator(
                        labelCol="actual_trip_count",
                        predictionCol="prediction",
                        metricName=metric,
                    ).evaluate(transformed)
                )
            target_sum = int(frame.agg(F.sum(LABEL).alias("n")).first()["n"])
            if target_sum != EXPECTED_TRIPS[split]:
                raise ValueError(f"Spark {split} target sum mismatch")
            spark_metrics.append(
                {
                    "implementation": "Controlled Spark MLlib",
                    "split": split,
                    "rows": materialized["rows"],
                    "trip_count_sum": target_sum,
                    "MAE": metric_values["MAE"],
                    "RMSE": metric_values["RMSE"],
                    "R2": metric_values["R2"],
                    "prediction_seconds": prediction_seconds,
                }
            )
            out_path = PREDICTIONS / "spark" / split
            out_path.parent.mkdir(parents=True, exist_ok=True)
            transformed.write.mode("errorifexists").parquet(str(out_path))
            transformed.unpersist()

        train_spark.unpersist()
        spark.stop()

        # scikit-learn reads one chronological partition group at a time.
        sklearn_train = arrow_split("training")
        sklearn_encoder = OneHotEncoder(
            categories=[categories_for_sklearn[c] for c in CATS],
            handle_unknown="error",
            drop=None,
            sparse_output=True,
            dtype=np.float64,
        )
        rf_sklearn = SkRandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            max_features=1.0,
            criterion="squared_error",
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            min_impurity_decrease=0.0,
            bootstrap=True,
            max_samples=None,
            oob_score=False,
            n_jobs=2,
            max_leaf_nodes=None,
            ccp_alpha=0.0,
        )
        sklearn_fit_start = time.perf_counter()
        sklearn_encoder.fit(as_category_strings(sklearn_train, labels)[CATS])
        x_train = sklearn_matrix(sklearn_train, sklearn_encoder, labels)
        rf_sklearn.fit(x_train, sklearn_train[LABEL].to_numpy(dtype=np.float64))
        sklearn_fit_seconds = time.perf_counter() - sklearn_fit_start
        if x_train.shape != (EXPECTED_ROWS["training"], feature_dimension):
            raise ValueError(f"scikit-learn input matrix shape mismatch: {x_train.shape}")
        del x_train, sklearn_train
        gc.collect()

        sklearn_feature_importance = rf_sklearn.feature_importances_.tolist()
        sklearn_importance_rows = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(feature_names, sklearn_feature_importance)
        ]
        sklearn_importance_rows.sort(key=lambda row: row["importance"], reverse=True)

        # One shared inference definition for timing: transform + predict and materialize every prediction.
        for split in SPLITS:
            frame = arrow_split(split)
            x = sklearn_matrix(frame, sklearn_encoder, labels)
            prediction_start = time.perf_counter()
            predictions = rf_sklearn.predict(x)
            prediction_seconds = time.perf_counter() - prediction_start
            row = compute_sklearn_metrics(split, frame, predictions, prediction_seconds)
            sklearn_metrics.append(row)
            save_sklearn_predictions(split, frame, predictions)
            del frame, x, predictions
            gc.collect()

        # Persist models and outputs exclusively in this new run directory.
        sklearn_model_path = MODELS / "sklearn.joblib"
        MODELS.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump({"encoder": sklearn_encoder, "model": rf_sklearn, "labels": labels, "features": feature_names}, sklearn_model_path, compress=3)

        controlled_rows = spark_metrics + sklearn_metrics
        write_csv(
            REPORTS / "controlled_metrics.csv",
            controlled_rows,
            ["implementation", "split", "rows", "trip_count_sum", "MAE", "RMSE", "R2", "prediction_seconds"],
        )
        # Baseline files are read-only inputs to the comparison report.
        baseline_spark = pd.read_csv(ROOT / "output/machine_learning/spark_rf/20261106T194743Z/reports/model_metrics.csv")
        baseline_sklearn = pd.read_csv(ROOT / "output/machine_learning/sklearn_rf/20261106T201549Z/reports/model_metrics.csv")
        baseline_spark["implementation"] = "Baseline Spark MLlib"
        baseline_sklearn["implementation"] = "Baseline scikit-learn"
        baseline_spark = baseline_spark.rename(columns={"prediction_materialization_seconds": "prediction_seconds"})
        baseline_sklearn = baseline_sklearn.rename(columns={"prediction_seconds": "prediction_seconds"})
        comparison = pd.concat(
            [
                baseline_spark[["implementation", "split", "rows", "trip_count_sum", "MAE", "RMSE", "R2", "fit_seconds", "prediction_seconds"]],
                baseline_sklearn[["implementation", "split", "rows", "trip_count_sum", "MAE", "RMSE", "R2", "fit_seconds", "prediction_seconds"]],
                pd.DataFrame([{**r, "fit_seconds": spark_fit_total_seconds if r["implementation"] == "Controlled Spark MLlib" else sklearn_fit_seconds} for r in controlled_rows]),
            ],
            ignore_index=True,
        )
        comparison.to_csv(REPORTS / "baseline_controlled_comparison.csv", index=False)

        write_csv(REPORTS / "spark_feature_importance.csv", spark_importance_rows, ["feature", "importance"])
        write_csv(REPORTS / "sklearn_feature_importance.csv", sklearn_importance_rows, ["feature", "importance"])
        configs = {
            "status": "VALIDATION_PASSED",
            "run_id": RUN_ID,
            "dataset": str(DATA_PATH),
            "splits": {k: list(v) for k, v in SPLITS.items()},
            "expected_rows": EXPECTED_ROWS,
            "expected_trip_count_sum": EXPECTED_TRIPS,
            "target": LABEL,
            "categorical_features": CATS,
            "numeric_features": NUMERIC,
            "encoding_verification": encoding_proof,
            "spark_parameters": {
                "numTrees": 50, "maxDepth": 8, "seed": 42,
                "featureSubsetStrategy": "all", "impurity": "variance",
                "maxBins": 32, "minInstancesPerNode": 1, "minInfoGain": 0.0,
                "minWeightFractionPerNode": 0.0, "bootstrap": True,
                "subsamplingRate": 1.0, "maxMemoryInMB": 256,
                "cacheNodeIds": False, "checkpointInterval": 10,
            },
            "sklearn_parameters": {
                "n_estimators": 50, "max_depth": 8, "random_state": 42,
                "max_features": 1.0, "criterion": "squared_error",
                "min_samples_split": 2, "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.0, "min_impurity_decrease": 0.0,
                "bootstrap": True, "max_samples": None, "oob_score": False,
                "n_jobs": 2, "max_leaf_nodes": None, "ccp_alpha": 0.0,
            },
            "resource": {
                "available_ram_before_run_bytes": available_before,
                "minimum_ram_safety_floor_bytes": int(MIN_AVAILABLE_GIB * 1024**3),
                "dataset_bytes_estimate": sum(p.stat().st_size for p in DATA_PATH.rglob("*.parquet")),
                "sparse_matrix_estimate_training_bytes": EXPECTED_ROWS["training"] * 10 * 16 + (EXPECTED_ROWS["training"] + 1) * 4,
                "peak_rss_python_process_bytes": peak_rss_bytes(),
                "comparable_spark_jvm_peak_rss": "not measured; Python RSS omits Spark JVM process",
            },
            "training_seconds": {"spark_including_preprocessing": spark_fit_total_seconds, "spark_rf_fit_only": spark_fit_seconds, "sklearn_including_encoder_and_rf": sklearn_fit_seconds},
            "prediction_timing_definition": "Spark transform plus distributed action forcing every prediction to be evaluated; sklearn transform plus predict materializing every prediction in the local array. Prediction file writes and metric calculations excluded for both.",
            "metrics": controlled_rows,
            "model_paths": {"spark": str(spark_model_path), "sklearn": str(sklearn_model_path)},
            "predictions_root": str(PREDICTIONS),
            "limitations": [
                "Identical seed values do not imply identical random streams/bootstrap samples across libraries.",
                "Spark maxBins has no direct scikit-learn equivalent; one-hot inputs are binary and Spark maxBins=32 supports them.",
                "Spark variance and sklearn squared_error are aligned regression criteria but tree split implementations can still differ.",
                "Comparable peak memory is unavailable because Spark runs a separate JVM; Python RSS is not a process-tree measurement.",
            ],
        }
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "experiment_configuration.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
        with (REPORTS / "encoding_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(encoding_proof, stream, indent=2)

        report_lines = [
            "# Controlled Random Forest Comparison",
            "",
            f"Run: `{RUN_ID}`",
            "",
            "Both models used the same training-derived one-hot vocabulary and 281-column feature matrix. The sample encoding check included a reserved unknown-category row and produced identical matrices.",
            "",
            "## Controlled metrics",
            "",
            "| Model | Split | Rows | MAE | RMSE | R² | Fit seconds | Predict seconds |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in controlled_rows:
            fit_s = spark_fit_total_seconds if row["implementation"] == "Controlled Spark MLlib" else sklearn_fit_seconds
            report_lines.append(
                f"| {row['implementation']} | {row['split']} | {row['rows']:,} | {row['MAE']:.6f} | {row['RMSE']:.6f} | {row['R2']:.6f} | {fit_s:.3f} | {row['prediction_seconds']:.3f} |"
            )
        report_lines += [
            "",
            "## Interpretation limits",
            "",
            "Seed values do not synchronize random generators across frameworks. Impurity criteria are analogous, not guaranteed bit-for-bit equivalent. Feature importance is reported on the same one-hot columns, but remains impurity-based and should not be interpreted causally. Timing includes model preprocessing and inference but excludes reading the partition data, metric aggregation, and prediction file writing. Spark's JVM peak memory was not measured comparably.",
            "",
            "## Baseline and controlled comparison",
            "",
            "See `baseline_controlled_comparison.csv` for all four runs. Baseline model artifacts and reports were read only.",
        ]
        (REPORTS / "controlled_comparison_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            for model_name in ("Controlled Spark MLlib", "Controlled scikit-learn"):
                rows = [r for r in controlled_rows if r["implementation"] == model_name]
                fig.add_trace(go.Bar(name=model_name, x=[r["split"] for r in rows], y=[r["MAE"] for r in rows]))
            fig.update_layout(title="Controlled Random Forest MAE", xaxis_title="Split", yaxis_title="MAE (trips per zone-hour)", barmode="group")
            fig.write_html(str(REPORTS / "controlled_metrics.html"), include_plotlyjs="cdn")
            imp_fig = go.Figure()
            for model_name, rows in (("Spark MLlib", spark_importance_rows), ("scikit-learn", sklearn_importance_rows)):
                imp_fig.add_trace(go.Bar(name=model_name, x=[r["feature"] for r in rows], y=[r["importance"] for r in rows]))
            imp_fig.update_layout(title="Controlled RF feature importance", xaxis_title="One-hot feature", yaxis_title="Impurity importance", barmode="group")
            imp_fig.write_html(str(REPORTS / "controlled_feature_importance.html"), include_plotlyjs="cdn")
            (REPORTS / "visualization_status.txt").write_text("CREATED\n", encoding="utf-8")
        except Exception as exc:
            (REPORTS / "visualization_status.txt").write_text(f"NOT_CREATED: {type(exc).__name__}: {exc}\n", encoding="utf-8")

        print(f"Run output: {RUN_ROOT}")
        print(f"Encoding verification: {encoding_proof['status']} rows={encoding_proof['checked_rows']} features={feature_dimension} max_diff={encoding_proof['max_absolute_difference']}")
        for row in controlled_rows:
            print(f"{row['implementation']} {row['split']}: n={row['rows']:,}, MAE={row['MAE']:.6f}, RMSE={row['RMSE']:.6f}, R2={row['R2']:.6f}, predict_s={row['prediction_seconds']:.3f}")
        print(f"Spark fit total={spark_fit_total_seconds:.3f}s; sklearn fit={sklearn_fit_seconds:.3f}s")
    finally:
        try:
            spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
