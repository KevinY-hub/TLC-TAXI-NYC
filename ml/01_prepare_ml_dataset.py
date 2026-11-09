"""Prepare the validated hourly HVFHV demand table from existing Spark outputs.

This script reads the compact Stage 3 zone-hour aggregate and hourly weather
Parquet only. It does not rescan trip-level Parquet or modify earlier stages.
"""

from __future__ import annotations

import csv
import json
from functools import reduce
from operator import and_, or_
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZONE_HOUR_PATH = PROJECT_ROOT / "output" / "analysis" / "weather_exploratory" / "pickup_zone_hour_weather_volume.parquet"
WEATHER_PATH = PROJECT_ROOT / "data" / "reference" / "weather_openmeteo" / "weather_hourly_parquet"
ML_ROOT = PROJECT_ROOT / "output" / "machine_learning"
DATASET_PATH = ML_ROOT / "dataset"
REPORT_PATH = ML_ROOT / "reports" / "dataset_preparation"

WEATHER_COLUMNS = [
    "weather_temperature_2m",
    "weather_relative_humidity_2m",
    "weather_precipitation",
    "weather_wind_speed_10m",
    "weather_code",
]
NUMERIC_WEATHER_COLUMNS = [c for c in WEATHER_COLUMNS if c != "weather_code"]
EXPECTED_MATCHED_TRIPS = 126_309_541
EXPECTED_MATCHED_ZONE_HOURS = 1_085_715


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Laporan tidak ditimpa: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require_columns(frame: DataFrame, expected: set[str], name: str) -> None:
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada {name}: {sorted(missing)}")


def main() -> None:
    if not ZONE_HOUR_PATH.is_dir():
        raise FileNotFoundError(f"Agregat zona-jam tidak ditemukan: {ZONE_HOUR_PATH}")
    if not WEATHER_PATH.is_dir():
        raise FileNotFoundError(f"Parquet cuaca tidak ditemukan: {WEATHER_PATH}")
    if DATASET_PATH.exists():
        raise FileExistsError(f"Dataset output sudah ada; tidak ditimpa: {DATASET_PATH}")
    if REPORT_PATH.exists():
        raise FileExistsError(f"Direktori laporan output sudah ada; tidak ditimpa: {REPORT_PATH}")

    spark = (
        SparkSession.builder.appName("HVFHVPrepareHourlyMLDataset")
        .master("local[2]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        zone_hour_all = spark.read.parquet(str(ZONE_HOUR_PATH))
        require_columns(
            zone_hour_all,
            {"PULocationID", "pickup_hour_utc_epoch", "weather_category", "trip_count"},
            "pickup_zone_hour_weather_volume",
        )
        zone_hour_source_rows = zone_hour_all.count()
        source_trip_rows = zone_hour_all.agg(F.sum("trip_count").alias("n")).first()["n"]
        null_utc_groups = zone_hour_all.filter(F.col("pickup_hour_utc_epoch").isNull())
        null_utc_group_rows = null_utc_groups.count()
        null_utc_trip_count = null_utc_groups.agg(F.sum("trip_count").alias("n")).first()["n"] or 0

        # Use matched weather categories only. The unavailable category includes
        # trips with no valid geometry/weather match and the DST-invalid pickups.
        matched_groups = zone_hour_all.filter(
            (F.col("weather_category") != F.lit("weather_unavailable"))
            & F.col("pickup_hour_utc_epoch").isNotNull()
            & F.col("PULocationID").isNotNull()
        )

        category_conflicts = (
            matched_groups.groupBy("PULocationID", "pickup_hour_utc_epoch")
            .agg(F.countDistinct("weather_category").alias("n_categories"))
            .filter(F.col("n_categories") > 1)
            .count()
        )
        if category_conflicts:
            raise ValueError(
                f"{category_conflicts} zona-jam memiliki lebih dari satu kategori cuaca; "
                "pembuatan target dihentikan agar trip_count tidak terduplikasi."
            )

        target = matched_groups.groupBy("PULocationID", "pickup_hour_utc_epoch").agg(
            F.sum("trip_count").cast("long").alias("trip_count")
        )
        target_rows = target.count()
        target_trip_count = target.agg(F.sum("trip_count").alias("n")).first()["n"] or 0
        if target_rows != EXPECTED_MATCHED_ZONE_HOURS:
            raise ValueError(
                f"Jumlah kombinasi zona-jam valid {target_rows:,}; "
                f"laporan audit sebelumnya mencatat {EXPECTED_MATCHED_ZONE_HOURS:,}."
            )
        if target_trip_count != EXPECTED_MATCHED_TRIPS:
            raise ValueError(
                f"Jumlah trip target {target_trip_count:,}; "
                f"laporan integrasi cuaca mencatat {EXPECTED_MATCHED_TRIPS:,}."
            )

        weather = spark.read.parquet(str(WEATHER_PATH))
        require_columns(
            weather,
            {
                "zone_id",
                "weather_timestamp_epoch",
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "weather_code",
            },
            "weather_hourly_parquet",
        )
        weather_keys = weather.select(
            F.col("zone_id").cast("int").alias("PULocationID"),
            F.col("weather_timestamp_epoch").cast("long").alias("pickup_hour_utc_epoch"),
            F.col("temperature_2m").cast("double").alias("weather_temperature_2m"),
            F.col("relative_humidity_2m").cast("double").alias("weather_relative_humidity_2m"),
            F.col("precipitation").cast("double").alias("weather_precipitation"),
            F.col("wind_speed_10m").cast("double").alias("weather_wind_speed_10m"),
            F.col("weather_code").cast("int").alias("weather_code"),
        )
        duplicate_weather_keys = (
            weather_keys.groupBy("PULocationID", "pickup_hour_utc_epoch")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
        if duplicate_weather_keys:
            raise ValueError(f"Weather key duplikat ditemukan: {duplicate_weather_keys:,}")

        joined = target.join(weather_keys, ["PULocationID", "pickup_hour_utc_epoch"], "left")
        weather_key_misses = joined.filter(F.col("weather_temperature_2m").isNull()).count()
        null_exprs = [
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in WEATHER_COLUMNS
        ]
        null_values = joined.agg(*null_exprs).first().asDict()
        invalid_weather_rows = joined.filter(
            reduce(
                or_,
                [F.col(c).isNull() for c in WEATHER_COLUMNS],
            )
        ).count()

        # A left join preserves counts for auditing; only rows with complete,
        # valid weather features enter the explicitly requested weather model.
        valid_weather = joined.filter(
            reduce(
                and_,
                [F.col(c).isNotNull() for c in WEATHER_COLUMNS],
            )
        )
        valid_rows = valid_weather.count()
        valid_trip_count = valid_weather.agg(F.sum("trip_count").alias("n")).first()["n"] or 0

        utc_ts = F.expr("timestamp_seconds(pickup_hour_utc_epoch)")
        local_ts = F.from_utc_timestamp(utc_ts, "America/New_York")
        model_data = valid_weather.select(
            "PULocationID",
            "pickup_hour_utc_epoch",
            F.hour(local_ts).cast("int").alias("pickup_hour_local"),
            F.dayofweek(local_ts).cast("int").alias("day_of_week"),
            F.month(local_ts).cast("int").alias("pickup_month"),
            F.when(F.dayofweek(local_ts).isin(1, 7), F.lit(1)).otherwise(F.lit(0))
            .cast("int")
            .alias("weekend_indicator"),
            "weather_temperature_2m",
            "weather_precipitation",
            "weather_relative_humidity_2m",
            "weather_wind_speed_10m",
            "weather_code",
            "trip_count",
            F.date_format(local_ts, "yyyy-MM").alias("pickup_month_local"),
        )

        model_nulls = model_data.agg(
            *[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in model_data.columns]
        ).first().asDict()
        if any((value or 0) > 0 for value in model_nulls.values()):
            raise ValueError(f"Null ditemukan pada kolom dataset model: {model_nulls}")

        duplicate_model_keys = (
            model_data.groupBy("PULocationID", "pickup_hour_utc_epoch")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
        if duplicate_model_keys:
            raise ValueError(f"Kunci zona-jam duplikat pada dataset model: {duplicate_model_keys:,}")

        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        model_data.write.mode("errorifexists").partitionBy("pickup_month_local").parquet(str(DATASET_PATH))

        readback = spark.read.parquet(str(DATASET_PATH))
        readback_rows = readback.count()
        readback_trip_count = readback.agg(F.sum("trip_count").alias("n")).first()["n"] or 0
        readback_duplicate_keys = (
            readback.groupBy("PULocationID", "pickup_hour_utc_epoch")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
        if (readback_rows, readback_trip_count, readback_duplicate_keys) != (
            valid_rows,
            valid_trip_count,
            0,
        ):
            raise ValueError(
                "Validasi readback gagal: "
                f"rows={readback_rows:,}, trip_count={readback_trip_count:,}, "
                f"duplicate_keys={readback_duplicate_keys:,}."
            )

        splits = []
        for row in (
            readback.groupBy("pickup_month_local")
            .agg(F.count(F.lit(1)).alias("zone_hour_rows"), F.sum("trip_count").alias("trip_count"))
            .orderBy("pickup_month_local")
            .toLocalIterator()
        ):
            month = row["pickup_month_local"]
            if month <= "2026-04":
                split = "training"
            elif month == "2026-05":
                split = "validation"
            else:
                split = "testing"
            splits.append(
                {
                    "pickup_month_local": month,
                    "split": split,
                    "zone_hour_rows": row["zone_hour_rows"],
                    "trip_count": row["trip_count"],
                }
            )

        summary_rows = [
            {"metric": "zone_hour_source_rows", "value": zone_hour_source_rows},
            {"metric": "source_trip_count", "value": source_trip_rows},
            {"metric": "null_utc_group_rows", "value": null_utc_group_rows},
            {"metric": "null_utc_trip_count", "value": null_utc_trip_count},
            {"metric": "weather_category_conflicts", "value": category_conflicts},
            {"metric": "matched_target_zone_hour_rows", "value": target_rows},
            {"metric": "matched_target_trip_count", "value": target_trip_count},
            {"metric": "weather_key_duplicates", "value": duplicate_weather_keys},
            {"metric": "weather_key_misses", "value": weather_key_misses},
            {"metric": "rows_with_any_weather_null", "value": invalid_weather_rows},
            {"metric": "weather_null_weather_temperature_2m", "value": null_values.get("weather_temperature_2m") or 0},
            {"metric": "weather_null_weather_relative_humidity_2m", "value": null_values.get("weather_relative_humidity_2m") or 0},
            {"metric": "weather_null_weather_precipitation", "value": null_values.get("weather_precipitation") or 0},
            {"metric": "weather_null_weather_wind_speed_10m", "value": null_values.get("weather_wind_speed_10m") or 0},
            {"metric": "weather_null_weather_code", "value": null_values.get("weather_code") or 0},
            {"metric": "valid_model_zone_hour_rows", "value": valid_rows},
            {"metric": "valid_model_trip_count", "value": valid_trip_count},
            {"metric": "excluded_trips_without_weather", "value": source_trip_rows - valid_trip_count},
            {"metric": "readback_rows", "value": readback_rows},
            {"metric": "readback_trip_count", "value": readback_trip_count},
            {"metric": "readback_duplicate_zone_hour_keys", "value": readback_duplicate_keys},
            {"metric": "validation_status", "value": "VALIDATION_PASSED"},
        ]
        REPORT_PATH.mkdir(parents=True, exist_ok=True)
        write_csv(REPORT_PATH / "dataset_preparation_summary.csv", summary_rows, ["metric", "value"])
        write_csv(
            REPORT_PATH / "dataset_split_counts.csv",
            splits,
            ["pickup_month_local", "split", "zone_hour_rows", "trip_count"],
        )
        manifest = {
            "status": "VALIDATION_PASSED",
            "input_zone_hour_aggregate": str(ZONE_HOUR_PATH),
            "input_weather_parquet": str(WEATHER_PATH),
            "output_dataset": str(DATASET_PATH),
            "row_unit": "PULocationID x one UTC hour",
            "target": "trip_count",
            "utc_key": "pickup_hour_utc_epoch",
            "local_timezone_features": "America/New_York",
            "row_count": readback_rows,
            "trip_count_sum": readback_trip_count,
            "zero_demand_rows_added": 0,
            "weather_missing_policy": "Only rows with valid weather features are retained for this weather-based model.",
            "dst_invalid_trip_count_excluded": 85,
            "other_unmatched_weather_trip_count_excluded": source_trip_rows - valid_trip_count - 85,
            "weather_key_duplicates": duplicate_weather_keys,
            "readback_duplicate_zone_hour_keys": readback_duplicate_keys,
        }
        manifest_path = REPORT_PATH / "dataset_manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"Manifest tidak ditimpa: {manifest_path}")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"ML dataset: {DATASET_PATH}")
        print(f"Zone-hour rows: {readback_rows:,}")
        print(f"Trip count total: {readback_trip_count:,}")
        print(f"Rows excluded for unavailable/null weather: {source_trip_rows - valid_trip_count:,}")
        print(f"Weather key misses: {weather_key_misses:,}; duplicate weather keys: {duplicate_weather_keys:,}")
        print(f"Validation: VALIDATION_PASSED; reports: {REPORT_PATH}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
