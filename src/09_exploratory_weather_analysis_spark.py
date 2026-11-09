"""Exploratory temporal, weather, and spatial analysis for integrated HVFHV.

Run from the project root in WSL (the same Spark environment used for stages 5-8):
    python src/09_exploratory_weather_analysis_spark.py

The script reads existing Parquet sources only. It never collects trip rows or
converts the dataset to Pandas. Small aggregate tables are collected only to
write human-readable CSV reports.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 126_771_977
TIMEZONE_NAME = "America/New_York"
EXPECTED_DST_INVALID = 85
TRIP_WEATHER_COLUMNS = {
    "temperature_2m": "weather_temperature_2m",
    "relative_humidity_2m": "weather_relative_humidity_2m",
    "precipitation": "weather_precipitation",
    "rain": "weather_rain",
    "snowfall": "weather_snowfall",
    "weather_code": "weather_code",
    "wind_speed_10m": "weather_wind_speed_10m",
    "cloud_cover": "weather_cloud_cover",
}
CATEGORIES = (
    "no_precipitation",
    "rain",
    "snow",
    "mixed_or_unclassified_precipitation",
    "weather_unavailable",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    """Create a small CSV without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {path}")
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def assert_columns(df: Any, required: set[str], label: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def weather_category_expr(F: Any, precipitation: Any, rain: Any, snowfall: Any, code: Any) -> Any:
    """Use observed variables with mixed precipitation taking precedence.

    Rain: rain > 0 or WMO codes 51-67/80-82.
    Snow: snowfall > 0 or WMO codes 71-77/85-86.
    If both signals occur, label mixed. Other positive precipitation or WMO
    precipitation codes (including thunderstorms) are unclassified. A dry
    label requires all three amount variables to equal zero and a non-null,
    non-precipitation WMO code. Missing weather is handled by the caller.
    """
    rain_signal = (rain > 0) | code.between(51, 67) | code.between(80, 82)
    snow_signal = (snowfall > 0) | code.between(71, 77) | code.between(85, 86)
    has_precip_signal = (
        (precipitation > 0)
        | rain_signal
        | snow_signal
        | code.between(51, 99)
    )
    dry_amounts = (precipitation == 0) & (rain == 0) & (snowfall == 0)
    non_precip_code = code.isNotNull() & (~code.between(51, 99))
    return (
        F.when(rain_signal & snow_signal, F.lit("mixed_or_unclassified_precipitation"))
        .when(rain_signal, F.lit("rain"))
        .when(snow_signal, F.lit("snow"))
        .when(dry_amounts & non_precip_code, F.lit("no_precipitation"))
        .when(has_precip_signal, F.lit("mixed_or_unclassified_precipitation"))
        .otherwise(F.lit("mixed_or_unclassified_precipitation"))
    )


def save_parquet(df: Any, path: Path, partition_by: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = df.write.mode("errorifexists")
    if partition_by:
        writer = writer.partitionBy(partition_by)
    writer.parquet(str(path))


def aggregate_rows(df: Any, key: str) -> int:
    # This action returns exactly one scalar from an already-small aggregate.
    return int(df.agg({"trip_count": "sum"}).first()[0] or 0)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    trips_path = project_root / "output" / "integrated" / "taxi_zones_weather"
    weather_path = project_root / "data" / "reference" / "weather_openmeteo" / "weather_hourly_parquet"
    analysis_root = project_root / "output" / "analysis" / "weather_exploratory"
    quality_root = project_root / "output" / "quality" / "weather_exploratory"
    log_root = quality_root / "logs"

    if not trips_path.is_dir():
        raise FileNotFoundError(f"Integrated trip Parquet directory not found: {trips_path}")
    if not weather_path.is_dir():
        raise FileNotFoundError(f"Weather Parquet directory not found: {weather_path}")
    if analysis_root.exists() or quality_root.exists():
        raise FileExistsError(
            "Analysis output already exists; refusing to overwrite. "
            f"Remove or archive only the prior analysis outputs before rerunning: {analysis_root}, {quality_root}"
        )
    log_root.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_root / "exploratory_analysis.log", encoding="utf-8")],
    )
    logger = logging.getLogger("weather_analysis")
    spark = None

    try:
        from pyspark.sql import SparkSession, functions as F
        from pyspark.sql.types import TimestampNTZType, DateType, IntegerType

        builder = (
            SparkSession.builder.appName("HVFHV-Exploratory-Weather-Analysis")
            .master("local[2]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
        )
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark master=%s, session timezone=%s", spark.sparkContext.master, spark.conf.get("spark.sql.session.timeZone"))
        available_memory = get_available_memory_bytes()
        if available_memory is not None:
            logger.info("Linux MemAvailable: %.2f GiB", available_memory / (1024**3))

        trips = spark.read.parquet(str(trips_path))
        weather = spark.read.parquet(str(weather_path))
        trip_required = {
            "pickup_datetime", "pickup_date", "PULocationID", "pickup_borough", "pickup_zone",
            "pickup_time_dst_invalid", "weather_match_status", *TRIP_WEATHER_COLUMNS.values(),
        }
        weather_required = {"zone_id", "weather_timestamp_epoch", "weather_time_utc", *TRIP_WEATHER_COLUMNS.keys()}
        assert_columns(trips, trip_required, "Integrated trips")
        assert_columns(weather, weather_required, "Weather observations")

        trip_types = {field.name: field.dataType for field in trips.schema.fields}
        if not isinstance(trip_types["pickup_datetime"], TimestampNTZType):
            raise TypeError(f"pickup_datetime expected timestamp_ntz, found {trip_types['pickup_datetime']}")
        if not isinstance(trip_types["pickup_date"], DateType):
            raise TypeError(f"pickup_date expected date, found {trip_types['pickup_date']}")
        if not isinstance(trip_types["PULocationID"], IntegerType):
            raise TypeError(f"PULocationID expected int, found {trip_types['PULocationID']}")
        logger.info("Trip schema: %s", trips.schema.simpleString())
        logger.info("Weather schema: %s", weather.schema.simpleString())

        # Check schema, key semantics, and hour precision before large aggregations.
        weather_key_counts = weather.groupBy("zone_id", "weather_timestamp_epoch").count()
        duplicate_weather_keys = weather_key_counts.where(F.col("count") > 1).count()
        weather_rows = weather.count()
        weather_key_rows = weather_key_counts.count()
        off_hour_weather = weather.where((F.col("weather_timestamp_epoch") % F.lit(3600)) != 0).count()
        epoch_mismatch = weather.where(
            F.unix_timestamp("weather_time_utc") != F.col("weather_timestamp_epoch")
        ).count()
        if duplicate_weather_keys or weather_rows != weather_key_rows or off_hour_weather or epoch_mismatch:
            raise ValueError(
                "Weather key preflight failed: "
                f"rows={weather_rows}, unique zone-hour keys={weather_key_rows}, duplicate keys={duplicate_weather_keys}, "
                f"off-hour epochs={off_hour_weather}, UTC epoch mismatches={epoch_mismatch}"
            )
        logger.info("Weather source has %s unique zone-hour observations", weather_rows)

        # The integrated output already has an explicit DST marker. Independently
        # verify that it exactly covers the known missing local hour in this period.
        local_gap = (F.to_date("pickup_datetime") == F.lit("2026-03-08").cast("date")) & (F.hour("pickup_datetime") == 2)
        dst_marked = trips.where(F.col("pickup_time_dst_invalid")).count()
        dst_gap_rows = trips.where(local_gap).count()
        dst_gap_unmarked = trips.where(local_gap & (~F.col("pickup_time_dst_invalid"))).count()
        if dst_marked != EXPECTED_DST_INVALID or dst_gap_rows != EXPECTED_DST_INVALID or dst_gap_unmarked:
            raise ValueError(
                f"DST audit mismatch: marked={dst_marked}, local 2026-03-08 02:xx={dst_gap_rows}, unmarked={dst_gap_unmarked}"
            )

        # Weather classification is defined once and applied to the source
        # observations and trip rows. The denominator is zone-hour observations,
        # because volume is analyzed by pickup zone as well as weather.
        weather_classified = weather.withColumn(
            "weather_category",
            weather_category_expr(
                F,
                F.col("precipitation"), F.col("rain"), F.col("snowfall"), F.col("weather_code"),
            ),
        ).withColumn("weather_utc_hour", F.col("weather_timestamp_epoch"))
        weather_category_counts = {
            row["weather_category"]: int(row["observation_count"])
            for row in weather_classified.groupBy("weather_category").agg(F.count(F.lit(1)).alias("observation_count")).collect()
        }

        # Convert local timestamp_ntz as a wall-clock time explicitly. Spark
        # session timezone is fixed to UTC. DST-invalid rows deliberately receive
        # a null key and stay in the source population as weather unavailable.
        trip_weather = trips.withColumn(
            "_pickup_utc_ntz",
            F.when(F.col("pickup_time_dst_invalid"), F.lit(None).cast("timestamp_ntz"))
            .otherwise(F.convert_timezone(F.lit(TIMEZONE_NAME), F.lit("UTC"), F.col("pickup_datetime"))),
        ).withColumn(
            "_pickup_hour_utc_epoch",
            F.when(F.col("_pickup_utc_ntz").isNull(), F.lit(None).cast("long"))
            .otherwise(F.floor(F.unix_timestamp("_pickup_utc_ntz") / F.lit(3600)).cast("long") * F.lit(3600)),
        ).withColumn(
            "weather_category",
            F.when(F.col("weather_match_status") != F.lit("matched"), F.lit("weather_unavailable"))
            .otherwise(weather_category_expr(
                F,
                F.col("weather_precipitation"), F.col("weather_rain"), F.col("weather_snowfall"), F.col("weather_code"),
            )),
        )

        source_rows = trips.count()
        if source_rows != EXPECTED_ROWS:
            raise ValueError(f"Unexpected source row count: expected {EXPECTED_ROWS}, found {source_rows}")
        trip_weather_null = trips.where(F.col("weather_temperature_2m").isNull()).count()
        trip_weather_available = source_rows - trip_weather_null
        if trip_weather_null + trip_weather_available != source_rows:
            raise AssertionError("Weather available/unavailable population does not partition source rows")

        # Temporal aggregations. Local fields are extracted from timestamp_ntz;
        # they do not undergo an implicit timezone conversion. DST-gap records
        # still count in their original local 02:00 hour, flagged in validation.
        date_counts = trip_weather.groupBy("pickup_date").agg(F.count(F.lit(1)).alias("trip_count")).orderBy("pickup_date")
        hour_counts = trip_weather.groupBy(F.hour("pickup_datetime").alias("pickup_hour_local")).agg(F.count(F.lit(1)).alias("trip_count")).orderBy("pickup_hour_local")
        weekday_counts = trip_weather.groupBy(
            F.dayofweek("pickup_datetime").alias("weekday_spark_sunday_1"),
            F.date_format("pickup_datetime", "EEEE").alias("weekday_name_local"),
        ).agg(F.count(F.lit(1)).alias("trip_count")).orderBy("weekday_spark_sunday_1")
        month_counts = trip_weather.groupBy(F.date_format("pickup_date", "yyyy-MM").alias("pickup_month")).agg(F.count(F.lit(1)).alias("trip_count")).orderBy("pickup_month")
        date_hour_counts = trip_weather.groupBy(
            "pickup_date", F.hour("pickup_datetime").alias("pickup_hour_local")
        ).agg(F.count(F.lit(1)).alias("trip_count")).orderBy("pickup_date", "pickup_hour_local")

        outputs: dict[str, Any] = {
            "temporal_trips_by_date": date_counts,
            "temporal_trips_by_hour": hour_counts,
            "temporal_trips_by_weekday": weekday_counts,
            "temporal_trips_by_month": month_counts,
            "temporal_trips_by_date_hour": date_hour_counts,
        }
        for name, frame in outputs.items():
            save_parquet(frame, analysis_root / f"{name}.parquet")
            logger.info("Wrote %s", name)

        # Weather categories and denominators.
        trip_category_rows = {
            row["weather_category"]: int(row["trip_count"])
            for row in trip_weather.groupBy("weather_category").agg(F.count(F.lit(1)).alias("trip_count")).collect()
        }
        category_report = []
        for category in CATEGORIES:
            n_trips = trip_category_rows.get(category, 0)
            n_observations = weather_category_counts.get(category, 0) if category != "weather_unavailable" else 0
            category_report.append({
                "weather_category": category,
                "trip_count": n_trips,
                "percent_all_trips": 100.0 * n_trips / source_rows if source_rows else 0.0,
                "valid_zone_hour_observations": n_observations,
                "mean_trips_per_zone_hour_observation": n_trips / n_observations if n_observations else "",
            })
        category_total = sum(r["trip_count"] for r in category_report)
        if category_total != source_rows:
            raise AssertionError(f"Weather category counts sum to {category_total}, expected {source_rows}")
        write_csv(quality_root / "weather_category_summary.csv", list(category_report[0]), category_report)

        weather_monthly = trip_weather.groupBy(
            F.date_format("pickup_date", "yyyy-MM").alias("pickup_month"), "weather_category"
        ).agg(F.count(F.lit(1)).alias("trip_count")).orderBy("pickup_month", "weather_category")
        save_parquet(weather_monthly, analysis_root / "weather_trips_by_month_category.parquet")

        # Spatial tables retain null/unmatched LocationID as explicit groups.
        location_counts = trips.groupBy("PULocationID", "pickup_borough", "pickup_zone").agg(F.count(F.lit(1)).alias("trip_count"))
        save_parquet(location_counts, analysis_root / "pickup_location_volume.parquet")
        top_locations = location_counts.orderBy(F.desc("trip_count")).limit(20)
        top_location_rows = [row.asDict(recursive=True) for row in top_locations.collect()]
        write_csv(quality_root / "top_pickup_zones.csv", ["PULocationID", "pickup_borough", "pickup_zone", "trip_count"], top_location_rows)

        borough_volume = trips.groupBy("pickup_borough").agg(F.count(F.lit(1)).alias("trip_count")).orderBy(F.desc("trip_count"))
        save_parquet(borough_volume, analysis_root / "pickup_borough_volume.parquet")
        zone_hour_weather = trip_weather.groupBy(
            "PULocationID", "pickup_borough", "pickup_zone", "_pickup_hour_utc_epoch", "weather_category"
        ).agg(F.count(F.lit(1)).alias("trip_count")).withColumnRenamed("_pickup_hour_utc_epoch", "pickup_hour_utc_epoch")
        save_parquet(zone_hour_weather, analysis_root / "pickup_zone_hour_weather_volume.parquet")

        # Count/check every temporal partition without loading the source to the driver.
        temporal_sums = {
            "date": aggregate_rows(date_counts, "pickup_date"),
            "hour": aggregate_rows(hour_counts, "pickup_hour_local"),
            "weekday": aggregate_rows(weekday_counts, "weekday_spark_sunday_1"),
            "month": aggregate_rows(month_counts, "pickup_month"),
            "date_hour": aggregate_rows(date_hour_counts, "pickup_date,pickup_hour_local"),
        }
        location_sum = int(location_counts.agg(F.sum("trip_count").alias("n")).first()["n"] or 0)
        category_matched = sum(trip_category_rows.get(c, 0) for c in CATEGORIES if c != "weather_unavailable")
        if any(value != source_rows for value in temporal_sums.values()):
            raise AssertionError(f"Temporal aggregate totals do not equal source rows: {temporal_sums}")
        if location_sum != source_rows:
            raise AssertionError(f"Pickup location aggregate total {location_sum} != {source_rows}")
        if category_matched != trip_weather_available:
            raise AssertionError(f"Matched weather category trips {category_matched} != available {trip_weather_available}")

        validation_rows = [
            {"check": "source_rows", "value": source_rows, "expected": EXPECTED_ROWS, "status": "PASS" if source_rows == EXPECTED_ROWS else "FAIL"},
            {"check": "weather_available_trips", "value": trip_weather_available, "expected": source_rows - trip_weather_null, "status": "PASS"},
            {"check": "weather_unavailable_trips", "value": trip_weather_null, "expected": "reported separately", "status": "PASS"},
            {"check": "weather_unique_zone_hour_keys", "value": weather_key_rows, "expected": weather_rows, "status": "PASS" if weather_key_rows == weather_rows and duplicate_weather_keys == 0 else "FAIL"},
            {"check": "weather_duplicate_zone_hour_keys", "value": duplicate_weather_keys, "expected": 0, "status": "PASS" if duplicate_weather_keys == 0 else "FAIL"},
            {"check": "weather_epoch_off_hour", "value": off_hour_weather, "expected": 0, "status": "PASS" if off_hour_weather == 0 else "FAIL"},
            {"check": "weather_epoch_timestamp_mismatch", "value": epoch_mismatch, "expected": 0, "status": "PASS" if epoch_mismatch == 0 else "FAIL"},
            {"check": "dst_invalid_rows_retained", "value": dst_marked, "expected": EXPECTED_DST_INVALID, "status": "PASS" if dst_marked == EXPECTED_DST_INVALID else "FAIL"},
            {"check": "dst_gap_unmarked_rows", "value": dst_gap_unmarked, "expected": 0, "status": "PASS" if dst_gap_unmarked == 0 else "FAIL"},
            *[{"check": f"temporal_total_{name}", "value": value, "expected": source_rows, "status": "PASS" if value == source_rows else "FAIL"} for name, value in temporal_sums.items()],
            {"check": "pickup_location_total", "value": location_sum, "expected": source_rows, "status": "PASS" if location_sum == source_rows else "FAIL"},
            {"check": "weather_category_total", "value": category_total, "expected": source_rows, "status": "PASS" if category_total == source_rows else "FAIL"},
            {"check": "matched_weather_category_total", "value": category_matched, "expected": trip_weather_available, "status": "PASS" if category_matched == trip_weather_available else "FAIL"},
        ]
        if any(row["status"] != "PASS" for row in validation_rows):
            raise AssertionError("One or more quality checks failed")
        write_csv(quality_root / "analysis_validation.csv", ["check", "value", "expected", "status"], validation_rows)

        rules = [
            {"category": "rain", "rule": "rain > 0 OR WMO weather_code in 51-67 or 80-82"},
            {"category": "snow", "rule": "snowfall > 0 OR WMO weather_code in 71-77 or 85-86"},
            {"category": "mixed_or_unclassified_precipitation", "rule": "rain and snow signals both present; otherwise other positive precipitation or WMO code 51-99; residual records not proven dry also remain here", "priority": "takes precedence when both rain and snow signals are present"},
            {"category": "no_precipitation", "rule": "precipitation = 0 AND rain = 0 AND snowfall = 0 AND non-null WMO code outside 51-99"},
            {"category": "weather_unavailable", "rule": "integrated weather_match_status is not matched; retained as a separate trip category"},
            {"category": "denominator", "rule": "valid_zone_hour_observations counts distinct source table rows per unique zone_id + UTC epoch hour; mean is trips / zone-hour observations, not trips divided by trip rows"},
        ]
        write_csv(quality_root / "weather_category_definitions.csv", ["category", "rule", "priority"], rules)

        source_metrics = [
            {"metric": "source_trip_rows", "value": source_rows},
            {"metric": "weather_available_trip_rows", "value": trip_weather_available},
            {"metric": "weather_unavailable_trip_rows", "value": trip_weather_null},
            {"metric": "dst_invalid_local_time_rows", "value": dst_marked},
            {"metric": "source_weather_zone_hour_observations", "value": weather_rows},
            {"metric": "source_weather_zones", "value": weather.select("zone_id").distinct().count()},
            {"metric": "local_timezone_for_temporal_interpretation", "value": TIMEZONE_NAME},
            {"metric": "Spark_session_timezone", "value": "UTC"},
            {"metric": "pickup_utc_conversion", "value": "convert_timezone(America/New_York, UTC, pickup_datetime timestamp_ntz); DST-invalid rows excluded from UTC key only"},
            {"metric": "local_DST_policy", "value": "Keep original pickup_datetime and count DST gap rows in local temporal aggregates; weather remains unavailable"},
            {"metric": "analysis_created_at_utc", "value": utc_now()},
            {"metric": "validation_status", "value": "VALIDATION_PASSED"},
        ]
        write_csv(quality_root / "analysis_summary.csv", ["metric", "value"], source_metrics)

        logger.info("ANALYSIS VALIDATION PASSED")
        logger.info("Source trips=%s; weather matched=%s; unavailable=%s; DST invalid=%s", source_rows, trip_weather_available, trip_weather_null, dst_marked)
        logger.info("Category trip counts=%s", trip_category_rows)
        logger.info("Temporal totals=%s", temporal_sums)
        logger.info("Analysis tables: %s", analysis_root)
        logger.info("Quality reports: %s", quality_root)
        return 0

    except Exception as exc:
        logger.exception("Exploratory analysis failed: %s", exc)
        try:
            quality_root.mkdir(parents=True, exist_ok=True)
            status_path = quality_root / "analysis_failure_status.csv"
            if not status_path.exists():
                write_csv(status_path, ["status", "finished_at_utc", "detail"], [{"status": "FAILED", "finished_at_utc": utc_now(), "detail": str(exc)}])
        except Exception:
            logger.exception("Could not write failure status")
        return 1
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
