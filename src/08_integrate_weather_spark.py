"""Join hourly Open-Meteo weather to HVFHV pickups without changing source data.

Run from the project root in the existing WSL environment:
    python src/08_integrate_weather_spark.py

The script performs schema, key, DST, sample, and disk preflight checks before
the full left join. It never collects the trip dataset or converts it to Pandas.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TRIP_ROWS = 126_771_977
EXPECTED_WEATHER_ROWS = 1_120_494
EXPECTED_WEATHER_ZONES = 258
EXPECTED_DST_INVALID_ROWS = 85
TIMEZONE_NAME = "America/New_York"
SPRING_DST_DATE = "2026-03-08"
NO_GEOMETRY_LOCATION_IDS = {56, 57, 103, 104, 105, 264, 265}
WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
    "cloud_cover",
)
WEATHER_OUTPUT_NAMES = {
    "temperature_2m": "weather_temperature_2m",
    "relative_humidity_2m": "weather_relative_humidity_2m",
    "precipitation": "weather_precipitation",
    "rain": "weather_rain",
    "snowfall": "weather_snowfall",
    "weather_code": "weather_code",
    "wind_speed_10m": "weather_wind_speed_10m",
    "cloud_cover": "weather_cloud_cover",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a new report atomically; refuse to replace an existing report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Laporan sudah ada; tidak ditimpa: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"File sementara laporan sudah ada: {temporary}")
    try:
        with temporary.open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def update_status(path: Path, status: str, detail: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["status", "finished_at_utc", "detail"])
        writer.writeheader()
        writer.writerow({"status": status, "finished_at_utc": now_utc(), "detail": detail})
    os.replace(temporary, path)


def get_available_memory_bytes() -> int | None:
    """Return Linux MemAvailable when running under WSL/Linux."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def load_lookup_ids(path: Path) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Lookup Taxi Zone tidak ditemukan: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "LocationID" not in reader.fieldnames:
            raise ValueError("Kolom LocationID tidak ada pada Taxi Zone lookup CSV.")
        ids = [int(row["LocationID"]) for row in reader if row.get("LocationID", "").strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("LocationID pada lookup Taxi Zone tidak unik.")
    return set(ids)


def main() -> int:
    try:
        from pyspark.sql import SparkSession, functions as F
    except ImportError as exc:
        print(f"ERROR: PySpark tidak tersedia pada environment ini: {exc}", file=sys.stderr)
        print("Jalankan dari WSL dengan environment proyek: source ~/hvfhs-venv/bin/activate", file=sys.stderr)
        return 2

    project_dir = Path(__file__).resolve().parent.parent
    trips_dir = project_dir / "output" / "integrated" / "taxi_zones"
    weather_dir = project_dir / "data" / "reference" / "weather_openmeteo" / "weather_hourly_parquet"
    lookup_path = project_dir / "data" / "reference" / "taxi_zones" / "taxi_zone_lookup.csv"
    output_dir = project_dir / "output" / "integrated" / "taxi_zones_weather"
    report_dir = project_dir / "output" / "quality" / "weather_integration"
    status_path = report_dir / "weather_integration_status.csv"

    if os.name == "nt":
        print("ERROR: Jalankan script ini dari WSL; Spark Windows gagal pada Hadoop NativeIO.", file=sys.stderr)
        return 2
    if not trips_dir.is_dir() or not weather_dir.is_dir():
        print(f"ERROR: Input tidak lengkap. Perjalanan={trips_dir}; cuaca={weather_dir}", file=sys.stderr)
        return 1
    if output_dir.exists():
        # A user interruption can leave Spark's empty target directory before
        # the first data file is committed. Reuse only this known-empty target;
        # any nonempty/partial Parquet output is protected from overwrite.
        empty_target_from_interrupted_run = (
            output_dir.is_dir()
            and not any(output_dir.iterdir())
            and status_path.is_file()
        )
        if empty_target_from_interrupted_run:
            output_dir.rmdir()
            print(f"Removed the empty output directory left by the interrupted run: {output_dir}")
        else:
            print(f"ERROR: Output sudah ada; tidak ditimpa: {output_dir}", file=sys.stderr)
            return 1
    resume_incomplete_run = False
    if report_dir.exists() and any(report_dir.iterdir()):
        existing_files = sorted(path.name for path in report_dir.iterdir() if path.is_file())
        if existing_files == ["weather_integration_status.csv"] and status_path.is_file():
            with status_path.open("r", newline="", encoding="utf-8-sig") as stream:
                existing_status = next(csv.DictReader(stream), {}).get("status", "")
            resume_incomplete_run = existing_status in {"RUNNING_PREFLIGHT", "RUNNING_FULL_JOIN", "FAILED"}
        if not resume_incomplete_run:
            print(f"ERROR: Folder laporan sudah berisi file; tidak ditimpa: {report_dir}", file=sys.stderr)
            return 1

    lookup_ids = load_lookup_ids(lookup_path)
    trip_bytes = sum(file.stat().st_size for file in trips_dir.rglob("*.parquet"))
    weather_bytes = sum(file.stat().st_size for file in weather_dir.rglob("*.parquet"))
    target_free = shutil.disk_usage(project_dir).free
    scratch_dir = Path.home() / "spark-local-weather-integration"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_free = shutil.disk_usage(scratch_dir).free
    available_memory = get_available_memory_bytes()
    minimum_target_free = max(5 * 1024**3, int(trip_bytes * 1.5))
    minimum_scratch_free = max(2 * 1024**3, int(trip_bytes))

    print(f"Project root: {project_dir}")
    print("Spark master: local[2]; shuffle partitions: 64; max input partition: 32 MiB")
    print(f"Target filesystem free: {target_free / 1024**3:.1f} GiB; input trips: {trip_bytes / 1024**3:.2f} GiB")
    print(f"Spark scratch free: {scratch_free / 1024**3:.1f} GiB; weather input: {weather_bytes / 1024**2:.1f} MiB")
    if available_memory is not None:
        print(f"WSL MemAvailable: {available_memory / 1024**3:.1f} GiB")
    if target_free < minimum_target_free:
        print(
            f"ERROR: Ruang output kurang. Dibutuhkan setidaknya {minimum_target_free / 1024**3:.1f} GiB "
            "(estimasi konservatif dari input dan ruang cadangan).",
            file=sys.stderr,
        )
        return 1
    if scratch_free < minimum_scratch_free:
        print(f"ERROR: Ruang Spark scratch kurang dari {minimum_scratch_free / 1024**3:.1f} GiB.", file=sys.stderr)
        return 1
    if available_memory is not None and available_memory < 4 * 1024**3:
        print("ERROR: Memori tersedia di WSL kurang dari 4 GiB; full join tidak dimulai.", file=sys.stderr)
        return 1

    report_dir.mkdir(parents=True, exist_ok=True)
    if resume_incomplete_run:
        update_status(status_path, "RUNNING_PREFLIGHT", "Melanjutkan run sebelumnya yang berhenti sebelum full join.")
    else:
        write_csv(status_path, ["status", "finished_at_utc", "detail"], [
            {"status": "RUNNING_PREFLIGHT", "finished_at_utc": now_utc(), "detail": "Belum ada full join."}
        ])

    spark = None
    final_status = "FAILED"
    final_detail = "Preflight atau integrasi belum selesai."
    try:
        # WSL may resolve its hostname to a synthetic, unreachable address;
        # local mode must advertise the same loopback interface it binds to.
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
        os.environ["SPARK_LOCAL_DIRS"] = str(scratch_dir)
        spark = (
            SparkSession.builder.appName("HVFHV2026WeatherIntegration")
            .master("local[2]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.files.maxPartitionBytes", "32m")
            .config("spark.sql.autoBroadcastJoinThreshold", "64m")
            .config("spark.local.dir", str(scratch_dir))
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        if spark.conf.get("spark.sql.session.timeZone") != "UTC":
            raise RuntimeError("Spark session timezone tidak berhasil dikunci ke UTC.")

        trips = spark.read.parquet(str(trips_dir))
        weather_all = spark.read.parquet(str(weather_dir))
        required_trip_types = {
            "PULocationID": "int",
            "pickup_datetime": "timestamp_ntz",
            "pickup_date": "date",
        }
        trip_types = {field.name: field.dataType.simpleString() for field in trips.schema.fields}
        for name, expected_type in required_trip_types.items():
            if trip_types.get(name) != expected_type:
                raise ValueError(f"Schema perjalanan {name}={trip_types.get(name)!r}, diharapkan {expected_type!r}.")
        weather_types = {field.name: field.dataType.simpleString() for field in weather_all.schema.fields}
        required_weather_types = {
            "zone_id": "int",
            "weather_timestamp_epoch": "bigint",
            "weather_time_utc": "timestamp",
            "weather_time_local": "string",
        }
        required_weather_types.update({name: ("int" if name == "weather_code" else "double") for name in WEATHER_VARIABLES})
        for name, expected_type in required_weather_types.items():
            if weather_types.get(name) != expected_type:
                raise ValueError(f"Schema cuaca {name}={weather_types.get(name)!r}, diharapkan {expected_type!r}.")
        if "pickup_time_dst_invalid" in trips.columns or "weather_match_status" in trips.columns:
            raise ValueError("Kolom audit cuaca target sudah ada di input; menolak membuat nama kolom bentrok.")
        weather_output_collisions = sorted(set(WEATHER_OUTPUT_NAMES.values()).intersection(trips.columns))
        if weather_output_collisions:
            raise ValueError(f"Kolom cuaca target sudah ada di input: {weather_output_collisions}")

        # Check the explicit LocationID -> zone_id mapping. IDs are read from
        # the project's lookup; no coordinate/order-based inference is used.
        weather_ids = {
            int(row["zone_id"])
            for row in weather_all.select("zone_id").distinct().collect()
            if row["zone_id"] is not None
        }
        if len(weather_ids) != EXPECTED_WEATHER_ZONES:
            raise ValueError(f"Jumlah zone_id cuaca {len(weather_ids)}, diharapkan {EXPECTED_WEATHER_ZONES}.")
        if not weather_ids.issubset(lookup_ids):
            raise ValueError(f"Weather zone_id tidak ada pada Taxi Zone lookup: {sorted(weather_ids - lookup_ids)}")
        no_weather_geometry = lookup_ids - weather_ids
        if no_weather_geometry != NO_GEOMETRY_LOCATION_IDS:
            raise ValueError(
                "ID lookup tanpa data cuaca berbeda dari audit geometri sebelumnya: "
                f"aktual={sorted(no_weather_geometry)}; diharapkan={sorted(NO_GEOMETRY_LOCATION_IDS)}"
            )

        # Verify weather keys are one row per explicit zone and UTC epoch.
        weather_key_issues = weather_all.agg(
            F.count(F.lit(1)).alias("rows"),
            F.sum(F.when(F.col("zone_id").isNull() | F.col("weather_timestamp_epoch").isNull(), 1).otherwise(0)).alias("null_keys"),
            F.sum(F.when(F.pmod(F.col("weather_timestamp_epoch"), F.lit(3600)) != 0, 1).otherwise(0)).alias("off_hour_keys"),
            F.sum(F.when(F.col("weather_time_utc").cast("long") != F.col("weather_timestamp_epoch"), 1).otherwise(0)).alias("epoch_timestamp_mismatch"),
        ).first().asDict()
        duplicate_keys = (
            weather_all.groupBy("zone_id", "weather_timestamp_epoch")
            .count()
            .where(F.col("count") > 1)
            .agg(
                F.count(F.lit(1)).alias("duplicate_combinations"),
                F.coalesce(F.sum(F.col("count") - 1), F.lit(0)).alias("duplicate_excess_rows"),
            )
            .first()
            .asDict()
        )
        weather_row_count = int(weather_key_issues["rows"] or 0)
        duplicate_combinations = int(duplicate_keys["duplicate_combinations"] or 0)
        duplicate_excess_rows = int(duplicate_keys["duplicate_excess_rows"] or 0)
        if weather_row_count != EXPECTED_WEATHER_ROWS:
            raise ValueError(f"Jumlah baris cuaca {weather_row_count}, diharapkan {EXPECTED_WEATHER_ROWS}.")
        if any(int(weather_key_issues[name] or 0) != 0 for name in ("null_keys", "off_hour_keys", "epoch_timestamp_mismatch")):
            raise ValueError(f"Kunci UTC cuaca tidak valid: {weather_key_issues}")
        if duplicate_combinations or duplicate_excess_rows:
            raise ValueError(f"Kunci zona-jam cuaca tidak unik: {duplicate_keys}")

        # Check the known spring-forward gap without converting it to UTC.
        dst_invalid_expr = (
            (F.to_date(F.col("pickup_datetime")) == F.lit(SPRING_DST_DATE).cast("date"))
            & (F.hour(F.col("pickup_datetime")) == 2)
        )
        source_metrics = trips.agg(
            F.count(F.lit(1)).alias("rows_before"),
            F.sum(F.when(dst_invalid_expr, 1).otherwise(0)).alias("dst_invalid_rows"),
            F.min("pickup_date").alias("pickup_date_min"),
            F.max("pickup_date").alias("pickup_date_max"),
        ).first().asDict()
        rows_before = int(source_metrics["rows_before"] or 0)
        dst_invalid_rows = int(source_metrics["dst_invalid_rows"] or 0)
        if rows_before != EXPECTED_TRIP_ROWS:
            raise ValueError(f"Jumlah perjalanan sebelum join {rows_before}, diharapkan {EXPECTED_TRIP_ROWS}.")
        if dst_invalid_rows != EXPECTED_DST_INVALID_ROWS:
            raise ValueError(
                f"Jumlah waktu lokal DST invalid {dst_invalid_rows}, hasil audit sebelumnya {EXPECTED_DST_INVALID_ROWS}; "
                "full join dibatalkan untuk audit ulang."
            )
        if str(source_metrics["pickup_date_min"]) < "2026-01-01" or str(source_metrics["pickup_date_max"]) > "2026-06-30":
            raise ValueError(f"Rentang pickup_date di luar cakupan enam bulan: {source_metrics}")

        # Synthetic DST tests prove the two valid local times map to distinct
        # UTC hours. The nonexistent 02:xx local time is explicitly nulled.
        dst_test = spark.createDataFrame(
            [("2026-03-08 01:30:00",), ("2026-03-08 02:30:00",), ("2026-03-08 03:30:00",)],
            ["local_text"],
        ).withColumn("local_ntz", F.to_timestamp_ntz("local_text"))
        synthetic_invalid = (
            (F.to_date("local_ntz") == F.lit(SPRING_DST_DATE).cast("date"))
            & (F.hour("local_ntz") == 2)
        )
        dst_test = dst_test.withColumn("is_dst_invalid", synthetic_invalid).withColumn(
            "utc_ntz",
            F.when(synthetic_invalid, F.lit(None).cast("timestamp_ntz")).otherwise(
                F.convert_timezone(F.lit(TIMEZONE_NAME), F.lit("UTC"), F.col("local_ntz"))
            ),
        )
        dst_test_rows = dst_test.select("local_text", "is_dst_invalid", "utc_ntz").collect()
        dst_test_map = {row["local_text"]: (row["is_dst_invalid"], str(row["utc_ntz"]) if row["utc_ntz"] is not None else None) for row in dst_test_rows}
        if dst_test_map != {
            "2026-03-08 01:30:00": (False, "2026-03-08 06:30:00"),
            "2026-03-08 02:30:00": (True, None),
            "2026-03-08 03:30:00": (False, "2026-03-08 07:30:00"),
        }:
            raise ValueError(f"Uji konversi timezone/DST gagal: {dst_test_map}")

        # Confirm several actual normal pickup -> weather pairs before the
        # full join; at most twelve rows cross to the driver.
        sample_partition = trips_dir / "pickup_date=2026-02-07"
        if not sample_partition.is_dir():
            raise FileNotFoundError(f"Partisi sampel audit tidak ditemukan: {sample_partition}")
        sample_trips = spark.read.parquet(str(sample_partition))
        sample_dst = (
            (F.to_date("pickup_datetime") == F.lit(SPRING_DST_DATE).cast("date"))
            & (F.hour("pickup_datetime") == 2)
        )
        sample_keyed = (
            sample_trips.select("pickup_datetime", "PULocationID", "pickup_latitude", "pickup_longitude")
            .where(F.col("pickup_datetime").isNotNull() & F.col("PULocationID").isNotNull() & ~sample_dst)
            .sample(False, 0.0001, seed=807)
            .limit(12)
            .withColumn("_utc_ntz", F.convert_timezone(F.lit(TIMEZONE_NAME), F.lit("UTC"), F.col("pickup_datetime")))
            .withColumn("_utc_epoch", F.col("_utc_ntz").cast("timestamp").cast("long"))
            .withColumn("_hour_epoch", (F.floor(F.col("_utc_epoch") / F.lit(3600)) * F.lit(3600)).cast("long"))
        )
        sample_weather = weather_all.select(
            F.col("zone_id").alias("_zone_id"),
            F.col("weather_timestamp_epoch").alias("_weather_epoch"),
            F.from_unixtime(F.col("weather_timestamp_epoch")).alias("weather_time_utc_display"),
            F.col("weather_time_local"),
            *[F.col(name) for name in WEATHER_VARIABLES],
        )
        sample_matches = sample_keyed.join(
            sample_weather,
            (sample_keyed.PULocationID == sample_weather._zone_id)
            & (sample_keyed._hour_epoch == sample_weather._weather_epoch),
            "inner",
        ).select(
            "pickup_datetime",
            F.col("_utc_ntz").alias("pickup_utc_ntz"),
            F.col("_hour_epoch").alias("pickup_hour_utc_epoch"),
            "PULocationID",
            "pickup_latitude",
            "pickup_longitude",
            F.col("_weather_epoch").alias("weather_timestamp_epoch_utc"),
            "weather_time_utc_display",
            "weather_time_local",
            *[F.col(name) for name in WEATHER_VARIABLES],
        ).limit(12)
        sample_rows = [row.asDict(recursive=True) for row in sample_matches.collect()]
        if len(sample_rows) < 5:
            raise ValueError(f"Hanya {len(sample_rows)} pasangan normal ditemukan pada sampel; join penuh dibatalkan.")
        for row in sample_rows:
            if int(row["pickup_hour_utc_epoch"]) != int(row["weather_timestamp_epoch_utc"]):
                raise ValueError(f"Sampel pickup dan weather tidak berada pada jam UTC yang sama: {row}")

        schema_rows = []
        for field in trips.schema.fields:
            schema_rows.append({"dataset": "taxi_zones", "column": field.name, "spark_type": field.dataType.simpleString(), "nullable": field.nullable})
        for field in weather_all.schema.fields:
            schema_rows.append({"dataset": "weather", "column": field.name, "spark_type": field.dataType.simpleString(), "nullable": field.nullable})
        sample_headers = [
            "pickup_datetime", "pickup_utc_ntz", "pickup_hour_utc_epoch", "PULocationID",
            "pickup_latitude", "pickup_longitude", "weather_timestamp_epoch_utc", "weather_time_utc_display", "weather_time_local",
            *WEATHER_VARIABLES,
        ]
        print(f"Preflight passed: rows={rows_before:,}; weather_rows={weather_row_count:,}; weather duplicate excess={duplicate_excess_rows}.")
        print(f"DST invalid pickup rows (retained, no UTC key): {dst_invalid_rows:,}")
        print(f"Weather zones without Taxi Zone representative geometry: {sorted(no_weather_geometry)}")
        print(f"Sample normal pairs validated: {len(sample_rows)}; timezone conversion test passed.")

        # Set a null key on DST-gap rows before conversion. UTC timestamp
        # casting is deterministic because the Spark session timezone is UTC.
        trip_keyed = (
            trips.withColumn("pickup_time_dst_invalid", dst_invalid_expr.cast("boolean"))
            .withColumn(
                "_pickup_utc_ntz",
                F.when(F.col("pickup_time_dst_invalid"), F.lit(None).cast("timestamp_ntz")).otherwise(
                    F.convert_timezone(F.lit(TIMEZONE_NAME), F.lit("UTC"), F.col("pickup_datetime"))
                ),
            )
            .withColumn("_pickup_utc_epoch", F.col("_pickup_utc_ntz").cast("timestamp").cast("long"))
            .withColumn(
                "_pickup_hour_utc_epoch",
                (F.floor(F.col("_pickup_utc_epoch") / F.lit(3600)) * F.lit(3600)).cast("long"),
            )
        )
        weather_keys = weather_all.select(
            F.col("zone_id").alias("_weather_zone_id"),
            F.col("weather_timestamp_epoch").alias("_weather_hour_utc_epoch"),
            F.lit(1).alias("_weather_match_marker"),
            *[F.col(name) for name in WEATHER_VARIABLES],
        )
        joined = trip_keyed.join(
            weather_keys,
            (trip_keyed.PULocationID == weather_keys._weather_zone_id)
            & (trip_keyed._pickup_hour_utc_epoch == weather_keys._weather_hour_utc_epoch),
            "left",
        )
        no_geometry_cond = F.col("PULocationID").isin(*sorted(NO_GEOMETRY_LOCATION_IDS))
        known_location_cond = F.col("PULocationID").isin(*sorted(lookup_ids))
        weather_found = F.col("_weather_match_marker").isNotNull()
        match_status = (
            F.when(F.col("pickup_time_dst_invalid"), F.lit("dst_invalid_local_time"))
            .when(F.col("PULocationID").isNull(), F.lit("null_pickup_location_id"))
            .when(no_geometry_cond, F.lit("zone_without_weather_geometry"))
            .when(~known_location_cond, F.lit("unknown_pickup_location_id"))
            .when(F.col("pickup_datetime").isNull(), F.lit("null_pickup_datetime"))
            .when(~weather_found, F.lit("no_weather_hour_match"))
            .otherwise(F.lit("matched"))
        )
        output_columns = [F.col(name) for name in trips.columns]
        output_columns.extend(
            [
                F.col("pickup_time_dst_invalid"),
                match_status.alias("weather_match_status"),
                *[F.col(source).alias(target) for source, target in WEATHER_OUTPUT_NAMES.items()],
            ]
        )
        output_frame = joined.select(*output_columns)
        # Metrics are computed from the same public status column that will be
        # persisted, so no internal join marker leaks into the final dataset.
        is_matched = F.col("weather_match_status") == F.lit("matched")

        metrics_expr = [
            F.count(F.lit(1)).alias("rows_after_join"),
            F.sum(F.when(is_matched, 1).otherwise(0)).alias("matched_rows"),
            F.sum(F.when(~is_matched, 1).otherwise(0)).alias("unmatched_rows"),
            F.sum(F.when(F.col("pickup_time_dst_invalid"), 1).otherwise(0)).alias("dst_invalid_rows"),
            F.sum(F.when(no_geometry_cond, 1).otherwise(0)).alias("zone_without_geometry_rows"),
            F.sum(F.when(F.col("pickup_time_dst_invalid") & no_geometry_cond, 1).otherwise(0)).alias("dst_and_no_geometry_overlap"),
            F.sum(F.when(F.col("PULocationID").isNull(), 1).otherwise(0)).alias("null_pickup_location_rows"),
            F.sum(F.when(F.col("PULocationID").isNotNull() & ~known_location_cond, 1).otherwise(0)).alias("unknown_pickup_location_rows"),
            F.sum(F.when(F.col("pickup_datetime").isNull(), 1).otherwise(0)).alias("null_pickup_datetime_rows"),
            F.sum(F.when((~is_matched) & (~F.col("pickup_time_dst_invalid")) & (~no_geometry_cond) & F.col("PULocationID").isNotNull() & F.col("pickup_datetime").isNotNull() & known_location_cond, 1).otherwise(0)).alias("other_no_weather_hour_rows"),
        ]
        metrics_expr.extend(
            F.sum(F.when(F.col(WEATHER_OUTPUT_NAMES[variable]).isNull(), 1).otherwise(0)).alias(f"null_{WEATHER_OUTPUT_NAMES[variable]}")
            for variable in WEATHER_VARIABLES
        )
        # The joined frame includes the final weather aliases for the null scan.
        metrics = output_frame.agg(*metrics_expr).first().asDict()
        rows_after_join = int(metrics["rows_after_join"] or 0)
        if rows_after_join != rows_before:
            raise ValueError(f"Join mengubah jumlah baris: before={rows_before}; after={rows_after_join}.")
        if int(metrics["matched_rows"] or 0) + int(metrics["unmatched_rows"] or 0) != rows_before:
            raise ValueError("Jumlah matched + unmatched tidak sama dengan row count sebelum join.")
        if int(metrics["dst_invalid_rows"] or 0) != dst_invalid_rows:
            raise ValueError("Jumlah kasus DST setelah join berbeda dari audit input.")
        update_status(status_path, "RUNNING_FULL_JOIN", "Preflight lolos; full left join dan penulisan Parquet sedang berjalan.")

        if output_dir.exists():
            raise FileExistsError(f"Output muncul selama proses; tidak ditimpa: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        # Hash partitioning by date limits output file count without caching the
        # 126M-row input. Shuffle/spill uses the checked WSL scratch filesystem.
        output_frame.repartition(64, "pickup_date").write.mode("errorifexists").partitionBy("pickup_date").parquet(str(output_dir))

        saved = spark.read.parquet(str(output_dir))
        saved_metrics_expr = [
            F.count(F.lit(1)).alias("rows_after_readback"),
            F.sum(F.when(F.col("weather_match_status") == "matched", 1).otherwise(0)).alias("matched_rows"),
            F.sum(F.when(F.col("weather_match_status") != "matched", 1).otherwise(0)).alias("unmatched_rows"),
            F.sum(F.when(F.col("pickup_time_dst_invalid"), 1).otherwise(0)).alias("dst_invalid_rows"),
        ]
        saved_metrics_expr.extend(
            F.sum(F.when(F.col(WEATHER_OUTPUT_NAMES[variable]).isNull(), 1).otherwise(0)).alias(f"null_{WEATHER_OUTPUT_NAMES[variable]}")
            for variable in WEATHER_VARIABLES
        )
        saved_metrics = saved.agg(*saved_metrics_expr).first().asDict()
        rows_after_readback = int(saved_metrics["rows_after_readback"] or 0)
        if rows_after_readback != rows_before or rows_after_readback != rows_after_join:
            raise ValueError(
                f"Validasi readback row count gagal: before={rows_before}; joined={rows_after_join}; readback={rows_after_readback}."
            )
        if int(saved_metrics["matched_rows"] or 0) != int(metrics["matched_rows"] or 0):
            raise ValueError("Jumlah pasangan cuaca berubah setelah penulisan Parquet.")
        if int(saved_metrics["dst_invalid_rows"] or 0) != EXPECTED_DST_INVALID_ROWS:
            raise ValueError("Jumlah penanda DST pada Parquet hasil baca ulang tidak sesuai.")

        expected_schema = {field.name: field.dataType.simpleString() for field in output_frame.schema.fields}
        saved_schema = {field.name: field.dataType.simpleString() for field in saved.schema.fields}
        if expected_schema != saved_schema:
            missing = sorted(set(expected_schema) - set(saved_schema))
            extra = sorted(set(saved_schema) - set(expected_schema))
            mismatched = {name: (expected_schema[name], saved_schema[name]) for name in expected_schema.keys() & saved_schema.keys() if expected_schema[name] != saved_schema[name]}
            raise ValueError(f"Schema Parquet readback berbeda: missing={missing}, extra={extra}, type_mismatch={mismatched}")

        unmatched_reason_rows = [
            {"reason": "dst_invalid_local_time", "rows": int(metrics["dst_invalid_rows"] or 0), "percent_total": 100 * int(metrics["dst_invalid_rows"] or 0) / rows_before},
            {"reason": "zone_without_weather_geometry", "rows": int(metrics["zone_without_geometry_rows"] or 0), "percent_total": 100 * int(metrics["zone_without_geometry_rows"] or 0) / rows_before},
            {"reason": "dst_and_no_geometry_overlap", "rows": int(metrics["dst_and_no_geometry_overlap"] or 0), "percent_total": 100 * int(metrics["dst_and_no_geometry_overlap"] or 0) / rows_before},
            {"reason": "null_pickup_location_id", "rows": int(metrics["null_pickup_location_rows"] or 0), "percent_total": 100 * int(metrics["null_pickup_location_rows"] or 0) / rows_before},
            {"reason": "unknown_pickup_location_id", "rows": int(metrics["unknown_pickup_location_rows"] or 0), "percent_total": 100 * int(metrics["unknown_pickup_location_rows"] or 0) / rows_before},
            {"reason": "null_pickup_datetime", "rows": int(metrics["null_pickup_datetime_rows"] or 0), "percent_total": 100 * int(metrics["null_pickup_datetime_rows"] or 0) / rows_before},
            {"reason": "other_no_weather_hour_match", "rows": int(metrics["other_no_weather_hour_rows"] or 0), "percent_total": 100 * int(metrics["other_no_weather_hour_rows"] or 0) / rows_before},
            {"reason": "all_unmatched_rows_exclusive_total", "rows": int(metrics["unmatched_rows"] or 0), "percent_total": 100 * int(metrics["unmatched_rows"] or 0) / rows_before},
        ]
        null_rows = [
            {
                "variable": WEATHER_OUTPUT_NAMES[variable],
                "null_rows": int(saved_metrics[f"null_{WEATHER_OUTPUT_NAMES[variable]}"] or 0),
                "percent_total": 100 * int(saved_metrics[f"null_{WEATHER_OUTPUT_NAMES[variable]}"] or 0) / rows_before,
            }
            for variable in WEATHER_VARIABLES
        ]
        report_schema_rows = [
            {"dataset": "taxi_zones_weather", "column": field.name, "spark_type": field.dataType.simpleString(), "nullable": field.nullable}
            for field in saved.schema.fields
        ]
        write_csv(report_dir / "weather_integration_schema.csv", ["dataset", "column", "spark_type", "nullable"], schema_rows + report_schema_rows)
        write_csv(report_dir / "weather_join_samples.csv", sample_headers, sample_rows)
        write_csv(report_dir / "weather_nulls_by_variable.csv", ["variable", "null_rows", "percent_total"], null_rows)
        write_csv(report_dir / "weather_unmatched_reasons.csv", ["reason", "rows", "percent_total"], unmatched_reason_rows)

        summary = [
            {"metric": "status", "value": "VALIDATION_PASSED"},
            {"metric": "rows_before_join", "value": rows_before},
            {"metric": "rows_after_join", "value": rows_after_join},
            {"metric": "rows_after_parquet_readback", "value": rows_after_readback},
            {"metric": "rows_changed_by_join", "value": rows_after_join - rows_before},
            {"metric": "matched_trips", "value": int(saved_metrics["matched_rows"] or 0)},
            {"metric": "unmatched_trips", "value": int(saved_metrics["unmatched_rows"] or 0)},
            {"metric": "dst_invalid_pickups_retained", "value": int(saved_metrics["dst_invalid_rows"] or 0)},
            {"metric": "trips_with_zone_without_geometry", "value": int(metrics["zone_without_geometry_rows"] or 0)},
            {"metric": "overlap_dst_and_zone_without_geometry", "value": int(metrics["dst_and_no_geometry_overlap"] or 0)},
            {"metric": "null_pickup_location_id_rows", "value": int(metrics["null_pickup_location_rows"] or 0)},
            {"metric": "unknown_pickup_location_id_rows", "value": int(metrics["unknown_pickup_location_rows"] or 0)},
            {"metric": "null_pickup_datetime_rows", "value": int(metrics["null_pickup_datetime_rows"] or 0)},
            {"metric": "other_no_weather_hour_match_rows", "value": int(metrics["other_no_weather_hour_rows"] or 0)},
            {"metric": "weather_rows", "value": weather_row_count},
            {"metric": "weather_unique_zone_hour_keys", "value": weather_row_count - duplicate_excess_rows},
            {"metric": "weather_duplicate_combinations", "value": duplicate_combinations},
            {"metric": "weather_duplicate_excess_rows", "value": duplicate_excess_rows},
            {"metric": "weather_hour_key_off_hour_rows", "value": int(weather_key_issues["off_hour_keys"] or 0)},
            {"metric": "weather_utc_epoch_timestamp_mismatches", "value": int(weather_key_issues["epoch_timestamp_mismatch"] or 0)},
            {"metric": "weather_zones", "value": len(weather_ids)},
            {"metric": "lookup_zones_without_weather_geometry", "value": ";".join(map(str, sorted(no_weather_geometry)))},
            {"metric": "timezone_conversion", "value": "convert_timezone(America/New_York, UTC, pickup_datetime timestamp_ntz); Spark session timezone explicitly UTC"},
            {"metric": "join_key", "value": "PULocationID = zone_id AND floor(valid pickup UTC epoch / 3600) * 3600 = weather_timestamp_epoch"},
            {"metric": "dst_rule", "value": "2026-03-08 local 02:xx retained; UTC hour key null; pickup_time_dst_invalid=true"},
            {"metric": "output_path", "value": str(output_dir)},
            {"metric": "output_partition", "value": "pickup_date"},
            {"metric": "weather_source_reused", "value": str(weather_dir)},
            {"metric": "spark_master", "value": "local[2]"},
            {"metric": "spark_session_timezone", "value": spark.conf.get("spark.sql.session.timeZone")},
        ]
        write_csv(report_dir / "weather_integration_summary.csv", ["metric", "value"], summary)
        final_detail = (
            f"Rows before/after/readback={rows_before}/{rows_after_join}/{rows_after_readback}; "
            f"matched={int(saved_metrics['matched_rows'] or 0)}; unmatched={int(saved_metrics['unmatched_rows'] or 0)}; "
            f"DST invalid retained={int(saved_metrics['dst_invalid_rows'] or 0)}; schema and Parquet readback validated."
        )
        update_status(status_path, "VALIDATION_PASSED", final_detail)
        final_status = "VALIDATION_PASSED"
        print("\nWeather integration validation passed.")
        print(f"Rows before/after/readback: {rows_before:,} / {rows_after_join:,} / {rows_after_readback:,}")
        print(f"Weather matches: {int(saved_metrics['matched_rows'] or 0):,}")
        print(f"Without weather: {int(saved_metrics['unmatched_rows'] or 0):,}")
        print(f"DST-invalid retained: {int(saved_metrics['dst_invalid_rows'] or 0):,}")
        print(f"Zone-without-geometry trips: {int(metrics['zone_without_geometry_rows'] or 0):,}; overlap with DST: {int(metrics['dst_and_no_geometry_overlap'] or 0):,}")
        print(f"Output: {output_dir}")
        print(f"Reports: {report_dir}")
        return 0
    except Exception as exc:
        final_detail = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: Weather integration failed: {final_detail}", file=sys.stderr)
        traceback.print_exc()
        if report_dir.exists() and status_path.exists():
            try:
                update_status(status_path, final_status, final_detail)
            except OSError as status_error:
                print(f"ERROR: Status report juga gagal diperbarui: {status_error}", file=sys.stderr)
        return 1
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
