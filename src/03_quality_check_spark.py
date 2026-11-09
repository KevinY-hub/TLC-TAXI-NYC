"""Distributed data quality counts for the six NYC TLC HVFHV 2026 files."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable


MONTH_FILES = [
    (f"2026-{month:02d}", f"fhvhv_tripdata_2026-{month:02d}.parquet")
    for month in range(1, 7)
]
NUMERIC_RULES = [
    ("trip_miles", "equals_zero", lambda F, c: c == 0),
    ("trip_miles", "less_than_zero", lambda F, c: c < 0),
    ("trip_time", "equals_zero", lambda F, c: c == 0),
    ("trip_time", "less_than_zero", lambda F, c: c < 0),
    ("base_passenger_fare", "less_than_zero", lambda F, c: c < 0),
    ("driver_pay", "less_than_zero", lambda F, c: c < 0),
    ("driver_pay", "equals_zero", lambda F, c: c == 0),
]
DUPLICATE_COLUMNS = [
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
]


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, object]]) -> None:
    """Stream a small aggregated result to CSV; raw rows never reach the driver."""
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    try:
        from pyspark.sql import SparkSession, functions as F
        from pyspark.sql.window import Window
    except ImportError:
        print(
            "PySpark belum terinstall pada Python environment ini.\n"
            "Dependency yang diperlukan: pyspark.\n"
            "Setelah siap, verifikasi dengan: "
            "python -c \"import pyspark; print(pyspark.__version__)\""
        )
        return 2

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    source_dir = data_dir if data_dir.is_dir() else project_dir
    output_dir = project_dir / "output" / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_files = [name for _, name in MONTH_FILES if not (source_dir / name).is_file()]
    if missing_files:
        print(f"Folder input: {source_dir}")
        print("File Parquet tidak ditemukan:")
        for filename in missing_files:
            print(f"  - {filename}")
        return 1

    spark = (
        SparkSession.builder.appName("HVFHS2026DataQuality")
        # A small local worker count reduces concurrent aggregation memory
        # pressure on typical Windows workstations.
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        monthly_frames = [
            spark.read.parquet(str(source_dir / filename)).withColumn(
                "_source_month", F.lit(month)
            )
            for month, filename in MONTH_FILES
        ]
        df = monthly_frames[0]
        for monthly_frame in monthly_frames[1:]:
            df = df.unionByName(monthly_frame)

        required_columns = {
            "originating_base_num", "trip_miles", "trip_time", "base_passenger_fare",
            "driver_pay", "pickup_datetime", "dropoff_datetime", "PULocationID",
            "DOLocationID", "hvfhs_license_num", "shared_request_flag",
            "shared_match_flag",
        }
        absent_columns = sorted(required_columns.difference(df.columns))
        if absent_columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(absent_columns)}")

        # This aggregation yields only six month-level rows to the driver.
        month_count_rows = [
            {"month": row["_source_month"], "rows": int(row["rows"])}
            for row in (
                df.groupBy("_source_month")
                .agg(F.count(F.lit(1)).alias("rows"))
                .orderBy("_source_month")
                .toLocalIterator()
            )
        ]
        total_rows = sum(row["rows"] for row in month_count_rows)
        row_count_output = [
            {"month": row["month"], "rows": row["rows"]} for row in month_count_rows
        ]
        row_count_output.append({"month": "TOTAL", "rows": total_rows})
        write_csv(
            output_dir / "row_counts.csv",
            ["month", "rows"],
            row_count_output,
        )

        # One distributed pass computes all null and numeric/timestamp/location
        # checks. Only this single aggregated result row is brought to Python.
        expressions = [F.count(F.lit(1)).alias("total_rows")]
        for name in df.columns:
            if name == "_source_month":
                continue
            expressions.append(
                F.sum(F.when(F.col(name).isNull(), 1).otherwise(0)).alias(f"missing__{name}")
            )
        for index, (column, metric, predicate) in enumerate(NUMERIC_RULES):
            expressions.append(
                F.sum(F.when(predicate(F, F.col(column)), 1).otherwise(0)).alias(
                    f"numeric__{index}"
                )
            )
        expressions.extend(
            [
                F.sum(F.when(F.col("pickup_datetime").isNull(), 1).otherwise(0)).alias(
                    "pickup_null"
                ),
                F.sum(F.when(F.col("dropoff_datetime").isNull(), 1).otherwise(0)).alias(
                    "dropoff_null"
                ),
                F.sum(
                    F.when(
                        F.col("dropoff_datetime") < F.col("pickup_datetime"), 1
                    ).otherwise(0)
                ).alias("dropoff_before_pickup"),
                F.sum(
                    F.when(
                        F.col("dropoff_datetime") <= F.col("pickup_datetime"),
                        1,
                    ).otherwise(0)
                ).alias("duration_nonpositive"),
                F.sum(F.when(F.col("PULocationID").isNull(), 1).otherwise(0)).alias(
                    "pu_location_null"
                ),
                F.sum(F.when(F.col("DOLocationID").isNull(), 1).otherwise(0)).alias(
                    "do_location_null"
                ),
            ]
        )
        aggregate = df.agg(*expressions).first()
        denominator = int(aggregate["total_rows"])

        missing_rows = []
        for name in df.columns:
            if name == "_source_month":
                continue
            count = int(aggregate[f"missing__{name}"] or 0)
            missing_rows.append(
                {
                    "column": name,
                    "missing_count": count,
                    "total_rows": denominator,
                    "missing_percent": (count / denominator * 100) if denominator else 0.0,
                }
            )
        write_csv(
            output_dir / "missing_values.csv",
            ["column", "missing_count", "total_rows", "missing_percent"],
            missing_rows,
        )

        numeric_rows = []
        for index, (column, metric, _) in enumerate(NUMERIC_RULES):
            count = int(aggregate[f"numeric__{index}"] or 0)
            numeric_rows.append(
                {
                    "column": column,
                    "check": metric,
                    "count": count,
                    "total_rows": denominator,
                    "percent": (count / denominator * 100) if denominator else 0.0,
                }
            )
        write_csv(
            output_dir / "numeric_quality.csv",
            ["column", "check", "count", "total_rows", "percent"],
            numeric_rows,
        )

        timestamp_checks = [
            ("pickup_datetime", "is_null", "pickup_null"),
            ("dropoff_datetime", "is_null", "dropoff_null"),
            ("dropoff_datetime < pickup_datetime", "is_true", "dropoff_before_pickup"),
            ("trip_duration", "less_than_or_equal_zero", "duration_nonpositive"),
        ]
        timestamp_rows = []
        for column, check, field in timestamp_checks:
            count = int(aggregate[field] or 0)
            timestamp_rows.append(
                {
                    "column": column,
                    "check": check,
                    "count": count,
                    "total_rows": denominator,
                    "percent": (count / denominator * 100) if denominator else 0.0,
                }
            )
        write_csv(
            output_dir / "timestamp_quality.csv",
            ["column", "check", "count", "total_rows", "percent"],
            timestamp_rows,
        )

        location_rows = []
        for column, field in (
            ("PULocationID", "pu_location_null"),
            ("DOLocationID", "do_location_null"),
        ):
            count = int(aggregate[field] or 0)
            location_rows.append(
                {
                    "column": column,
                    "check": "is_null",
                    "count": count,
                    "total_rows": denominator,
                    "percent": (count / denominator * 100) if denominator else 0.0,
                }
            )
        write_csv(
            output_dir / "location_quality.csv",
            ["column", "check", "count", "total_rows", "percent"],
            location_rows,
        )

        def distribution_rows(column: str):
            result = (
                df.groupBy(column)
                .agg(F.count(F.lit(1)).alias("count"))
                .withColumn("percent", F.col("count") * F.lit(100.0 / denominator))
                .orderBy(F.desc("count"), F.col(column).asc_nulls_first())
                .toLocalIterator()
            )
            for row in result:
                category = row[column]
                yield {
                    "column": column,
                    "value": "<NULL>" if category is None else str(category),
                    "count": int(row["count"]),
                    "total_rows": denominator,
                    "percent": float(row["percent"]),
                }

        write_csv(
            output_dir / "provider_distribution.csv",
            ["column", "value", "count", "total_rows", "percent"],
            distribution_rows("hvfhs_license_num"),
        )
        shared_rows = (
            item
            for column in ("shared_request_flag", "shared_match_flag")
            for item in distribution_rows(column)
        )
        write_csv(
            output_dir / "shared_ride_distribution.csv",
            ["column", "value", "count", "total_rows", "percent"],
            shared_rows,
        )

        # Grouping by the requested key is the necessary distributed duplicate
        # aggregation. Only its one-row summary is returned to the driver.
        duplicate_groups = df.groupBy(*DUPLICATE_COLUMNS).agg(
            F.count(F.lit(1)).alias("combination_rows")
        )
        duplicate_summary = duplicate_groups.agg(
            F.count(F.lit(1)).alias("unique_combinations"),
            F.sum(
                F.when(F.col("combination_rows") > 1, F.col("combination_rows") - 1)
                .otherwise(0)
            ).alias("duplicate_rows_excess"),
            F.sum(
                F.when(F.col("combination_rows") > 1, F.col("combination_rows"))
                .otherwise(0)
            ).alias("rows_in_duplicate_groups"),
        ).first()
        write_csv(
            output_dir / "duplicate_check.csv",
            ["total_rows", "unique_combinations", "duplicate_rows_excess", "rows_in_duplicate_groups"],
            [
                {
                    "total_rows": denominator,
                    "unique_combinations": int(duplicate_summary["unique_combinations"] or 0),
                    "duplicate_rows_excess": int(duplicate_summary["duplicate_rows_excess"] or 0),
                    "rows_in_duplicate_groups": int(duplicate_summary["rows_in_duplicate_groups"] or 0),
                }
            ],
        )

        print(f"Selesai. Total rows: {denominator:,}")
        print(f"Output tersimpan di: {output_dir}")
        print("Duplicate check tambahan: duplicate_check.csv")
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
