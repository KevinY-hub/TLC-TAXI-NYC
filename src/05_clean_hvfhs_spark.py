"""Apply the finalized HVFHV cleaning rules and write partitioned Parquet."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable


MONTH_FILES = [
    (f"2026-{month:02d}", f"fhvhv_tripdata_2026-{month:02d}.parquet")
    for month in range(1, 7)
]
KEY_COLUMNS = [
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
]


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def percent_change(before: int, difference: int) -> float:
    return (difference / before * 100.0) if before else 0.0


def main() -> int:
    try:
        from pyspark import StorageLevel
        from pyspark.sql import SparkSession, functions as F
    except ImportError:
        print(
            "PySpark belum tersedia pada interpreter ini. Aktifkan .venv lalu "
            "verifikasi dengan: python -c \"import pyspark; print(pyspark.__version__)\""
        )
        return 2

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    source_dir = data_dir if data_dir.is_dir() else project_dir
    output_dir = project_dir / "output" / "cleaned"
    quality_dir = project_dir / "output" / "quality"
    # WSL commonly mounts /tmp as a size-limited tmpfs. Keep Spark shuffle and
    # spill files on the Ubuntu filesystem, which has much more disk capacity.
    spark_local_dir = Path.home() / "spark-local"
    spark_local_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for _, name in MONTH_FILES if not (source_dir / name).is_file()]
    if missing:
        print(f"Folder input: {source_dir}")
        print("File Parquet tidak ditemukan:")
        for name in missing:
            print(f"  - {name}")
        return 1
    if output_dir.exists():
        if any(output_dir.iterdir()):
            print(f"Output sudah berisi data; skrip menolak menimpanya: {output_dir}")
            return 1
        # A prior failed attempt can leave an empty directory before Spark's
        # Hadoop writer starts. Reuse the requested target without deleting data.
        output_dir.rmdir()

    spark = (
        SparkSession.builder.appName("HVFHS2026Cleaning")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "64")
        # Smaller scan partitions reduce peak aggregation memory on local Windows.
        .config("spark.sql.files.maxPartitionBytes", "32m")
        .config("spark.local.dir", str(spark_local_dir))
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        frames = [
            spark.read.parquet(str(source_dir / filename)).withColumn(
                "_source_month", F.lit(month)
            )
            for month, filename in MONTH_FILES
        ]
        raw = frames[0]
        for frame in frames[1:]:
            raw = raw.unionByName(frame)
        source_columns = [name for name in raw.columns if name != "_source_month"]

        required = set(KEY_COLUMNS + ["pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID", "originating_base_num", "hvfhs_license_num"])
        absent = sorted(required.difference(source_columns))
        if absent:
            raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(absent)}")

        pickup = F.col("pickup_datetime")
        dropoff = F.col("dropoff_datetime")
        dropoff_before_pickup = dropoff < pickup
        duration_nonpositive = dropoff <= pickup
        # These two final rules together are equivalent to dropoff <= pickup.
        invalid_timestamp = dropoff_before_pickup | duration_nonpositive

        before_exprs = [
            F.count(F.lit(1)).alias("rows"),
            F.sum(F.when(dropoff_before_pickup, 1).otherwise(0)).alias("dropoff_before_pickup"),
            F.sum(F.when(duration_nonpositive, 1).otherwise(0)).alias("duration_nonpositive"),
            F.sum(F.when(invalid_timestamp, 1).otherwise(0)).alias("invalid_timestamp_union"),
        ]
        for column in (
            "PULocationID",
            "DOLocationID",
            "pickup_datetime",
            "dropoff_datetime",
            "originating_base_num",
        ):
            before_exprs.append(
                F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(f"null__{column}")
            )
        before = raw.agg(*before_exprs).first()
        rows_before = int(before["rows"])
        # Spark comparisons involving null timestamps evaluate to NULL; retain
        # those rows because no final rule authorizes removing them.
        valid = raw.where(F.coalesce(~invalid_timestamp, F.lit(True)))

        def duplicate_components(frame):
            """Return repeated-key groups and exact full-row groups within them."""
            repeated_keys = (
                frame.groupBy(*KEY_COLUMNS)
                .agg(F.count(F.lit(1)).alias("key_count"))
                .where(F.col("key_count") > 1)
                .persist(StorageLevel.MEMORY_AND_DISK)
            )
            data_alias = frame.alias("data")
            key_alias = repeated_keys.alias("keys")
            join_condition = None
            for column in KEY_COLUMNS:
                term = F.col(f"data.{column}").eqNullSafe(F.col(f"keys.{column}"))
                join_condition = term if join_condition is None else join_condition & term
            candidates = data_alias.join(key_alias, join_condition, "inner").select(
                *[F.col(f"data.{column}").alias(column) for column in source_columns],
                F.col("data._source_month").alias("_source_month"),
            )
            full_groups = (
                candidates.groupBy(*source_columns)
                .agg(
                    F.count(F.lit(1)).alias("full_row_count"),
                    F.min("_source_month").alias("_source_month"),
                )
                .persist(StorageLevel.MEMORY_AND_DISK)
            )
            summary = full_groups.agg(
                F.sum(
                    F.when(F.col("full_row_count") > 1, F.col("full_row_count") - 1).otherwise(0)
                ).alias("exact_duplicate_excess"),
                F.count(F.lit(1)).alias("unique_candidate_rows"),
            ).first()
            exact_excess = int(summary["exact_duplicate_excess"] or 0)
            return repeated_keys, full_groups, exact_excess

        repeated_keys_before, exact_groups_before, exact_duplicates_before = duplicate_components(valid)

        key_only = repeated_keys_before.select(*KEY_COLUMNS)
        data_alias = valid.alias("data")
        key_alias = F.broadcast(key_only).alias("keys")
        anti_condition = None
        for column in KEY_COLUMNS:
            term = F.col(f"data.{column}").eqNullSafe(F.col(f"keys.{column}"))
            anti_condition = term if anti_condition is None else anti_condition & term
        non_candidates = data_alias.join(key_alias, anti_condition, "left_anti").select(
            *[F.col(f"data.{column}").alias(column) for column in source_columns],
            F.col("data._source_month").alias("_source_month"),
        )
        # Candidate duplicate groups are small (quality check found 809 rows).
        # Keep one payload-identical copy and preserve one source-month label.
        exact_candidates_unique = exact_groups_before.drop("full_row_count")
        cleaned_with_month = non_candidates.unionByName(exact_candidates_unique)
        cleaned = cleaned_with_month.withColumn("pickup_date", F.to_date("pickup_datetime"))
        cleaned = cleaned.select(*source_columns, "pickup_date")

        # Fail safely if an output appeared after the initial existence check.
        cleaned.write.mode("errorifexists").partitionBy("pickup_date").parquet(str(output_dir))

        # Validate from the written Parquet so the report describes the artifact.
        after_df = spark.read.parquet(str(output_dir))
        after_exprs = [
            F.count(F.lit(1)).alias("rows"),
            F.sum(F.when(F.col("dropoff_datetime") < F.col("pickup_datetime"), 1).otherwise(0)).alias(
                "dropoff_before_pickup"
            ),
            F.sum(F.when(F.col("dropoff_datetime") <= F.col("pickup_datetime"), 1).otherwise(0)).alias(
                "duration_nonpositive"
            ),
            F.sum(
                F.when(
                    (F.col("dropoff_datetime") < F.col("pickup_datetime"))
                    | (F.col("dropoff_datetime") <= F.col("pickup_datetime")),
                    1,
                ).otherwise(0)
            ).alias("invalid_timestamp_union"),
        ]
        for column in (
            "PULocationID",
            "DOLocationID",
            "pickup_datetime",
            "dropoff_datetime",
            "originating_base_num",
        ):
            after_exprs.append(
                F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(f"null__{column}")
            )
        after = after_df.agg(*after_exprs).first()
        rows_after = int(after["rows"])
        rows_deleted = rows_before - rows_after

        repeated_keys_after, exact_groups_after, exact_duplicates_after = duplicate_components(
            after_df.withColumn("_source_month", F.date_format("pickup_date", "yyyy-MM"))
        )
        expected_deleted = int(before["invalid_timestamp_union"] or 0) + exact_duplicates_before
        if rows_deleted != expected_deleted:
            print(
                "PERINGATAN: rows_deleted tidak sama dengan invalid timestamp + "
                f"exact duplicate excess ({rows_deleted} != {expected_deleted})."
            )

        def metric_row(name: str, before_value: int, after_value: int) -> dict[str, object]:
            difference = before_value - after_value
            return {
                "metric": name,
                "before": before_value,
                "after": after_value,
                "difference": difference,
                "percentage": percent_change(before_value, difference),
            }

        summary_rows = [
            metric_row("total_rows", rows_before, rows_after),
            {
                "metric": "rows_deleted",
                "before": 0,
                "after": rows_deleted,
                "difference": rows_deleted,
                "percentage": (rows_deleted / rows_before * 100.0) if rows_before else 0.0,
            },
            metric_row(
                "dropoff_before_pickup",
                int(before["dropoff_before_pickup"] or 0),
                int(after["dropoff_before_pickup"] or 0),
            ),
            metric_row(
                "duration_nonpositive",
                int(before["duration_nonpositive"] or 0),
                int(after["duration_nonpositive"] or 0),
            ),
            metric_row(
                "invalid_timestamp_union",
                int(before["invalid_timestamp_union"] or 0),
                int(after["invalid_timestamp_union"] or 0),
            ),
            metric_row("exact_duplicate_excess_rows", exact_duplicates_before, exact_duplicates_after),
        ]
        for column in (
            "PULocationID",
            "DOLocationID",
            "pickup_datetime",
            "dropoff_datetime",
            "originating_base_num",
        ):
            summary_rows.append(
                metric_row(
                    f"null_{column}",
                    int(before[f"null__{column}"] or 0),
                    int(after[f"null__{column}"] or 0),
                )
            )
        write_csv(
            quality_dir / "cleaning_summary.csv",
            ["metric", "before", "after", "difference", "percentage"],
            summary_rows,
        )

        def grouped_count(frame, group_column: str, output_name: str):
            output = {}
            for row in frame.groupBy(group_column).agg(F.count(F.lit(1)).alias("rows")).toLocalIterator():
                key = row[group_column]
                output["<NULL>" if key is None else str(key)] = int(row["rows"])
            return output

        before_months = grouped_count(raw, "_source_month", "month")
        after_months = grouped_count(
            after_df.withColumn("_validation_month", F.date_format("pickup_date", "yyyy-MM")),
            "_validation_month",
            "month",
        )
        all_months = sorted(set(before_months) | set(after_months))
        monthly_rows = []
        for month in all_months:
            before_count = before_months.get(month, 0)
            after_count = after_months.get(month, 0)
            monthly_rows.append(
                {
                    "month": month,
                    "before": before_count,
                    "after": after_count,
                    "rows_deleted": before_count - after_count,
                    "percentage_deleted": percent_change(before_count, before_count - after_count),
                }
            )
        monthly_rows.append(
            {
                "month": "TOTAL",
                "before": rows_before,
                "after": rows_after,
                "rows_deleted": rows_deleted,
                "percentage_deleted": percent_change(rows_before, rows_deleted),
            }
        )
        write_csv(
            quality_dir / "cleaning_by_month.csv",
            ["month", "before", "after", "rows_deleted", "percentage_deleted"],
            monthly_rows,
        )

        provider_before = grouped_count(raw, "hvfhs_license_num", "provider")
        provider_after = grouped_count(after_df, "hvfhs_license_num", "provider")
        provider_rows = []
        for provider in sorted(set(provider_before) | set(provider_after)):
            b = provider_before.get(provider, 0)
            a = provider_after.get(provider, 0)
            provider_rows.append(
                {
                    "hvfhs_license_num": provider,
                    "before": b,
                    "after": a,
                    "rows_deleted": b - a,
                    "percentage_deleted": percent_change(b, b - a),
                }
            )
        write_csv(
            quality_dir / "cleaning_provider_distribution.csv",
            ["hvfhs_license_num", "before", "after", "rows_deleted", "percentage_deleted"],
            provider_rows,
        )

        print(f"Rows before: {rows_before:,}")
        print(f"Rows after: {rows_after:,}")
        print(f"Rows deleted: {rows_deleted:,}")
        print(f"Percent deleted: {percent_change(rows_before, rows_deleted):.8f}%")
        print(f"Invalid timestamps remaining: {int(after['dropoff_before_pickup'] or 0) + int(after['duration_nonpositive'] or 0)}")
        print(f"Exact duplicate excess rows remaining: {exact_duplicates_after:,}")
        print(f"Cleaned dataset: {output_dir}")
        print(f"Quality reports: {quality_dir}")

        if rows_deleted != expected_deleted or exact_duplicates_after != 0:
            print("VALIDATION GAGAL: hasil tidak sesuai dengan cleaning rules.")
            return 1
        print("Validation passed; process selesai tanpa error fatal.")
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
