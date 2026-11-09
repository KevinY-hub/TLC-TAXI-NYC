"""Investigate HVFHV anomalies with bounded examples and Spark aggregates."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable


TOTAL_ROWS = 126_771_982
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
STAT_COLUMNS = ["base_passenger_fare", "driver_pay", "tips", "trip_miles", "trip_time"]
EXAMPLE_COLUMNS = [
    "hvfhs_license_num",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "driver_pay",
    "tips",
]


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, object]]) -> None:
    """Stream a bounded result or small aggregate to CSV."""
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(count: int) -> float:
    return count / TOTAL_ROWS * 100.0


def main() -> int:
    try:
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
    output_dir = project_dir / "output" / "quality" / "anomaly_investigation"
    output_dir.mkdir(parents=True, exist_ok=True)

    absent_files = [name for _, name in MONTH_FILES if not (source_dir / name).is_file()]
    if absent_files:
        print(f"Folder input: {source_dir}")
        print("File tidak ditemukan:")
        for name in absent_files:
            print(f"  - {name}")
        return 1

    spark = (
        SparkSession.builder.appName("HVFHS2026AnomalyInvestigation")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "32")
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
        df = frames[0]
        for frame in frames[1:]:
            df = df.unionByName(frame)

        required = set(KEY_COLUMNS + STAT_COLUMNS + ["hvfhs_license_num"])
        missing_columns = sorted(required.difference(df.columns))
        if missing_columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing_columns)}")
        original_columns = [name for name in df.columns if name != "_source_month"]

        conditions = {
            "negative_base_passenger_fare": F.col("base_passenger_fare") < 0,
            "zero_trip_miles": F.col("trip_miles") == 0,
            "zero_trip_time": F.col("trip_time") == 0,
            "zero_trip_miles_and_time": (F.col("trip_miles") == 0) & (F.col("trip_time") == 0),
            "zero_miles_positive_time": (F.col("trip_miles") == 0) & (F.col("trip_time") > 0),
            "positive_miles_zero_time": (F.col("trip_miles") > 0) & (F.col("trip_time") == 0),
            "negative_driver_pay": F.col("driver_pay") < 0,
            "zero_driver_pay": F.col("driver_pay") == 0,
            "pickup_datetime_null": F.col("pickup_datetime").isNull(),
            "dropoff_datetime_null": F.col("dropoff_datetime").isNull(),
            "dropoff_before_pickup": F.col("dropoff_datetime") < F.col("pickup_datetime"),
            "duration_nonpositive": F.col("dropoff_datetime") <= F.col("pickup_datetime"),
            "duration_over_24_hours": F.col("dropoff_datetime") > F.col("pickup_datetime") + F.expr("INTERVAL 24 HOURS"),
            "duration_over_48_hours": F.col("dropoff_datetime") > F.col("pickup_datetime") + F.expr("INTERVAL 48 HOURS"),
        }

        # A single distributed aggregate obtains global anomaly counts and
        # conditional statistics. The driver receives only this one-row result.
        expressions = [F.count(F.lit(1)).alias("actual_total_rows")]
        for name, condition in conditions.items():
            expressions.append(
                F.sum(F.when(condition, 1).otherwise(0)).alias(f"count__{name}")
            )

        stat_specs = {
            "negative_base_passenger_fare": ["base_passenger_fare"],
            "zero_trip_miles": ["base_passenger_fare", "driver_pay", "tips"],
            "zero_trip_time": ["base_passenger_fare", "driver_pay", "tips"],
            "negative_driver_pay": ["base_passenger_fare", "tips", "trip_miles", "trip_time"],
        }
        for category, columns in stat_specs.items():
            condition = conditions[category]
            for column in columns:
                selected = F.when(condition, F.col(column))
                prefix = f"stat__{category}__{column}__"
                expressions.extend(
                    [
                        F.min(selected).alias(prefix + "min"),
                        F.percentile_approx(selected, 0.5, 10_000).alias(prefix + "median_approx"),
                        F.avg(selected).alias(prefix + "mean"),
                        F.max(selected).alias(prefix + "max"),
                    ]
                )

        negative_fare_condition = conditions["negative_base_passenger_fare"]
        for column in ("trip_miles", "trip_time", "driver_pay", "tips"):
            expressions.extend(
                [
                    F.sum(
                        F.when(negative_fare_condition & F.col(column).isNotNull(), 1).otherwise(0)
                    ).alias(f"negative_fare__{column}__non_null"),
                    F.sum(
                        F.when(negative_fare_condition & F.col(column).isNull(), 1).otherwise(0)
                    ).alias(f"negative_fare__{column}__null"),
                ]
            )

        result = df.agg(*expressions).first()
        actual_total = int(result["actual_total_rows"])

        main_categories = [
            "negative_base_passenger_fare",
            "zero_trip_miles",
            "zero_trip_time",
            "zero_trip_miles_and_time",
            "zero_miles_positive_time",
            "positive_miles_zero_time",
            "negative_driver_pay",
            "zero_driver_pay",
            "pickup_datetime_null",
            "dropoff_datetime_null",
            "dropoff_before_pickup",
            "duration_nonpositive",
            "duration_over_24_hours",
            "duration_over_48_hours",
        ]
        main_rows = []
        for category in main_categories:
            count = int(result[f"count__{category}"] or 0)
            main_rows.append(
                {
                    "category": category,
                    "row_count": count,
                    "percent_of_126771982": pct(count),
                    "denominator": TOTAL_ROWS,
                }
            )
        write_csv(
            output_dir / "anomaly_investigation_summary.csv",
            ["category", "row_count", "percent_of_126771982", "denominator"],
            main_rows,
        )

        def stats_for(category: str, columns: list[str]) -> dict[str, object]:
            row: dict[str, object] = {
                "category": category,
                "row_count": int(result[f"count__{category}"] or 0),
                "percent_of_126771982": pct(int(result[f"count__{category}"] or 0)),
                "denominator": TOTAL_ROWS,
            }
            for column in columns:
                prefix = f"stat__{category}__{column}__"
                for metric in ("min", "median_approx", "mean", "max"):
                    row[f"{column}_{metric}"] = result[prefix + metric]
            return row

        negative_fare_row = stats_for("negative_base_passenger_fare", ["base_passenger_fare"])
        for column in ("trip_miles", "trip_time", "driver_pay", "tips"):
            negative_fare_row[f"{column}_non_null_rows"] = int(
                result[f"negative_fare__{column}__non_null"] or 0
            )
            negative_fare_row[f"{column}_null_rows"] = int(
                result[f"negative_fare__{column}__null"] or 0
            )
        write_csv(output_dir / "negative_fare_summary.csv", list(negative_fare_row), [negative_fare_row])

        for category, columns, filename in (
            ("zero_trip_miles", ["base_passenger_fare", "driver_pay", "tips"], "zero_miles_summary.csv"),
            ("zero_trip_time", ["base_passenger_fare", "driver_pay", "tips"], "zero_time_summary.csv"),
            ("negative_driver_pay", ["base_passenger_fare", "tips", "trip_miles", "trip_time"], "negative_driver_pay_summary.csv"),
        ):
            row = stats_for(category, columns)
            write_csv(output_dir / filename, list(row), [row])

        zero_driver_row = {
            "category": "zero_driver_pay",
            "row_count": int(result["count__zero_driver_pay"] or 0),
            "percent_of_126771982": pct(int(result["count__zero_driver_pay"] or 0)),
            "denominator": TOTAL_ROWS,
        }
        write_csv(output_dir / "zero_driver_pay_summary.csv", list(zero_driver_row), [zero_driver_row])

        # Month and provider distributions are bounded group aggregates.
        distribution_conditions = [
            ("negative_base_passenger_fare", conditions["negative_base_passenger_fare"]),
            ("zero_trip_miles", conditions["zero_trip_miles"]),
            ("zero_trip_time", conditions["zero_trip_time"]),
            ("negative_driver_pay", conditions["negative_driver_pay"]),
            ("zero_driver_pay", conditions["zero_driver_pay"]),
        ]

        def write_group_distributions(group_column: str, label: str, suffix: str) -> None:
            aggregations = [
                F.sum(F.when(condition, 1).otherwise(0)).alias(category)
                for category, condition in distribution_conditions
            ]
            grouped = df.groupBy(group_column).agg(*aggregations).toLocalIterator()
            materialized = list(grouped)  # at most six months or a few provider values
            for category, _ in distribution_conditions:
                rows = []
                for item in materialized:
                    value = item[group_column]
                    count = int(item[category] or 0)
                    rows.append(
                        {
                            label: "<NULL>" if value is None else str(value),
                            "row_count": count,
                            "percent_of_126771982": pct(count),
                            "denominator": TOTAL_ROWS,
                        }
                    )
                rows.sort(key=lambda entry: (entry["row_count"], entry[label]), reverse=True)
                write_csv(
                    output_dir / f"{category}{suffix}.csv",
                    [label, "row_count", "percent_of_126771982", "denominator"],
                    rows,
                )

        write_group_distributions("_source_month", "month", "_by_month")
        write_group_distributions("hvfhs_license_num", "hvfhs_license_num", "_by_provider")

        zero_combo_rows = [
            {
                "combination": category,
                "row_count": int(result[f"count__{category}"] or 0),
                "percent_of_126771982": pct(int(result[f"count__{category}"] or 0)),
                "denominator": TOTAL_ROWS,
            }
            for category in (
                "zero_trip_miles_and_time",
                "zero_miles_positive_time",
                "positive_miles_zero_time",
            )
        ]
        write_csv(
            output_dir / "zero_combination_summary.csv",
            ["combination", "row_count", "percent_of_126771982", "denominator"],
            zero_combo_rows,
        )

        timestamp_rows = []
        for category in (
            "pickup_datetime_null",
            "dropoff_datetime_null",
            "dropoff_before_pickup",
            "duration_nonpositive",
            "duration_over_24_hours",
            "duration_over_48_hours",
        ):
            count = int(result[f"count__{category}"] or 0)
            timestamp_rows.append(
                {
                    "category": category,
                    "row_count": count,
                    "percent_of_126771982": pct(count),
                    "denominator": TOTAL_ROWS,
                }
            )
        write_csv(
            output_dir / "timestamp_anomaly_summary.csv",
            ["category", "row_count", "percent_of_126771982", "denominator"],
            timestamp_rows,
        )
        extreme_rows = [
            row for row in timestamp_rows if row["category"] in ("duration_over_24_hours", "duration_over_48_hours")
        ]
        write_csv(
            output_dir / "extreme_duration_summary.csv",
            ["category", "row_count", "percent_of_126771982", "denominator"],
            extreme_rows,
        )

        # Examples are intentionally limited at source, before crossing to Python.
        example_fields = [name for name in EXAMPLE_COLUMNS if name in original_columns]

        def write_examples(filename: str, condition, extra_columns: list[str] | None = None) -> None:
            selected_columns = list(dict.fromkeys(example_fields + (extra_columns or [])))
            examples = (
                df.where(condition)
                .select(*selected_columns)
                .limit(20)
                .toLocalIterator()
            )
            headers = selected_columns
            write_csv(output_dir / filename, headers, (row.asDict(recursive=True) for row in examples))

        write_examples("negative_fare_examples.csv", conditions["negative_base_passenger_fare"])
        write_examples("zero_miles_examples.csv", conditions["zero_trip_miles"])
        write_examples("zero_time_examples.csv", conditions["zero_trip_time"])
        write_examples("negative_driver_pay_examples.csv", conditions["negative_driver_pay"])
        write_examples("zero_driver_pay_examples.csv", conditions["zero_driver_pay"])
        timestamp_problem = (
            conditions["pickup_datetime_null"]
            | conditions["dropoff_datetime_null"]
            | conditions["dropoff_before_pickup"]
            | conditions["duration_nonpositive"]
            | conditions["duration_over_24_hours"]
        )
        write_examples("timestamp_anomaly_examples.csv", timestamp_problem, ["_source_month"])

        # First count repeated four-key combinations. Only rows in those
        # combinations can be exact duplicates, so inspect all 25 columns on
        # this small candidate subset rather than grouping the full dataset.
        key_groups = df.groupBy(*KEY_COLUMNS).agg(F.count(F.lit(1)).alias("key_count")).where(
            F.col("key_count") > 1
        )
        left = df.alias("data")
        right = key_groups.alias("keys")
        join_condition = None
        for column in KEY_COLUMNS:
            term = F.col(f"data.{column}").eqNullSafe(F.col(f"keys.{column}"))
            join_condition = term if join_condition is None else join_condition & term
        candidates = left.join(right, join_condition, "inner").select(
            *[F.col(f"data.{column}").alias(column) for column in original_columns]
        )
        exact_duplicate_groups = candidates.groupBy(*original_columns).agg(
            F.count(F.lit(1)).alias("full_row_count")
        ).where(F.col("full_row_count") > 1)
        full_duplicate_summary = exact_duplicate_groups.agg(
            F.count(F.lit(1)).alias("duplicate_full_row_patterns"),
            F.sum(F.col("full_row_count") - 1).alias("full_duplicate_excess_rows"),
            F.sum(F.col("full_row_count")).alias("rows_in_full_duplicate_groups"),
        ).first()
        key_summary = key_groups.agg(
            F.sum(F.col("key_count") - 1).alias("four_key_duplicate_excess_rows"),
            F.sum(F.col("key_count")).alias("rows_in_four_key_duplicate_groups"),
            F.count(F.lit(1)).alias("repeated_four_key_combinations"),
        ).first()
        duplicate_rows = [
            {
                "check": "four_key_combination",
                "row_count": int(key_summary["four_key_duplicate_excess_rows"] or 0),
                "percent_of_126771982": pct(int(key_summary["four_key_duplicate_excess_rows"] or 0)),
                "duplicate_patterns": int(key_summary["repeated_four_key_combinations"] or 0),
                "rows_in_repeated_groups": int(key_summary["rows_in_four_key_duplicate_groups"] or 0),
                "denominator": TOTAL_ROWS,
            },
            {
                "check": "identical_all_25_columns_within_four_key_candidates",
                "row_count": int(full_duplicate_summary["full_duplicate_excess_rows"] or 0),
                "percent_of_126771982": pct(int(full_duplicate_summary["full_duplicate_excess_rows"] or 0)),
                "duplicate_patterns": int(full_duplicate_summary["duplicate_full_row_patterns"] or 0),
                "rows_in_repeated_groups": int(full_duplicate_summary["rows_in_full_duplicate_groups"] or 0),
                "denominator": TOTAL_ROWS,
            },
        ]
        write_csv(
            output_dir / "duplicate_investigation_summary.csv",
            ["check", "row_count", "percent_of_126771982", "duplicate_patterns", "rows_in_repeated_groups", "denominator"],
            duplicate_rows,
        )

        key_examples = (
            key_groups.orderBy(F.desc("key_count"))
            .limit(20)
            .toLocalIterator()
        )
        write_csv(
            output_dir / "duplicate_examples.csv",
            KEY_COLUMNS + ["key_count"],
            (row.asDict(recursive=True) for row in key_examples),
        )

        identical_examples = (
            exact_duplicate_groups.orderBy(F.desc("full_row_count"))
            .limit(20)
            .toLocalIterator()
        )
        write_csv(
            output_dir / "identical_full_row_examples.csv",
            original_columns + ["full_row_count"],
            (row.asDict(recursive=True) for row in identical_examples),
        )

        print(f"Total denominator: {TOTAL_ROWS:,}; rows read: {actual_total:,}")
        if actual_total != TOTAL_ROWS:
            print("PERINGATAN: jumlah baris terbaca berbeda dari denominator yang diminta.")
        print("Ringkasan anomali:")
        for row in main_rows:
            print(
                f"  {row['category']}: {row['row_count']:,} "
                f"({row['percent_of_126771982']:.6f}%)"
            )
        print(
            "Duplicate excess: "
            f"4-key={duplicate_rows[0]['row_count']:,}; "
            f"identik 25 kolom={duplicate_rows[1]['row_count']:,}"
        )
        print(f"Output tersimpan di: {output_dir}")
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
