"""Profile a bounded, reproducible sample of NYC TLC HVFHV Parquet files."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SAMPLE_PER_FILE = 100_000
RANDOM_SEED = 2026
NUMERIC_COLUMNS = [
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tips",
    "driver_pay",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
]
DATETIME_COLUMNS = ["pickup_datetime", "dropoff_datetime"]
CATEGORICAL_COLUMNS = [
    "hvfhs_license_num",
    "dispatching_base_num",
    "originating_base_num",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
]
DISTRIBUTION_COLUMNS = [
    "hvfhs_license_num",
    "shared_request_flag",
    "shared_match_flag",
]


def take_file_sample(path: Path, sample_limit: int, rng: random.Random) -> pd.DataFrame:
    """Read a few randomly selected row groups, then randomly sample their rows.

    Parquet does not provide cheap arbitrary row access. Reading selected row
    groups bounds I/O to a fraction of the file while random row selection
    avoids always taking the first rows from those groups.
    """
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    group_count = metadata.num_row_groups
    if group_count == 0 or metadata.num_rows == 0:
        return pd.DataFrame(columns=parquet.schema_arrow.names)

    target = min(sample_limit, metadata.num_rows)
    # Read enough groups to keep the selected fraction modest; each group is
    # loaded separately so the full file is never held in memory.
    group_rows = [metadata.row_group(i).num_rows for i in range(group_count)]
    typical_group_rows = sorted(group_rows)[len(group_rows) // 2]
    groups_needed = max(1, (target + typical_group_rows - 1) // typical_group_rows)
    groups_needed = min(group_count, groups_needed)
    selected_groups = rng.sample(range(group_count), groups_needed)
    selected_groups.sort()

    base_quota, extra = divmod(target, groups_needed)
    sampled_parts: list[pd.DataFrame] = []
    for position, group_index in enumerate(selected_groups):
        quota = base_quota + (1 if position < extra else 0)
        row_count = group_rows[group_index]
        if quota >= row_count:
            row_indices = list(range(row_count))
        else:
            row_indices = sorted(rng.sample(range(row_count), quota))

        # Keep one row group in memory at a time, select sampled rows in Arrow,
        # and convert only the bounded sample to Pandas.
        table = parquet.read_row_group(group_index)
        indices = pa.array(row_indices, type=pa.int64())
        sampled_parts.append(table.take(indices).to_pandas())
        del table

    sample = pd.concat(sampled_parts, ignore_index=True)
    if len(sample) > target:
        sample = sample.iloc[:target].copy()
    return sample


def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    source_dir = data_dir if data_dir.is_dir() else project_dir
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f"fhvhv_tripdata_2026-{month:02d}.parquet" for month in range(1, 7)]
    missing_files = [name for name in files if not (source_dir / name).is_file()]
    if missing_files:
        print(f"Folder data: {source_dir}")
        print("File Parquet tidak ditemukan:")
        for name in missing_files:
            print(f"  - {name}")
        return 1

    rng = random.Random(RANDOM_SEED)
    month_samples: list[tuple[str, pd.DataFrame]] = []
    profile_rows: list[dict[str, object]] = []

    for filename in files:
        sample = take_file_sample(source_dir / filename, SAMPLE_PER_FILE, rng)
        month = filename.removeprefix("fhvhv_tripdata_").removesuffix(".parquet")
        month_samples.append((month, sample))
        profile_rows.append(
            {"section": "sample", "column": "all", "metric": f"rows_{month}", "value": len(sample)}
        )
        print(f"{month}: {len(sample):,} sample rows")

    combined = pd.concat([sample for _, sample in month_samples], ignore_index=True)
    print(f"\nTotal sample rows: {len(combined):,}")

    missing_rows = []
    for column in combined.columns:
        count = int(combined[column].isna().sum())
        missing_rows.append(
            {
                "column": column,
                "missing_count": count,
                "sample_rows": len(combined),
                "missing_percent": (count / len(combined) * 100) if len(combined) else 0.0,
            }
        )
        profile_rows.append(
            {"section": "dtype", "column": column, "metric": "sample_dtype", "value": str(combined[column].dtype)}
        )

    for column in CATEGORICAL_COLUMNS:
        if column in combined.columns:
            profile_rows.append(
                {
                    "section": "categorical",
                    "column": column,
                    "metric": "unique_count",
                    "value": int(combined[column].nunique(dropna=True)),
                }
            )

    for column in NUMERIC_COLUMNS:
        if column not in combined.columns:
            continue
        values = pd.to_numeric(combined[column], errors="coerce")
        stats = {
            "min": values.min(),
            "median": values.median(),
            "mean": values.mean(),
            "max": values.max(),
        }
        for metric, value in stats.items():
            profile_rows.append(
                {
                    "section": "numeric_summary",
                    "column": column,
                    "metric": metric,
                    "value": value,
                }
            )

    for column in DATETIME_COLUMNS:
        if column not in combined.columns:
            continue
        values = pd.to_datetime(combined[column], errors="coerce")
        for metric, value in (("min", values.min()), ("max", values.max())):
            profile_rows.append(
                {
                    "section": "datetime_summary",
                    "column": column,
                    "metric": metric,
                    "value": value,
                }
            )

    for column in ("PULocationID", "DOLocationID"):
        if column in combined.columns:
            profile_rows.append(
                {
                    "section": "location_summary",
                    "column": column,
                    "metric": "unique_count",
                    "value": int(combined[column].nunique(dropna=True)),
                }
            )

    for column in DISTRIBUTION_COLUMNS:
        if column not in combined.columns:
            continue
        counts = combined[column].value_counts(dropna=False)
        for category, count in counts.items():
            profile_rows.append(
                {
                    "section": "distribution",
                    "column": column,
                    "metric": "count",
                    "value": int(count),
                    "category": "<NA>" if pd.isna(category) else str(category),
                }
            )

    profile_path = output_dir / "hvfhs_sample_profile.csv"
    missing_path = output_dir / "hvfhs_missing_values.csv"
    pd.DataFrame(profile_rows).to_csv(profile_path, index=False)
    pd.DataFrame(missing_rows).to_csv(missing_path, index=False)

    print("\nDistribusi kategorikal:")
    for column in DISTRIBUTION_COLUMNS:
        if column in combined.columns:
            print(f"\n{column}:")
            print(combined[column].value_counts(dropna=False).to_string())
    print(f"\nHasil disimpan ke:\n  {profile_path}\n  {missing_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
