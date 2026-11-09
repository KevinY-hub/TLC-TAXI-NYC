"""Inspect NYC TLC HVFHV Parquet metadata without reading trip rows."""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq


FILES = [
    f"fhvhv_tripdata_2026-{month:02d}.parquet"
    for month in range(1, 7)
]


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    header = " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers))
    rule = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(value.ljust(widths[i]) for i, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    # Accept files in the project root too, which is how this project is
    # currently arranged. Prefer data/ whenever it exists.
    source_dir = data_dir if data_dir.is_dir() else project_dir

    print(f"Folder data: {source_dir}")
    paths = [source_dir / filename for filename in FILES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        print("File tidak ditemukan:")
        for name in missing:
            print(f"  - {name}")

    records: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            continue
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        records.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "rows": metadata.num_rows,
                "row_groups": metadata.num_row_groups,
                "columns": metadata.num_columns,
                "schema": parquet_file.schema_arrow,
            }
        )

    if records:
        rows = [
            [
                str(record["name"]),
                f"{int(record['bytes']) / 1_000_000:,.2f}",
                f"{int(record['bytes']) / 1_000_000_000:,.3f}",
                f"{int(record['rows']):,}",
                f"{int(record['row_groups']):,}",
                f"{int(record['columns']):,}",
            ]
            for record in records
        ]
        print("\nMetadata Parquet (MB/GB menggunakan satuan desimal):")
        print(
            format_table(
                ["File", "Ukuran (MB)", "Ukuran (GB)", "Rows", "Row groups", "Columns"],
                rows,
            )
        )

        total_bytes = sum(int(record["bytes"]) for record in records)
        total_rows = sum(int(record["rows"]) for record in records)
        print(f"\nTotal rows: {total_rows:,}")
        print(
            "Total ukuran: "
            f"{total_bytes / 1_000_000:,.2f} MB "
            f"({total_bytes / 1_000_000_000:,.3f} GB)"
        )

        print(f"\nSchema representatif ({records[0]['name']}):")
        print(records[0]["schema"])

        reference_schema = records[0]["schema"]
        inconsistent = [
            str(record["name"])
            for record in records[1:]
            if not reference_schema.equals(record["schema"], check_metadata=False)
        ]
        if inconsistent:
            print("\nKonsistensi schema: TIDAK KONSISTEN")
            print(f"Berbeda dari {records[0]['name']}: {', '.join(inconsistent)}")
        else:
            print("\nKonsistensi schema: KONSISTEN")

    if missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
