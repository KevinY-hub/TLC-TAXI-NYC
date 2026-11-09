"""Integrate NYC TLC Taxi Zone names and representative coordinates.

The script uses the standard library to download and validate the small
reference files and to compute an interior point from GeoJSON polygons. It
uses PySpark only for the full cleaned-trip join and distributed validation.
No geospatial package is installed or required.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_ROWS = 126_771_977
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
GEOJSON_URL = (
    "https://data.cityofnewyork.us/api/v3/views/8meu-9t5y/"
    "query.geojson?accessType=DOWNLOAD"
)
LOOKUP_NAME = "taxi_zone_lookup.csv"
GEOJSON_NAME = "nyc_taxi_zones.geojson"
LOOKUP_REQUIRED = ("LocationID", "Borough", "Zone", "service_zone")
OUTPUT_COLUMNS = (
    "pickup_borough",
    "pickup_zone",
    "pickup_service_zone",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_borough",
    "dropoff_zone",
    "dropoff_service_zone",
    "dropoff_latitude",
    "dropoff_longitude",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_reference(url: str, destination: Path) -> dict[str, str]:
    """Download to a new file only; never replace an existing reference."""
    inspected_at = utc_now()
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"Referensi sudah ada tetapi bukan file valid: {destination}")
        return {
            "file": destination.name,
            "url": url,
            "action": "reused_existing",
            "downloaded_at_utc": "",
            "inspected_at_utc": inspected_at,
            "sha256": sha256_file(destination),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".download-partial")
    if partial.exists():
        raise RuntimeError(f"File unduhan sementara sudah ada; periksa manual: {partial}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "HVFHV-Taxi-Zone-Integration/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("xb") as out:
            content_type = response.headers.get("Content-Type", "")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if partial.stat().st_size == 0:
            raise RuntimeError(f"Unduhan kosong dari {url} (Content-Type: {content_type})")
        # os.rename fails rather than replacing an existing destination on Windows.
        if destination.exists():
            raise FileExistsError(f"File muncul selama unduhan; tidak ditimpa: {destination}")
        os.rename(partial, destination)
        downloaded_at = utc_now()
        return {
            "file": destination.name,
            "url": url,
            "action": "downloaded",
            "downloaded_at_utc": downloaded_at,
            "inspected_at_utc": downloaded_at,
            "sha256": sha256_file(destination),
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(f"Gagal mengunduh {url}: {exc}") from exc
    except Exception:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def append_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    mode = "a" if exists else "x"
    with path.open(mode, newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        if not exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)


def write_csv_new(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    """Write a generated report atomically so a failed run can be retried."""
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Temporary report already exists; inspect manually: {temporary}")
    with temporary.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_lookup(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        missing = [name for name in LOOKUP_REQUIRED if name not in headers]
        if missing:
            raise ValueError(f"Kolom wajib lookup tidak ditemukan: {missing}; tersedia={headers}")
        records: list[dict[str, Any]] = []
        seen: set[int] = set()
        duplicates: set[int] = set()
        for row_number, raw in enumerate(reader, start=2):
            value = (raw.get("LocationID") or "").strip()
            try:
                location_id = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"LocationID lookup tidak valid pada baris {row_number}: {value!r}") from exc
            if location_id in seen:
                duplicates.add(location_id)
            seen.add(location_id)
            records.append(
                {
                    "LocationID": location_id,
                    "Borough": (raw.get("Borough") or "").strip() or None,
                    "Zone": (raw.get("Zone") or "").strip() or None,
                    "service_zone": (raw.get("service_zone") or "").strip() or None,
                    "latitude": None,
                    "longitude": None,
                }
            )
    if duplicates:
        raise ValueError(f"LocationID lookup duplikat: {sorted(duplicates)}")
    if not records:
        raise ValueError("Lookup CSV tidak memiliki record.")
    return records, headers


def extract_feature_id(properties: dict[str, Any]) -> int:
    for key, value in properties.items():
        normalized = "".join(character for character in str(key).lower() if character.isalnum())
        if normalized == "locationid":
            return int(value)
    raise ValueError("Feature GeoJSON tidak memiliki property LocationID/locationid.")


def ring_intersections(ring: list[list[float]], y: float) -> list[float]:
    crossings: list[float] = []
    for first, second in zip(ring, ring[1:]):
        x1, y1 = float(first[0]), float(first[1])
        x2, y2 = float(second[0]), float(second[1])
        if (y1 <= y < y2) or (y2 <= y < y1):
            crossings.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    return crossings


def point_in_polygon(point: tuple[float, float], rings: list[list[list[float]]]) -> bool:
    x, y = point

    def in_ring(ring: list[list[float]]) -> bool:
        inside = False
        for first, second in zip(ring, ring[1:]):
            x1, y1 = float(first[0]), float(first[1])
            x2, y2 = float(second[0]), float(second[1])
            if (y1 > y) != (y2 > y):
                crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < crossing_x:
                    inside = not inside
        return inside

    return bool(rings) and in_ring(rings[0]) and not any(in_ring(ring) for ring in rings[1:])


def polygon_representative_point(rings: list[list[list[float]]]) -> tuple[float, float] | None:
    """Return a deterministic scanline point strictly inside a valid polygon.

    GeoJSON coordinates are [longitude, latitude]. Scanlines are placed
    strictly between vertex latitudes; even-odd pairing across exterior and
    interior rings excludes holes. The chosen interval midpoint is validated
    against the polygon before being returned.
    """
    if not rings or not rings[0]:
        return None
    ys = sorted({float(point[1]) for ring in rings for point in ring if len(point) >= 2})
    best: tuple[float, float, float] | None = None  # width, longitude, latitude
    for low, high in zip(ys, ys[1:]):
        if high <= low:
            continue
        y = (low + high) / 2.0
        crossings = sorted(x for ring in rings for x in ring_intersections(ring, y))
        if len(crossings) < 2:
            continue
        # Even-odd fill rule supports polygon holes without relying on ring orientation.
        for left, right in zip(crossings[::2], crossings[1::2]):
            width = right - left
            if width <= 0:
                continue
            point = ((left + right) / 2.0, y)
            if point_in_polygon(point, rings) and (best is None or width > best[0]):
                best = (width, point[0], point[1])
    return None if best is None else (best[1], best[2])


def representative_point(geometry: dict[str, Any]) -> tuple[float, float] | None:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        return None
    candidates = [polygon_representative_point(rings) for rings in polygons or []]
    valid = [point for point in candidates if point is not None]
    if not valid:
        return None
    # Each point has been tested inside its polygon. Choose deterministically.
    return max(valid, key=lambda point: (point[0], -point[1]))


def crs_description(geojson: dict[str, Any]) -> str:
    crs = geojson.get("crs")
    if not crs:
        return "GeoJSON default WGS84 longitude/latitude (EPSG:4326 / CRS84)"
    properties = crs.get("properties", {}) if isinstance(crs, dict) else {}
    name = str(properties.get("name", ""))
    normalized = name.lower().replace(" ", "")
    if "4326" not in normalized and "crs84" not in normalized and "crs:84" not in normalized:
        raise ValueError(
            f"GeoJSON menyatakan CRS {name!r}; dependency reproyeksi tidak tersedia, dibatalkan."
        )
    return name


def read_geometries(path: Path) -> tuple[dict[int, tuple[float, float]], list[dict[str, Any]], str, int]:
    with path.open("r", encoding="utf-8-sig") as stream:
        geojson = json.load(stream)
    if geojson.get("type") != "FeatureCollection" or not isinstance(geojson.get("features"), list):
        raise ValueError("GeoJSON harus berupa FeatureCollection dengan daftar features.")
    crs = crs_description(geojson)
    feature_groups: dict[int, list[dict[str, Any]]] = {}
    issues: list[dict[str, Any]] = []
    for index, feature in enumerate(geojson["features"]):
        try:
            location_id = extract_feature_id(feature.get("properties") or {})
        except (TypeError, ValueError) as exc:
            issues.append({"LocationID": "", "status": "invalid_geometry_id", "detail": f"feature={index}: {exc}"})
            continue
        feature_groups.setdefault(location_id, []).append(feature)

    points: dict[int, tuple[float, float]] = {}
    for location_id, features in feature_groups.items():
        if len(features) != 1:
            issues.append({"LocationID": location_id, "status": "duplicate_geometry_id", "detail": f"features={len(features)}"})
            continue
        geometry = features[0].get("geometry") or {}
        point = representative_point(geometry)
        if point is None:
            issues.append({"LocationID": location_id, "status": "representative_point_failed", "detail": str(geometry.get("type"))})
            continue
        longitude, latitude = point
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            issues.append({"LocationID": location_id, "status": "coordinate_out_of_range", "detail": f"lon={longitude};lat={latitude}"})
            continue
        # Broad NYC-area sanity bounds catch axis swaps or an unexpected CRS.
        if not (-75 <= longitude <= -72 and 39 <= latitude <= 42):
            issues.append({"LocationID": location_id, "status": "outside_nyc_sanity_bounds", "detail": f"lon={longitude};lat={latitude}"})
            continue
        points[location_id] = (latitude, longitude)
    return points, issues, crs, len(geojson["features"])


def main() -> int:
    try:
        from pyspark import StorageLevel
        from pyspark.sql import SparkSession, functions as F, types as T
    except ImportError as exc:
        print(f"ERROR: PySpark tidak tersedia di interpreter ini: {exc}", file=sys.stderr)
        print("Aktifkan ~/hvfhs-venv di WSL; dependency geospasial tidak diperlukan oleh script ini.", file=sys.stderr)
        return 2

    project_dir = Path(__file__).resolve().parent.parent
    references_dir = project_dir / "data" / "reference" / "taxi_zones"
    cleaned_dir = project_dir / "output" / "cleaned"
    output_dir = project_dir / "output" / "integrated" / "taxi_zones"
    reports_dir = project_dir / "output" / "quality" / "taxi_zone_integration"
    lookup_path = references_dir / LOOKUP_NAME
    geojson_path = references_dir / GEOJSON_NAME

    if not cleaned_dir.is_dir():
        print(f"ERROR: Cleaned dataset tidak ditemukan: {cleaned_dir}", file=sys.stderr)
        return 1
    if output_dir.exists():
        print(f"ERROR: Output sudah ada; script menolak menimpa: {output_dir}", file=sys.stderr)
        return 1

    try:
        source_records = [
            fetch_reference(LOOKUP_URL, lookup_path),
            fetch_reference(GEOJSON_URL, geojson_path),
        ]
        lookup_records, lookup_headers = read_lookup(lookup_path)
        geometry_points, geometry_issues, crs, geojson_feature_count = read_geometries(geojson_path)
        lookup_ids = {record["LocationID"] for record in lookup_records}
        geometry_ids = set(geometry_points)
        for record in lookup_records:
            point = geometry_points.get(record["LocationID"])
            if point:
                record["latitude"], record["longitude"] = point
        for location_id in sorted(geometry_ids - lookup_ids):
            geometry_issues.append({"LocationID": location_id, "status": "geometry_id_not_in_lookup", "detail": "Tidak ada LocationID pada CSV lookup."})
        for location_id in sorted(lookup_ids - geometry_ids):
            geometry_issues.append({"LocationID": location_id, "status": "lookup_id_without_valid_geometry", "detail": "Koordinat representatif tidak tersedia."})
    except Exception as exc:
        print(f"ERROR saat mengambil/memvalidasi referensi: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        # Append-only provenance: existing manifest entries are preserved.
        append_csv(
            references_dir / "source_manifest.csv",
            ["file", "url", "action", "downloaded_at_utc", "inspected_at_utc", "sha256"],
            source_records,
        )
        schema_rows = [
            {"source_file": LOOKUP_NAME, "column": name, "role": "lookup", "type": "string in CSV; LocationID parsed as int32" if name == "LocationID" else "string"}
            for name in lookup_headers
        ]
        schema_rows.extend(
            [
                {"source_file": GEOJSON_NAME, "column": "properties.locationid", "role": "join key", "type": "integer"},
                {"source_file": GEOJSON_NAME, "column": "geometry.type", "role": "geometry", "type": "Polygon or MultiPolygon"},
                {"source_file": GEOJSON_NAME, "column": "geometry.coordinates", "role": "coordinates", "type": "WGS84 [longitude, latitude]"},
            ]
        )
        reports_dir.mkdir(parents=True, exist_ok=True)
        write_csv_new(reports_dir / "reference_schema.csv", ["source_file", "column", "role", "type"], schema_rows)
        write_csv_new(
            reports_dir / "geometry_issues.csv",
            ["LocationID", "status", "detail"],
            geometry_issues,
        )
        geometry_summary = [
            {"metric": "lookup_rows", "value": len(lookup_records)},
            {"metric": "lookup_unique_location_ids", "value": len(lookup_ids)},
            {"metric": "geojson_features", "value": geojson_feature_count},
            {"metric": "geometry_ids_with_representative_point", "value": len(geometry_points)},
            {"metric": "lookup_ids_with_geometry_point", "value": len(lookup_ids & geometry_ids)},
            {"metric": "lookup_ids_without_valid_geometry", "value": len(lookup_ids - geometry_ids)},
            {"metric": "geometry_ids_not_in_lookup", "value": len(geometry_ids - lookup_ids)},
            {"metric": "geometry_issues", "value": len(geometry_issues)},
            {"metric": "source_crs", "value": crs},
        ]
        write_csv_new(reports_dir / "zone_geometry_summary.csv", ["metric", "value"], geometry_summary)
    except (OSError, PermissionError) as exc:
        print(f"ERROR permission saat menyimpan referensi/laporan: {exc}", file=sys.stderr)
        print("Jalankan dari WSL setelah mount metadata /mnt/c aktif dan pastikan folder belum terkunci.", file=sys.stderr)
        return 1

    if not lookup_records:
        print("ERROR: Lookup kosong; Spark join dibatalkan.", file=sys.stderr)
        return 1

    spark_local_dir = Path.home() / "spark-local"
    spark_local_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SPARK_LOCAL_DIRS", str(spark_local_dir))
    spark = None
    try:
        spark = (
            SparkSession.builder.appName("HVFHS2026TaxiZoneIntegration")
            .master("local[2]")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.files.maxPartitionBytes", "32m")
            .config("spark.local.dir", str(spark_local_dir))
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        # The cleaned dataset is several GB; avoid caching all 126M rows in the
        # local JVM heap. Spark will stream Parquet partitions for each action.
        trips = spark.read.parquet(str(cleaned_dir))
        required_trip_columns = {"PULocationID", "DOLocationID"}
        missing_trip_columns = sorted(required_trip_columns.difference(trips.columns))
        if missing_trip_columns:
            raise ValueError(f"Kolom lokasi tidak ditemukan: {missing_trip_columns}")
        if any(column in trips.columns for column in OUTPUT_COLUMNS):
            raise ValueError("Output attribute Taxi Zone sudah ada pada cleaned dataset; menolak menimpa kolom.")
        for column in ("PULocationID", "DOLocationID"):
            data_type = trips.schema[column].dataType
            if not isinstance(data_type, (T.IntegerType, T.LongType)):
                raise TypeError(f"{column} harus integer, schema aktual: {data_type.simpleString()}")

        rows_before = trips.count()
        if rows_before != EXPECTED_ROWS:
            raise ValueError(f"Jumlah cleaned rows {rows_before:,} tidak sama dengan ekspektasi {EXPECTED_ROWS:,}.")

        lookup_schema = T.StructType(
            [
                T.StructField("LocationID", T.IntegerType(), nullable=False),
                T.StructField("Borough", T.StringType(), nullable=True),
                T.StructField("Zone", T.StringType(), nullable=True),
                T.StructField("service_zone", T.StringType(), nullable=True),
                T.StructField("latitude", T.DoubleType(), nullable=True),
                T.StructField("longitude", T.DoubleType(), nullable=True),
            ]
        )
        zone_lookup = spark.createDataFrame(lookup_records, schema=lookup_schema)
        lookup_count = zone_lookup.count()
        lookup_unique = zone_lookup.select("LocationID").distinct().count()
        if lookup_count != lookup_unique:
            raise ValueError(f"Lookup LocationID tidak unik ({lookup_count} rows vs {lookup_unique} IDs); join dibatalkan.")

        pickup_lookup = F.broadcast(
            zone_lookup.select(
                F.col("LocationID").alias("_pickup_id"),
                F.col("Borough").alias("pickup_borough"),
                F.col("Zone").alias("pickup_zone"),
                F.col("service_zone").alias("pickup_service_zone"),
                F.col("latitude").alias("pickup_latitude"),
                F.col("longitude").alias("pickup_longitude"),
            )
        )
        dropoff_lookup = F.broadcast(
            zone_lookup.select(
                F.col("LocationID").alias("_dropoff_id"),
                F.col("Borough").alias("dropoff_borough"),
                F.col("Zone").alias("dropoff_zone"),
                F.col("service_zone").alias("dropoff_service_zone"),
                F.col("latitude").alias("dropoff_latitude"),
                F.col("longitude").alias("dropoff_longitude"),
            )
        )
        joined = (
            trips.withColumn("_pickup_join_id", F.col("PULocationID").cast("int"))
            .withColumn("_dropoff_join_id", F.col("DOLocationID").cast("int"))
            .join(pickup_lookup, F.col("_pickup_join_id") == F.col("_pickup_id"), "left")
            .join(dropoff_lookup, F.col("_dropoff_join_id") == F.col("_dropoff_id"), "left")
            .drop("_pickup_join_id", "_dropoff_join_id", "_pickup_id", "_dropoff_id")
        )

        pickup_unmatched = F.col("PULocationID").isNotNull() & F.col("pickup_borough").isNull()
        dropoff_unmatched = F.col("DOLocationID").isNotNull() & F.col("dropoff_borough").isNull()
        agg_exprs = [
            F.count(F.lit(1)).alias("rows_after_join"),
            F.sum(F.when(pickup_unmatched, 1).otherwise(0)).alias("pickup_unmatched_rows"),
            F.countDistinct(F.when(pickup_unmatched, F.col("PULocationID"))).alias("pickup_unmatched_distinct_ids"),
            F.sum(F.when(dropoff_unmatched, 1).otherwise(0)).alias("dropoff_unmatched_rows"),
            F.countDistinct(F.when(dropoff_unmatched, F.col("DOLocationID"))).alias("dropoff_unmatched_distinct_ids"),
            F.sum(F.when(F.col("PULocationID").isNull(), 1).otherwise(0)).alias("pickup_null_id_rows"),
            F.sum(F.when(F.col("DOLocationID").isNull(), 1).otherwise(0)).alias("dropoff_null_id_rows"),
            F.sum(F.when(F.col("pickup_latitude").isNull() | F.col("pickup_longitude").isNull(), 1).otherwise(0)).alias("pickup_coordinate_null_rows"),
            F.sum(F.when(F.col("dropoff_latitude").isNull() | F.col("dropoff_longitude").isNull(), 1).otherwise(0)).alias("dropoff_coordinate_null_rows"),
            F.sum(F.when(F.col("pickup_latitude").isNotNull() & ((F.col("pickup_latitude") < -90) | (F.col("pickup_latitude") > 90) | (F.col("pickup_longitude") < -180) | (F.col("pickup_longitude") > 180)), 1).otherwise(0)).alias("pickup_coordinate_invalid_rows"),
            F.sum(F.when(F.col("dropoff_latitude").isNotNull() & ((F.col("dropoff_latitude") < -90) | (F.col("dropoff_latitude") > 90) | (F.col("dropoff_longitude") < -180) | (F.col("dropoff_longitude") > 180)), 1).otherwise(0)).alias("dropoff_coordinate_invalid_rows"),
        ]
        join_stats = joined.agg(*agg_exprs).first().asDict()
        rows_after = int(join_stats["rows_after_join"] or 0)
        row_count_unchanged = rows_before == rows_after

        # Do not begin a write if join cardinality already indicates multiplication.
        if not row_count_unchanged:
            raise RuntimeError(
                f"Join mengubah jumlah baris ({rows_before:,} -> {rows_after:,}); output tidak ditulis."
            )

        joined.write.mode("errorifexists").partitionBy("pickup_date").parquet(str(output_dir))
        saved = spark.read.parquet(str(output_dir))
        rows_saved = saved.count()
        if rows_saved != rows_before:
            raise RuntimeError(f"Validasi output gagal: rows tersimpan {rows_saved:,}, seharusnya {rows_before:,}.")

        joined_stats = saved.agg(
            F.sum(F.when(F.col("pickup_latitude").isNull() | F.col("pickup_longitude").isNull(), 1).otherwise(0)).alias("pickup_coordinate_null_rows"),
            F.sum(F.when(F.col("dropoff_latitude").isNull() | F.col("dropoff_longitude").isNull(), 1).otherwise(0)).alias("dropoff_coordinate_null_rows"),
            F.sum(F.when(F.col("pickup_latitude").isNotNull() & ((F.col("pickup_latitude") < -90) | (F.col("pickup_latitude") > 90) | (F.col("pickup_longitude") < -180) | (F.col("pickup_longitude") > 180)), 1).otherwise(0)).alias("pickup_coordinate_invalid_rows"),
            F.sum(F.when(F.col("dropoff_latitude").isNotNull() & ((F.col("dropoff_latitude") < -90) | (F.col("dropoff_latitude") > 90) | (F.col("dropoff_longitude") < -180) | (F.col("dropoff_longitude") > 180)), 1).otherwise(0)).alias("dropoff_coordinate_invalid_rows"),
        ).first().asDict()

        summary_rows = [
            {"metric": "rows_before_join", "value": rows_before, "percent_of_rows": 100.0},
            {"metric": "rows_after_join", "value": rows_after, "percent_of_rows": rows_after / rows_before * 100.0},
            {"metric": "rows_saved", "value": rows_saved, "percent_of_rows": rows_saved / rows_before * 100.0},
            {"metric": "rows_added_or_lost_by_join", "value": rows_after - rows_before, "percent_of_rows": (rows_after - rows_before) / rows_before * 100.0},
            {"metric": "lookup_rows", "value": lookup_count, "percent_of_rows": ""},
            {"metric": "lookup_unique_location_ids", "value": lookup_unique, "percent_of_rows": ""},
            {"metric": "pickup_unmatched_rows", "value": int(join_stats["pickup_unmatched_rows"] or 0), "percent_of_rows": int(join_stats["pickup_unmatched_rows"] or 0) / rows_before * 100.0},
            {"metric": "pickup_unmatched_distinct_ids", "value": int(join_stats["pickup_unmatched_distinct_ids"] or 0), "percent_of_rows": ""},
            {"metric": "pickup_unmatched_distinct_ids_percent_of_lookup", "value": int(join_stats["pickup_unmatched_distinct_ids"] or 0) / lookup_unique * 100.0, "percent_of_rows": ""},
            {"metric": "pickup_null_id_rows", "value": int(join_stats["pickup_null_id_rows"] or 0), "percent_of_rows": int(join_stats["pickup_null_id_rows"] or 0) / rows_before * 100.0},
            {"metric": "dropoff_unmatched_rows", "value": int(join_stats["dropoff_unmatched_rows"] or 0), "percent_of_rows": int(join_stats["dropoff_unmatched_rows"] or 0) / rows_before * 100.0},
            {"metric": "dropoff_unmatched_distinct_ids", "value": int(join_stats["dropoff_unmatched_distinct_ids"] or 0), "percent_of_rows": ""},
            {"metric": "dropoff_unmatched_distinct_ids_percent_of_lookup", "value": int(join_stats["dropoff_unmatched_distinct_ids"] or 0) / lookup_unique * 100.0, "percent_of_rows": ""},
            {"metric": "dropoff_null_id_rows", "value": int(join_stats["dropoff_null_id_rows"] or 0), "percent_of_rows": int(join_stats["dropoff_null_id_rows"] or 0) / rows_before * 100.0},
            {"metric": "pickup_coordinate_null_rows", "value": int(joined_stats["pickup_coordinate_null_rows"] or 0), "percent_of_rows": int(joined_stats["pickup_coordinate_null_rows"] or 0) / rows_before * 100.0},
            {"metric": "dropoff_coordinate_null_rows", "value": int(joined_stats["dropoff_coordinate_null_rows"] or 0), "percent_of_rows": int(joined_stats["dropoff_coordinate_null_rows"] or 0) / rows_before * 100.0},
            {"metric": "pickup_coordinate_invalid_rows", "value": int(joined_stats["pickup_coordinate_invalid_rows"] or 0), "percent_of_rows": int(joined_stats["pickup_coordinate_invalid_rows"] or 0) / rows_before * 100.0},
            {"metric": "dropoff_coordinate_invalid_rows", "value": int(joined_stats["dropoff_coordinate_invalid_rows"] or 0), "percent_of_rows": int(joined_stats["dropoff_coordinate_invalid_rows"] or 0) / rows_before * 100.0},
            {"metric": "join_row_count_unchanged", "value": str(row_count_unchanged).lower(), "percent_of_rows": ""},
            {"metric": "join_did_not_multiply_rows", "value": str(rows_after <= rows_before).lower(), "percent_of_rows": ""},
        ]
        write_csv_new(reports_dir / "integration_validation.csv", ["metric", "value", "percent_of_rows"], summary_rows)

        pickup_unmatched_counts = (
            joined.where(pickup_unmatched)
            .groupBy(F.col("PULocationID").alias("LocationID"))
            .count()
            .select(F.lit("pickup").alias("side"), "LocationID", "count")
        )
        dropoff_unmatched_counts = (
            joined.where(dropoff_unmatched)
            .groupBy(F.col("DOLocationID").alias("LocationID"))
            .count()
            .select(F.lit("dropoff").alias("side"), "LocationID", "count")
        )
        unmatched_rows = []
        for row in (
            pickup_unmatched_counts.unionByName(dropoff_unmatched_counts)
            .orderBy("side", "LocationID")
            .toLocalIterator()
        ):
            unmatched_rows.append({"side": row["side"], "LocationID": row["LocationID"], "trip_rows": row["count"]})
        write_csv_new(reports_dir / "unmatched_location_ids.csv", ["side", "LocationID", "trip_rows"], unmatched_rows)

        critical_ok = row_count_unchanged and rows_saved == rows_before and lookup_count == lookup_unique
        no_invalid_coords = (
            int(joined_stats["pickup_coordinate_invalid_rows"] or 0) == 0
            and int(joined_stats["dropoff_coordinate_invalid_rows"] or 0) == 0
        )
        quality_issue_count = len(geometry_issues) + int(join_stats["pickup_unmatched_rows"] or 0) + int(join_stats["dropoff_unmatched_rows"] or 0)
        status = "VALIDATION_PASSED" if critical_ok and no_invalid_coords else "VALIDATION_FAILED"
        append_csv(
            reports_dir / "integration_run_status.csv",
            ["status", "timestamp_utc", "detail"],
            [{"status": status, "timestamp_utc": utc_now(), "detail": f"critical_ok={critical_ok}; no_invalid_coords={no_invalid_coords}; geometry_or_unmatched_issues={quality_issue_count}"}],
        )
        print(f"Rows before join: {rows_before:,}")
        print(f"Rows after join: {rows_after:,}")
        print(f"Rows saved: {rows_saved:,}")
        print(f"Pickup unmatched rows: {int(join_stats['pickup_unmatched_rows'] or 0):,}")
        print(f"Drop-off unmatched rows: {int(join_stats['dropoff_unmatched_rows'] or 0):,}")
        print(f"Zones with representative points: {len(geometry_points):,}")
        print(f"Geometry quality issues: {len(geometry_issues):,}")
        print(f"Integrated Parquet: {output_dir}")
        print(f"Reports: {reports_dir}")
        print(status)
        return 0 if status == "VALIDATION_PASSED" else 1
    except PermissionError as exc:
        print(f"ERROR permission saat Spark membaca/menulis: {exc}", file=sys.stderr)
        print("Jalankan dari Ubuntu WSL dan pastikan automount metadata aktif untuk /mnt/c.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR integrasi Spark: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception as stop_exc:
                print(f"WARN: Spark gagal berhenti secara bersih: {stop_exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
