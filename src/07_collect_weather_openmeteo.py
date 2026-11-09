"""Collect hourly Open-Meteo historical weather for NYC Taxi Zone points.

Raw API responses are saved by batch for audit/resume. Normalized JSON Lines
are an intermediate format; PySpark converts them to Parquet after validation.
This script does not read or join trip rows.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


API_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 6, 30)
TIMEZONE_NAME = "America/New_York"
HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
    "cloud_cover",
)
EXPECTED_UNITS = {
    "temperature_2m": {"°c"},
    "relative_humidity_2m": {"%"},
    "precipitation": {"mm"},
    "rain": {"mm"},
    "snowfall": {"cm"},
    "weather_code": {"wmo code", "wmo"},
    "wind_speed_10m": {"km/h"},
    "cloud_cover": {"%"},
}
BATCH_SIZE = 10
REQUEST_TIMEOUT_SECONDS = 90
MAX_ATTEMPTS = 4
EXPECTED_ZONE_COUNT = 258
RAW_SCHEMA_VERSION = 1


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("openmeteo_weather_collection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def write_csv_atomic(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def append_csv(path: Path, headers: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Temporary raw file exists; inspect before continuing: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def import_zone_source(project_dir: Path):
    module_path = project_dir / "src" / "06_integrate_taxi_zones_spark.py"
    spec = importlib.util.spec_from_file_location("taxi_zone_source", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Tidak dapat memuat fungsi titik zona dari {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_zone_coordinates(project_dir: Path) -> list[dict[str, Any]]:
    reference_dir = project_dir / "data" / "reference" / "taxi_zones"
    lookup_path = reference_dir / "taxi_zone_lookup.csv"
    geojson_path = reference_dir / "nyc_taxi_zones.geojson"
    required_files = (lookup_path, geojson_path)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Referensi Taxi Zone belum tersedia: {missing}")

    source_module = import_zone_source(project_dir)
    lookup, _headers = source_module.read_lookup(lookup_path)
    points, issues, crs, _feature_count = source_module.read_geometries(geojson_path)
    if "4326" not in crs.lower() and "crs84" not in crs.lower():
        raise ValueError(f"CRS referensi bukan WGS84: {crs}")

    zones = []
    for row in lookup:
        location_id = int(row["LocationID"])
        point = points.get(location_id)
        if point is None:
            continue
        latitude, longitude = point
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            raise ValueError(f"Koordinat tidak finite untuk LocationID={location_id}")
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError(f"Koordinat di luar rentang untuk LocationID={location_id}")
        zones.append(
            {
                "zone_id": location_id,
                "borough": row.get("Borough"),
                "zone_name": row.get("Zone"),
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
        )
    zones.sort(key=lambda item: item["zone_id"])
    if len(zones) != EXPECTED_ZONE_COUNT:
        raise ValueError(
            f"Ditemukan {len(zones)} titik zona yang valid; laporan integrasi sebelumnya menyatakan "
            f"{EXPECTED_ZONE_COUNT}. Refusing to change request scope silently. Geometry issues={len(issues)}"
        )
    if len({zone["zone_id"] for zone in zones}) != len(zones):
        raise ValueError("Lookup lokasi tidak unik setelah titik koordinat dipasangkan.")
    return zones


def make_coordinate_groups(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Request a coordinate only once if multiple zones share an exact point."""
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for zone in zones:
        key = (zone["latitude"], zone["longitude"])
        grouped.setdefault(key, []).append(zone)
    groups = []
    for (latitude, longitude), members in grouped.items():
        groups.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "zones": sorted(members, key=lambda item: item["zone_id"]),
            }
        )
    groups.sort(key=lambda item: (item["zones"][0]["zone_id"], item["latitude"], item["longitude"]))
    return groups


def make_params(groups: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "latitude": ",".join(f"{group['latitude']:.8f}" for group in groups),
        "longitude": ",".join(f"{group['longitude']:.8f}" for group in groups),
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        # The docs permit comma-separated timezone values for coordinate lists.
        "timezone": ",".join(TIMEZONE_NAME for _group in groups),
        "timeformat": "unixtime",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }


def request_api(params: dict[str, str]) -> tuple[Any, int]:
    url = API_URL + "?" + urllib.parse.urlencode(params)
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "NYC-HVFHV-Research-Weather/1.0", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = int(response.status)
                content_type = response.headers.get("Content-Type", "")
                body = response.read()
            if status != 200:
                raise RuntimeError(f"API merespons HTTP {status}")
            if "json" not in content_type.lower():
                raise RuntimeError(f"Content-Type respons bukan JSON: {content_type!r}")
            try:
                decoded = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Respons API bukan JSON valid: {exc}") from exc
            if isinstance(decoded, dict) and decoded.get("error"):
                raise RuntimeError(f"API menolak parameter: {decoded.get('reason', decoded)}")
            return decoded, status
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (408, 425, 429, 500, 502, 503, 504) or attempt == MAX_ATTEMPTS:
                detail = exc.read(2000).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = min(60, int(retry_after)) if retry_after and retry_after.isdigit() else 2 ** attempt
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"Kegagalan koneksi setelah {attempt} percobaan: {exc}") from exc
            delay = 2 ** attempt
        if delay:
            time.sleep(delay)
    raise RuntimeError(f"Request API gagal: {last_error}")


def load_or_fetch_batch(
    raw_path: Path,
    endpoint: str,
    params: dict[str, str],
    groups: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, int | None]:
    expected_locations = [
        {"latitude": group["latitude"], "longitude": group["longitude"], "zone_ids": [z["zone_id"] for z in group["zones"]]}
        for group in groups
    ]
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as stream:
            wrapper = json.load(stream)
        if wrapper.get("schema_version") != RAW_SCHEMA_VERSION:
            raise ValueError(f"Versi schema raw batch tidak dikenal: {raw_path}")
        if wrapper.get("endpoint") != endpoint or wrapper.get("params") != params:
            raise ValueError(f"Request batch raw tidak cocok dengan run ini; tidak ditimpa: {raw_path}")
        if wrapper.get("requested_locations") != expected_locations:
            raise ValueError(f"Daftar lokasi pada raw batch tidak cocok; tidak ditimpa: {raw_path}")
        return normalize_response_shape(wrapper.get("response"), len(groups)), "reused_raw", wrapper.get("http_status")

    response, status = request_api(params)
    data = normalize_response_shape(response, len(groups))
    atomic_json_write(
        raw_path,
        {
            "schema_version": RAW_SCHEMA_VERSION,
            "endpoint": endpoint,
            "params": params,
            "requested_locations": expected_locations,
            "requested_at_utc": now_utc(),
            "http_status": status,
            "response": response,
        },
    )
    return data, "downloaded", status


def normalize_response_shape(response: Any, expected_locations: int) -> list[dict[str, Any]]:
    if expected_locations == 1 and isinstance(response, dict):
        response = [response]
    if not isinstance(response, list):
        raise ValueError(f"Response JSON harus berupa list of locations, aktual={type(response).__name__}")
    if len(response) != expected_locations:
        raise ValueError(f"API mengembalikan {len(response)} lokasi, diminta {expected_locations}.")
    if any(not isinstance(item, dict) for item in response):
        raise ValueError("Salah satu item respons lokasi bukan objek JSON.")
    return response


def validate_units(location_data: dict[str, Any]) -> dict[str, dict[str, str]]:
    units = location_data.get("hourly_units")
    hourly = location_data.get("hourly")
    if not isinstance(units, dict) or not isinstance(hourly, dict):
        raise ValueError("Response tidak mempunyai hourly_units/hourly object.")
    validations = {}
    for variable in HOURLY_VARIABLES:
        actual = units.get(variable)
        if not isinstance(actual, str):
            raise ValueError(f"Unit untuk {variable} tidak tersedia: {actual!r}")
        normalized = actual.strip().lower().replace("degrees", "°")
        expected = EXPECTED_UNITS[variable]
        status = "passed" if normalized in expected else "failed"
        validations[variable] = {"expected": " or ".join(sorted(expected)), "actual": actual, "status": status}
    failed = [name for name, row in validations.items() if row["status"] != "passed"]
    if failed:
        raise ValueError(f"Satuan API tidak sesuai untuk {failed}: {validations}")
    return validations


def expected_timestamp_count() -> int:
    zone = ZoneInfo(TIMEZONE_NAME)
    first = datetime.combine(START_DATE, datetime.min.time(), tzinfo=zone)
    after_last = datetime.combine(END_DATE + timedelta(days=1), datetime.min.time(), tzinfo=zone)
    return int((after_last.astimezone(timezone.utc) - first.astimezone(timezone.utc)).total_seconds() // 3600)


def response_to_records(
    location_data: dict[str, Any],
    coordinate_group: dict[str, Any],
    zone_info: ZoneInfo,
    units_seen: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if location_data.get("timezone") != TIMEZONE_NAME:
        raise ValueError(f"Timezone respons={location_data.get('timezone')!r}; diharapkan {TIMEZONE_NAME!r}")
    if not isinstance(location_data.get("utc_offset_seconds"), int):
        raise ValueError("Response tidak memiliki utc_offset_seconds integer.")
    local_units = validate_units(location_data)
    for variable, row in local_units.items():
        previous = units_seen.get(variable)
        if previous and previous != row:
            raise ValueError(f"Unit berubah antarlokasi untuk {variable}: {previous} vs {row}")
        units_seen[variable] = row

    hourly = location_data["hourly"]
    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        raise ValueError("hourly.time kosong atau bukan list.")
    for variable in HOURLY_VARIABLES:
        values = hourly.get(variable)
        if not isinstance(values, list) or len(values) != len(times):
            raise ValueError(
                f"Array hourly.{variable} hilang atau panjangnya berbeda dengan time "
                f"({len(values) if isinstance(values, list) else 'missing'} vs {len(times)})."
            )

    epochs: list[int] = []
    for value in times:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"timeformat=unixtime menghasilkan nilai waktu bukan numeric: {value!r}")
        if not float(value).is_integer():
            raise ValueError(f"Timestamp Unix bukan integer detik: {value!r}")
        epochs.append(int(value))
    if epochs != sorted(epochs) or len(set(epochs)) != len(epochs):
        raise ValueError("Timestamp API tidak berurutan atau memiliki duplikasi dalam satu zona.")
    if any(second - first != 3600 for first, second in zip(epochs, epochs[1:])):
        raise ValueError("Interval timestamp API tidak tepat satu jam.")

    records: list[dict[str, Any]] = []
    zone_quality: list[dict[str, Any]] = []
    out_of_range_indices: list[int] = []
    duplicate_count = 0
    missing_by_variable = {variable: 0 for variable in HOURLY_VARIABLES}
    in_range_epochs: list[tuple[int, int, datetime]] = []
    for epoch_index, epoch in enumerate(epochs):
        utc_dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        local_dt = utc_dt.astimezone(zone_info)
        local_date = local_dt.date()
        if START_DATE <= local_date <= END_DATE:
            in_range_epochs.append((epoch_index, epoch, local_dt))
        else:
            out_of_range_indices.append(epoch_index)

    # Keep any request-time fixed-offset boundary hour in raw JSON, but normalize
    # only the desired New York calendar dates. An out-of-window hour in the
    # middle of the time series is a schema/data error and stops the run.
    if len(out_of_range_indices) > 2:
        raise ValueError(f"Terlalu banyak timestamp API di luar tanggal lokal yang diminta: {out_of_range_indices}")
    if any(index not in (0, len(epochs) - 1) for index in out_of_range_indices):
        raise ValueError("Timestamp di luar rentang muncul di tengah deret API, bukan pada batas request.")

    for epoch_index, epoch, local_dt in in_range_epochs:
        local_date = local_dt.date()
        for zone in coordinate_group["zones"]:
            record: dict[str, Any] = {
                "zone_id": zone["zone_id"],
                "borough": zone["borough"],
                "zone_name": zone["zone_name"],
                "latitude": zone["latitude"],
                "longitude": zone["longitude"],
                "weather_timestamp_epoch": epoch,
                "weather_time_utc": datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "weather_time_local": local_dt.isoformat(timespec="seconds"),
                "weather_date_local": local_date.isoformat(),
                "weather_hour_local": local_dt.hour,
            }
            for variable in HOURLY_VARIABLES:
                value = hourly[variable][epoch_index]
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"Nilai {variable} bukan numeric: {value!r}")
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"Nilai {variable} tidak finite: {value!r}")
                    if variable == "weather_code":
                        if not value.is_integer():
                            raise ValueError(f"weather_code bukan integer WMO: {value}")
                        value = int(value)
                else:
                    missing_by_variable[variable] += 1
                record[variable] = value
            records.append(record)
    expected = expected_timestamp_count()
    for zone in coordinate_group["zones"]:
        zone_quality.append(
            {
                "zone_id": zone["zone_id"],
                "rows_in_response": len(in_range_epochs),
                "api_returned_timestamp_count": len(epochs),
                "expected_hourly_timestamps": expected,
                "missing_timestamp_count_vs_expected": max(0, expected - len(in_range_epochs)),
                "duplicate_timestamp_count": duplicate_count,
                "out_of_requested_local_date_count": len(out_of_range_indices),
                "response_first_local": datetime.fromtimestamp(epochs[0], timezone.utc).astimezone(zone_info).isoformat(timespec="seconds"),
                "response_last_local": datetime.fromtimestamp(epochs[-1], timezone.utc).astimezone(zone_info).isoformat(timespec="seconds"),
                "normalized_first_local": in_range_epochs[0][2].isoformat(timespec="seconds") if in_range_epochs else "",
                "normalized_last_local": in_range_epochs[-1][2].isoformat(timespec="seconds") if in_range_epochs else "",
                **{f"missing_{name}": missing_by_variable[name] for name in HOURLY_VARIABLES},
            }
        )
    return records, zone_quality


def write_normalized_batch(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists():
        raise FileExistsError(f"Normalized batch already exists; refusing to overwrite: {path}")
    if temporary.exists():
        raise FileExistsError(f"Temporary normalized batch exists; inspect manually: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_existing_normalized(path: Path) -> tuple[int, list[int]]:
    row_count = 0
    zone_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                zone_id = int(row["zone_id"])
                int(row["weather_timestamp_epoch"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Normalized batch rusak pada {path}:{line_number}: {exc}") from exc
            zone_ids.add(zone_id)
            row_count += 1
    if row_count == 0:
        raise ValueError(f"Normalized batch kosong: {path}")
    return row_count, sorted(zone_ids)


def audit_project_schema(spark, project_dir: Path, report_dir: Path) -> list[dict[str, str]]:
    integrated_dir = project_dir / "output" / "integrated" / "taxi_zones"
    validation_path = project_dir / "output" / "quality" / "taxi_zone_integration" / "integration_run_status.csv"
    if not integrated_dir.is_dir():
        raise FileNotFoundError(f"Output Taxi Zone tidak ada: {integrated_dir}")
    if not validation_path.is_file():
        raise FileNotFoundError(f"Laporan validasi Taxi Zone tidak ada: {validation_path}")
    validation_rows = read_csv_rows(validation_path)
    if not validation_rows or validation_rows[-1].get("status") != "VALIDATION_PASSED":
        raise ValueError("Status validasi integrasi Taxi Zone terakhir bukan VALIDATION_PASSED.")

    frame = spark.read.parquet(str(integrated_dir))
    type_by_name = {field.name: field.dataType.simpleString() for field in frame.schema.fields}
    expected = {
        "pickup_datetime": "timestamp_ntz",
        "pickup_date": "date",
        "PULocationID": "int",
        "pickup_latitude": "double",
        "pickup_longitude": "double",
    }
    differences = {
        name: {"expected": data_type, "actual": type_by_name.get(name)}
        for name, data_type in expected.items()
        if type_by_name.get(name) != data_type
    }
    if differences:
        raise TypeError(f"Schema dataset terintegrasi berubah: {differences}")
    rows = [
        {"column": field.name, "spark_data_type": field.dataType.simpleString(), "nullable": field.nullable}
        for field in frame.schema.fields
    ]
    write_csv_atomic(report_dir / "integrated_trip_schema.csv", ["column", "spark_data_type", "nullable"], rows)
    return rows


def write_source_manifest(path: Path, batch_count: int, coordinate_count: int, zone_count: int) -> None:
    row = {
        "source": "Open-Meteo Historical Weather API",
        "endpoint": API_URL,
        "api_parameters": "hourly; timeformat=unixtime; temperature_unit=celsius; wind_speed_unit=kmh; precipitation_unit=mm; cell_selection=land",
        "date_start_inclusive": START_DATE.isoformat(),
        "date_end_inclusive": END_DATE.isoformat(),
        "timezone": TIMEZONE_NAME,
        "requested_variables": ",".join(HOURLY_VARIABLES),
        "requested_zones": zone_count,
        "unique_coordinates": coordinate_count,
        "batch_size_coordinates": BATCH_SIZE,
        "planned_batches": batch_count,
        "request_time_utc": now_utc(),
        "coordinate_source": "data/reference/taxi_zones/taxi_zone_lookup.csv + nyc_taxi_zones.geojson; same interior-point implementation as src/06_integrate_taxi_zones_spark.py",
        "normalized_staging_directory": "normalized_local_window_v2/",
        "raw_response_directory": "raw/",
        "api_documentation": "https://open-meteo.com/en/docs/historical-weather-api",
    }
    write_csv_atomic(path, list(row), [row])


def ensure_spark():
    try:
        from pyspark.sql import SparkSession, functions as F, types as T
    except ImportError as exc:
        raise RuntimeError("PySpark tidak tersedia; tidak memasang dependency secara otomatis.") from exc
    return SparkSession, F, T


def spark_quality_and_parquet(
    spark,
    F,
    T,
    normalized_dir: Path,
    parquet_dir: Path,
    quality_dir: Path,
    requested_zones: list[dict[str, Any]],
    expected_count: int,
    units_seen: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if parquet_dir.exists():
        raise FileExistsError(f"Parquet cuaca sudah ada; tidak ditimpa: {parquet_dir}")
    normalized_files = sorted(normalized_dir.glob("batch_*.jsonl"))
    if not normalized_files:
        raise FileNotFoundError("Tidak ada normalized batch JSONL untuk dibaca.")
    schema = T.StructType(
        [
            T.StructField("zone_id", T.IntegerType(), False),
            T.StructField("borough", T.StringType(), True),
            T.StructField("zone_name", T.StringType(), True),
            T.StructField("latitude", T.DoubleType(), False),
            T.StructField("longitude", T.DoubleType(), False),
            T.StructField("weather_timestamp_epoch", T.LongType(), False),
            T.StructField("weather_time_utc", T.StringType(), False),
            T.StructField("weather_time_local", T.StringType(), False),
            T.StructField("weather_date_local", T.StringType(), False),
            T.StructField("weather_hour_local", T.IntegerType(), False),
            T.StructField("temperature_2m", T.DoubleType(), True),
            T.StructField("relative_humidity_2m", T.DoubleType(), True),
            T.StructField("precipitation", T.DoubleType(), True),
            T.StructField("rain", T.DoubleType(), True),
            T.StructField("snowfall", T.DoubleType(), True),
            T.StructField("weather_code", T.IntegerType(), True),
            T.StructField("wind_speed_10m", T.DoubleType(), True),
            T.StructField("cloud_cover", T.DoubleType(), True),
        ]
    )
    frame = spark.read.schema(schema).json([str(path) for path in normalized_files])
    frame = frame.withColumn(
        "weather_time_utc",
        F.to_timestamp(F.from_unixtime(F.col("weather_timestamp_epoch"))),
    ).withColumn("weather_date_local", F.to_date("weather_date_local"))

    row_count = frame.count()
    zone_count = frame.select("zone_id").distinct().count()
    duplicate_groups = (
        frame.groupBy("zone_id", "weather_timestamp_epoch")
        .count()
        .where(F.col("count") > 1)
    )
    duplicate_summary = duplicate_groups.agg(
        F.count(F.lit(1)).alias("duplicate_combinations"),
        F.coalesce(F.sum(F.col("count") - 1), F.lit(0)).alias("duplicate_excess_rows"),
    ).first().asDict()

    aggregates = [
        F.min("weather_date_local").alias("date_min"),
        F.max("weather_date_local").alias("date_max"),
        F.count(F.lit(1)).alias("observations"),
        F.countDistinct("weather_timestamp_epoch").alias("unique_timestamps"),
        F.sum(F.when(F.col("weather_date_local") < F.lit(START_DATE.isoformat()), 1).otherwise(0)).alias("before_start_rows"),
        F.sum(F.when(F.col("weather_date_local") > F.lit(END_DATE.isoformat()), 1).otherwise(0)).alias("after_end_rows"),
    ]
    aggregates.extend(
        F.sum(F.when(F.col(variable).isNull(), 1).otherwise(0)).alias(f"missing_{variable}")
        for variable in HOURLY_VARIABLES
    )
    zone_rows = []
    for row in frame.groupBy("zone_id").agg(*aggregates).orderBy("zone_id").toLocalIterator():
        zone_rows.append(
            {
                "zone_id": row["zone_id"],
                "date_min_local": row["date_min"],
                "date_max_local": row["date_max"],
                "observations": row["observations"],
                "expected_observations": expected_count,
                "missing_observations_vs_expected": max(0, expected_count - row["observations"]),
                "unique_timestamps": row["unique_timestamps"],
                "duplicate_timestamps": row["observations"] - row["unique_timestamps"],
                "rows_before_start": row["before_start_rows"] or 0,
                "rows_after_end": row["after_end_rows"] or 0,
                **{f"missing_{variable}": row[f"missing_{variable}"] or 0 for variable in HOURLY_VARIABLES},
            }
        )
    write_csv_atomic(
        quality_dir / "weather_quality_by_zone.csv",
        [
            "zone_id", "date_min_local", "date_max_local", "observations", "expected_observations",
            "missing_observations_vs_expected", "unique_timestamps", "duplicate_timestamps",
            "rows_before_start", "rows_after_end", *[f"missing_{v}" for v in HOURLY_VARIABLES],
        ],
        zone_rows,
    )

    global_agg = frame.agg(*aggregates).first().asDict()
    missing_total = sum(int(global_agg.get(f"missing_{v}") or 0) for v in HOURLY_VARIABLES)
    quality_summary = [
        {"metric": "requested_zones", "value": len(requested_zones)},
        {"metric": "zones_with_weather", "value": zone_count},
        {"metric": "zones_missing_weather", "value": len(requested_zones) - zone_count},
        {"metric": "total_observations", "value": row_count},
        {"metric": "expected_observations_per_zone", "value": expected_count},
        {"metric": "expected_total_observations", "value": expected_count * len(requested_zones)},
        {"metric": "date_min_local", "value": global_agg["date_min"]},
        {"metric": "date_max_local", "value": global_agg["date_max"]},
        {"metric": "duplicate_zone_timestamp_combinations", "value": duplicate_summary["duplicate_combinations"]},
        {"metric": "duplicate_excess_rows", "value": duplicate_summary["duplicate_excess_rows"]},
        {"metric": "rows_before_start", "value": global_agg["before_start_rows"] or 0},
        {"metric": "rows_after_end", "value": global_agg["after_end_rows"] or 0},
        {"metric": "missing_weather_values_total_across_variables", "value": missing_total},
    ]
    for variable in HOURLY_VARIABLES:
        quality_summary.append(
            {"metric": f"missing_{variable}", "value": global_agg.get(f"missing_{variable}") or 0}
        )
    write_csv_atomic(quality_dir / "weather_quality_summary.csv", ["metric", "value"], quality_summary)

    unit_rows = [
        {
            "variable": variable,
            "expected_unit": unit_row["expected"],
            "actual_unit": unit_row["actual"],
            "status": unit_row["status"],
        }
        for variable, unit_row in sorted(units_seen.items())
    ]
    write_csv_atomic(quality_dir / "weather_units_validation.csv", ["variable", "expected_unit", "actual_unit", "status"], unit_rows)

    # Convert the documented Unix timestamps into Spark timestamps in UTC.
    # Keep weather_time_local (with numeric UTC offset) for local-time use and audit.
    if parquet_dir.exists():
        raise FileExistsError(f"Parquet output appeared during run; not overwriting: {parquet_dir}")
    frame.repartition(181, "weather_date_local").sortWithinPartitions(
        "zone_id", "weather_timestamp_epoch"
    ).write.mode("errorifexists").partitionBy("weather_date_local").parquet(str(parquet_dir))

    saved = spark.read.parquet(str(parquet_dir))
    saved_count = saved.count()
    output_rows = [
        {"metric": "rows_before_parquet_write", "value": row_count},
        {"metric": "rows_after_parquet_readback", "value": saved_count},
        {"metric": "row_count_preserved", "value": str(row_count == saved_count).lower()},
        {"metric": "spark_parquet_output", "value": str(parquet_dir)},
    ]
    write_csv_atomic(quality_dir / "parquet_write_validation.csv", ["metric", "value"], output_rows)

    return {
        "row_count": row_count,
        "zone_count": zone_count,
        "saved_count": saved_count,
        "duplicate_combinations": int(duplicate_summary["duplicate_combinations"] or 0),
        "duplicate_excess_rows": int(duplicate_summary["duplicate_excess_rows"] or 0),
        "quality_summary": quality_summary,
    }


def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent
    integrated_dir = project_dir / "output" / "integrated" / "taxi_zones"
    weather_dir = project_dir / "data" / "reference" / "weather_openmeteo"
    raw_dir = weather_dir / "raw"
    # Versioned staging preserves earlier intermediate JSONL files; the previous
    # attempt included Open-Meteo's request-time fixed-offset boundary hour.
    normalized_dir = weather_dir / "normalized_local_window_v2"
    parquet_dir = weather_dir / "weather_hourly_parquet"
    quality_dir = weather_dir / "quality"
    log_path = weather_dir / "logs" / "collect_weather.log"
    logger = setup_logging(log_path)

    spark = None
    try:
        SparkSession, F, T = ensure_spark()
        if not integrated_dir.is_dir():
            raise FileNotFoundError(f"Dataset Taxi Zone terintegrasi tidak ditemukan: {integrated_dir}")
        if parquet_dir.exists():
            raise FileExistsError(f"Output Parquet sudah ada; tidak ditimpa: {parquet_dir}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        normalized_dir.mkdir(parents=True, exist_ok=True)
        quality_dir.mkdir(parents=True, exist_ok=True)

        spark_local_dir = Path.home() / "spark-local-weather"
        spark_local_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("SPARK_LOCAL_DIRS", str(spark_local_dir))
        spark = (
            SparkSession.builder.appName("HVFHS2026OpenMeteoWeather")
            .master("local[2]")
            .config("spark.sql.shuffle.partitions", "32")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.local.dir", str(spark_local_dir))
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        schema_rows = audit_project_schema(spark, project_dir, quality_dir)
        logger.info(
            "Verified integrated schema: pickup_datetime=timestamp_ntz, pickup_date=date, "
            "pickup coordinates=double; read schema only, no TLC row scan/join requested."
        )

        zones = load_zone_coordinates(project_dir)
        coordinate_groups = make_coordinate_groups(zones)
        expected_per_zone = expected_timestamp_count()
        batches = [
            coordinate_groups[index:index + BATCH_SIZE]
            for index in range(0, len(coordinate_groups), BATCH_SIZE)
        ]
        source_manifest_path = weather_dir / "source_manifest.csv"
        write_source_manifest(source_manifest_path, len(batches), len(coordinate_groups), len(zones))
        logger.info(
            "Beginning hourly collection: zones=%d unique_coordinates=%d batches=%d expected_hours_per_zone=%d",
            len(zones), len(coordinate_groups), len(batches), expected_per_zone,
        )

        manifest_headers = [
            "batch_id", "status", "action", "attempted_at_utc", "completed_at_utc", "coordinate_count",
            "zone_count", "endpoint", "date_start", "date_end", "timezone", "variables",
            "expected_hours_per_coordinate", "returned_rows", "http_status", "request_sha256", "error",
        ]
        manifest_path = weather_dir / "batch_manifest.csv"
        quality_by_zone: dict[int, dict[str, Any]] = {}
        units_seen: dict[str, dict[str, str]] = {}
        failures: list[dict[str, Any]] = []
        all_params = []
        batch_ids = []
        zone_by_id = {zone["zone_id"]: zone for zone in zones}
        zone_info = ZoneInfo(TIMEZONE_NAME)

        for batch_number, group_batch in enumerate(batches, start=1):
            batch_id = f"batch_{batch_number:03d}"
            batch_ids.append(batch_id)
            params = make_params(group_batch)
            all_params.append(params)
            raw_path = raw_dir / f"{batch_id}.json"
            normalized_path = normalized_dir / f"{batch_id}.jsonl"
            attempted_at = now_utc()
            request_hash = hash_text(json.dumps(params, sort_keys=True, separators=(",", ":")))
            logger.info("Batch %s/%s: %s coordinates", batch_number, len(batches), len(group_batch))
            try:
                location_payloads, action, http_status = load_or_fetch_batch(
                    raw_path, API_URL, params, group_batch
                )
                records: list[dict[str, Any]] = []
                batch_quality: list[dict[str, Any]] = []
                for location_payload, coordinate_group in zip(location_payloads, group_batch):
                    normalized, zone_quality = response_to_records(
                        location_payload, coordinate_group, zone_info, units_seen
                    )
                    records.extend(normalized)
                    batch_quality.extend(zone_quality)

                if normalized_path.exists():
                    existing_count, existing_zone_ids = load_existing_normalized(normalized_path)
                    expected_zone_ids = sorted(zone_by_id[z["zone_id"]]["zone_id"] for group in group_batch for z in group["zones"])
                    if existing_count != len(records) or existing_zone_ids != expected_zone_ids:
                        raise ValueError(
                            f"Normalized file tidak sesuai raw batch dan tidak ditimpa: {normalized_path}"
                        )
                    action += "+reused_normalized"
                else:
                    write_normalized_batch(normalized_path, records)
                for item in batch_quality:
                    quality_by_zone[item["zone_id"]] = item
                append_csv(
                    manifest_path,
                    manifest_headers,
                    {
                        "batch_id": batch_id,
                        "status": "success",
                        "action": action,
                        "attempted_at_utc": attempted_at,
                        "completed_at_utc": now_utc(),
                        "coordinate_count": len(group_batch),
                        "zone_count": sum(len(group["zones"]) for group in group_batch),
                        "endpoint": API_URL,
                        "date_start": START_DATE.isoformat(),
                        "date_end": END_DATE.isoformat(),
                        "timezone": TIMEZONE_NAME,
                        "variables": ",".join(HOURLY_VARIABLES),
                        "expected_hours_per_coordinate": expected_per_zone,
                        "returned_rows": len(records),
                        "http_status": http_status or "reused",
                        "request_sha256": request_hash,
                        "error": "",
                    },
                )
                logger.info("Batch %s successful: normalized_rows=%s (%s)", batch_id, f"{len(records):,}", action)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                failures.append({"batch_id": batch_id, "status": "failed", "error": detail})
                append_csv(
                    manifest_path,
                    manifest_headers,
                    {
                        "batch_id": batch_id,
                        "status": "failed",
                        "action": "failed",
                        "attempted_at_utc": attempted_at,
                        "completed_at_utc": now_utc(),
                        "coordinate_count": len(group_batch),
                        "zone_count": sum(len(group["zones"]) for group in group_batch),
                        "endpoint": API_URL,
                        "date_start": START_DATE.isoformat(),
                        "date_end": END_DATE.isoformat(),
                        "timezone": TIMEZONE_NAME,
                        "variables": ",".join(HOURLY_VARIABLES),
                        "expected_hours_per_coordinate": expected_per_zone,
                        "returned_rows": 0,
                        "http_status": "",
                        "request_sha256": request_hash,
                        "error": detail,
                    },
                )
                logger.exception("Batch %s failed", batch_id)
                break

        if failures:
            write_csv_atomic(
                quality_dir / "weather_collection_failures.csv",
                ["batch_id", "status", "error"],
                failures,
            )
            logger.error("Collection stopped safely: %d failed batch(es); raw successful responses are retained.", len(failures))
            return 1

        expected_zone_ids = {zone["zone_id"] for zone in zones}
        if set(quality_by_zone) != expected_zone_ids:
            missing = sorted(expected_zone_ids.difference(quality_by_zone))
            write_csv_atomic(
                quality_dir / "weather_collection_failures.csv",
                ["batch_id", "status", "error"],
                [{"batch_id": "", "status": "failed", "error": f"Tidak ada hasil untuk zone_id: {missing}"}],
            )
            raise RuntimeError(f"Cuaca tidak berhasil diperoleh untuk semua zona: {missing}")

        # Confirm the returned local-time span before constructing a final dataset.
        wrong_range = [
            row for row in quality_by_zone.values()
            if row["rows_in_response"] != expected_per_zone
            or row["missing_timestamp_count_vs_expected"] != 0
            or row["out_of_requested_local_date_count"] > 2
        ]
        if wrong_range:
            write_csv_atomic(
                quality_dir / "weather_collection_failures.csv",
                ["batch_id", "status", "error"],
                [{"batch_id": "timestamp_validation", "status": "failed", "error": f"Respons waktu di luar cakupan/kurang lengkap: {wrong_range[:10]}"}],
            )
            raise ValueError(
                "Rentang timestamp API tidak cocok dengan 2026-01-01 s.d. 2026-06-30 "
                "dalam America/New_York; data mentah dipertahankan dan Parquet final tidak dibuat."
            )

        write_csv_atomic(
            quality_dir / "weather_api_boundary_audit.csv",
            [
                "zone_id", "api_returned_timestamp_count", "normalized_observations",
                "boundary_rows_excluded_from_requested_local_window", "api_first_local_time",
                "api_last_local_time", "normalized_first_local_time", "normalized_last_local_time",
            ],
            [
                {
                    "zone_id": row["zone_id"],
                    "api_returned_timestamp_count": row["api_returned_timestamp_count"],
                    "normalized_observations": row["rows_in_response"],
                    "boundary_rows_excluded_from_requested_local_window": row["out_of_requested_local_date_count"],
                    "api_first_local_time": row["response_first_local"],
                    "api_last_local_time": row["response_last_local"],
                    "normalized_first_local_time": row["normalized_first_local"],
                    "normalized_last_local_time": row["normalized_last_local"],
                }
                for row in sorted(quality_by_zone.values(), key=lambda item: item["zone_id"])
            ],
        )

        # Per-coordinate JSON unit information is retained in raw responses; this
        # manifest records the validated units actually returned by the API.
        write_csv_atomic(
            quality_dir / "weather_units_validation.csv",
            ["variable", "expected_unit", "actual_unit", "status"],
            [
                {
                    "variable": variable,
                    "expected_unit": unit_row["expected"],
                    "actual_unit": unit_row["actual"],
                    "status": unit_row["status"],
                }
                for variable, unit_row in sorted(units_seen.items())
            ],
        )
        api_schema_rows = [
            {"json_path": "timezone", "observed_type": "string", "description": "Must equal America/New_York"},
            {"json_path": "utc_offset_seconds", "observed_type": "integer", "description": "Offset metadata supplied by API"},
            {"json_path": "hourly.time", "observed_type": "array of Unix epoch seconds", "description": "Requested with timeformat=unixtime; converted to UTC and America/New_York"},
            {"json_path": "hourly_units", "observed_type": "object of variable to unit strings", "description": "Validated against expected units"},
        ]
        api_schema_rows.extend(
            {"json_path": f"hourly.{variable}", "observed_type": "array of numeric values or null", "description": "Array length must match hourly.time"}
            for variable in HOURLY_VARIABLES
        )
        write_csv_atomic(
            quality_dir / "api_response_schema.csv",
            ["json_path", "observed_type", "description"],
            api_schema_rows,
        )
        logger.info("All batches validated. Writing Parquet with PySpark.")
        final_stats = spark_quality_and_parquet(
            spark,
            F,
            T,
            normalized_dir,
            parquet_dir,
            quality_dir,
            zones,
            expected_per_zone,
            units_seen,
        )

        run_status = "VALIDATION_PASSED"
        expected_total = expected_per_zone * len(zones)
        if final_stats["saved_count"] != final_stats["row_count"]:
            run_status = "VALIDATION_FAILED"
        if final_stats["zone_count"] != len(zones):
            run_status = "VALIDATION_FAILED"
        if final_stats["row_count"] != expected_total:
            run_status = "VALIDATION_FAILED"
        if final_stats["duplicate_excess_rows"] != 0:
            run_status = "VALIDATION_FAILED"
        append_csv(
            quality_dir / "weather_run_status.csv",
            ["status", "finished_at_utc", "requested_zones", "zones_with_weather", "expected_observations", "actual_observations", "parquet_rows", "duplicate_excess_rows", "detail"],
            {
                "status": run_status,
                "finished_at_utc": now_utc(),
                "requested_zones": len(zones),
                "zones_with_weather": final_stats["zone_count"],
                "expected_observations": expected_total,
                "actual_observations": final_stats["row_count"],
                "parquet_rows": final_stats["saved_count"],
                "duplicate_excess_rows": final_stats["duplicate_excess_rows"],
                "detail": "No trip join performed. Earlier failed attempts are retained in weather_collection_failures.csv; this row records the completed run.",
            },
        )
        logger.info("Requested zones: %d; weather zones: %d", len(zones), final_stats["zone_count"])
        logger.info("Weather observations: %s (expected %s)", f"{final_stats['row_count']:,}", f"{expected_total:,}")
        logger.info("Duplicate excess rows: %d", final_stats["duplicate_excess_rows"])
        logger.info("Parquet output: %s", parquet_dir)
        logger.info("Quality reports: %s", quality_dir)
        logger.info("Final status: %s", run_status)
        return 0 if run_status == "VALIDATION_PASSED" else 1
    except Exception as exc:
        logger.exception("Weather collection failed safely: %s", exc)
        try:
            write_csv_atomic(
                quality_dir / "weather_collection_failures.csv",
                ["batch_id", "status", "error"],
                [{"batch_id": "preflight_or_final", "status": "failed", "error": f"{type(exc).__name__}: {exc}"}],
            )
        except Exception:
            pass
        return 1
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception as exc:
                logger.warning("Spark gagal berhenti secara bersih: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
