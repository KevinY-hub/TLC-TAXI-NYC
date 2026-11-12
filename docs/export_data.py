"""Export the approved dashboard aggregates as static JSON for GitHub Pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
ANALYSIS_DIR = PROJECT_ROOT / "output" / "analysis" / "weather_exploratory"
QUALITY_DIR = PROJECT_ROOT / "output" / "quality" / "weather_exploratory"
GRAPH_DIR = PROJECT_ROOT / "output" / "analysis" / "graph_analytics" / "20261107T174911Z"
GEOJSON_PATH = PROJECT_ROOT / "data" / "reference" / "taxi_zones" / "nyc_taxi_zones.geojson"

TABLES = {
    "daily": "temporal_trips_by_date",
    "date_hour": "temporal_trips_by_date_hour",
    "weather_monthly": "weather_trips_by_month_category",
    "locations": "pickup_location_volume",
    "boroughs": "pickup_borough_volume",
}

REPORTS = (
    "analysis_summary.csv",
    "weather_category_summary.csv",
    "weather_category_definitions.csv",
)


def write_records(name: str, frame: pd.DataFrame) -> int:
    """Write a compact JSON array, converting pandas values and nulls safely."""
    path = DATA_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(
        path,
        orient="records",
        date_format="iso",
        force_ascii=False,
        double_precision=15,
    )
    return len(frame)


def require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Data sumber tidak ditemukan: {path}")


def export_geojson() -> int:
    require_path(GEOJSON_PATH)
    with GEOJSON_PATH.open(encoding="utf-8") as source:
        geojson = json.load(source)

    features = geojson.get("features", [])
    identifiers = [
        str(feature.get("properties", {}).get("locationid", ""))
        for feature in features
    ]
    counts = pd.Series(identifiers).value_counts()
    compact_features: list[dict[str, Any]] = []
    for feature in features:
        properties = feature.get("properties", {})
        location_id = str(properties.get("locationid", ""))
        geometry = feature.get("geometry")
        if not location_id or counts.get(location_id, 0) != 1 or not geometry:
            continue
        compact_features.append(
            {
                "type": "Feature",
                "properties": {
                    "locationid": location_id,
                    "zone": properties.get("zone"),
                    "borough": properties.get("borough"),
                },
                "geometry": geometry,
            }
        )

    if not compact_features:
        raise ValueError(f"Tidak ada poligon Taxi Zone valid dalam {GEOJSON_PATH}")
    output = {"type": "FeatureCollection", "features": compact_features}
    path = DATA_DIR / "taxi_zones.json"
    with path.open("w", encoding="utf-8") as target:
        json.dump(output, target, ensure_ascii=False, separators=(",", ":"))
    return len(compact_features)


def export_graph() -> tuple[int, int]:
    for filename in ("graph_summary.csv", "zone_metrics_by_month.csv", "edges_by_month.csv"):
        require_path(GRAPH_DIR / filename)

    summary = pd.read_csv(GRAPH_DIR / "graph_summary.csv")
    zones = pd.read_csv(GRAPH_DIR / "zone_metrics_by_month.csv")
    edges = pd.read_csv(GRAPH_DIR / "edges_by_month.csv")
    write_records("graph_summary.json", summary)
    write_records("zone_metrics.json", zones)

    period_edges: list[pd.DataFrame] = []
    for month in sorted(edges["month"].astype(str).unique()):
        period_edges.append(
            edges.loc[edges["month"].astype(str) == month]
            .nlargest(20, "trip_count")
            .assign(period=month)
        )
    combined = (
        edges.groupby(["PULocationID", "DOLocationID"], as_index=False)["trip_count"]
        .sum()
        .nlargest(20, "trip_count")
        .assign(period="ALL_2026-01_2026-06")
    )
    period_edges.append(combined)
    top_edges = pd.concat(period_edges, ignore_index=True)
    write_records("top_graph_edges.json", top_edges)
    return len(zones), len(top_edges)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    exported: dict[str, int] = {}

    for key, table in TABLES.items():
        path = ANALYSIS_DIR / f"{table}.parquet"
        require_path(path)
        exported[f"{key}.json"] = write_records(key + ".json", pd.read_parquet(path))

    zone_weather_path = ANALYSIS_DIR / "pickup_zone_hour_weather_volume.parquet"
    require_path(zone_weather_path)
    zone_weather = pd.read_parquet(zone_weather_path)
    zone_columns = ["PULocationID", "pickup_borough", "pickup_zone", "weather_category"]
    zone_weather = (
        zone_weather.groupby(zone_columns, dropna=False, as_index=False)["trip_count"]
        .sum()
    )
    exported["zone_weather.json"] = write_records("zone_weather.json", zone_weather)

    for report in REPORTS:
        path = QUALITY_DIR / report
        require_path(path)
        exported[report.replace(".csv", ".json")] = write_records(
            report.replace(".csv", ".json"), pd.read_csv(path)
        )

    geojson_features = export_geojson()
    zone_count, edge_count = export_graph()
    exported["taxi_zones.json"] = geojson_features

    manifest = {
        "source": "NYC TLC HVFHV and Open-Meteo validated aggregates",
        "period": "2026-01 through 2026-06",
        "raw_trip_rows_included": False,
        "records": exported,
        "graph_zone_records": zone_count,
        "top_graph_edge_records": edge_count,
    }
    with (DATA_DIR / "manifest.json").open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)

    for name, count in exported.items():
        print(f"{name}: {count:,} records")
    print(f"Static data written to {DATA_DIR}")


if __name__ == "__main__":
    main()
