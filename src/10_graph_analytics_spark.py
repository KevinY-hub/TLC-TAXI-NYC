"""Build monthly directed, weighted Taxi Zone graphs from integrated HVFHV trips.

Run from the project root in WSL:
    python src/10_graph_analytics_spark.py

Spark performs the large-data aggregation. Only the aggregated origin-destination
edge table (bounded by the number of distinct zone pairs) is streamed to the
driver for graph metrics; individual trips are never collected or converted to
Pandas. Graph metrics use the Python standard library, so NetworkX is not needed.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TOTAL_ROWS = 126_771_977
EXPECTED_MONTHS = [f"2026-{month:02d}" for month in range(1, 7)]
MAX_AGGREGATED_EDGES = 1_000_000
ESTIMATED_BYTES_PER_EDGE = 512
LOG = logging.getLogger("hvfhs_graph_analytics")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def id_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, value)


def make_graph() -> dict[str, Any]:
    return {
        "nodes": set(),
        "edge_pairs": set(),
        "edges": 0,
        "trips": 0,
        "self_loop_edges": 0,
        "self_loop_trips": 0,
        "in_strength": defaultdict(int),
        "out_strength": defaultdict(int),
        "in_neighbors": defaultdict(set),
        "out_neighbors": defaultdict(set),
        "undirected_neighbors": defaultdict(set),
    }


def add_edge(graph: dict[str, Any], origin: str, destination: str, weight: int) -> None:
    graph["nodes"].update((origin, destination))
    pair = (origin, destination)
    if pair not in graph["edge_pairs"]:
        graph["edge_pairs"].add(pair)
        graph["edges"] += 1
        if origin == destination:
            graph["self_loop_edges"] += 1
    graph["trips"] += weight
    graph["out_strength"][origin] += weight
    graph["in_strength"][destination] += weight
    graph["out_neighbors"][origin].add(destination)
    graph["in_neighbors"][destination].add(origin)
    graph["undirected_neighbors"][origin].add(destination)
    graph["undirected_neighbors"][destination].add(origin)
    if origin == destination:
        graph["self_loop_trips"] += weight


def weak_component_count(graph: dict[str, Any]) -> int:
    remaining = set(graph["nodes"])
    components = 0
    adjacency = graph["undirected_neighbors"]
    while remaining:
        components += 1
        seed = remaining.pop()
        stack = [seed]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def label_for(location_id: str, labels: dict[str, dict[str, str]]) -> tuple[str, str]:
    row = labels.get(location_id, {})
    return row.get("borough", ""), row.get("zone", "")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "output" / "integrated" / "taxi_zones"
    zone_lookup_path = project_root / "data" / "reference" / "taxi_zones" / "taxi_zone_lookup.csv"
    reference_monthly = project_root / "output" / "quality" / "cleaning_by_month.csv"
    analysis_root = project_root / "output" / "analysis" / "graph_analytics"
    quality_root = project_root / "output" / "quality" / "graph_analytics"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    analysis_dir = analysis_root / run_id
    quality_dir = quality_root / run_id

    if not input_path.is_dir():
        raise FileNotFoundError(f"Taxi Zone integrated Parquet not found: {input_path}")
    if not reference_monthly.is_file():
        raise FileNotFoundError(f"Cleaning monthly row-count reference not found: {reference_monthly}")
    if not zone_lookup_path.is_file():
        raise FileNotFoundError(f"Taxi Zone lookup CSV not found: {zone_lookup_path}")
    if analysis_dir.exists() or quality_dir.exists():
        raise FileExistsError(f"Run output already exists; refusing to overwrite: {run_id}")
    analysis_dir.mkdir(parents=True)
    quality_dir.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Existing cleaning report is the reference for per-month counts.
    with reference_monthly.open(newline="", encoding="utf-8-sig") as stream:
        monthly_reference = {
            row["month"]: int(row["after"])
            for row in csv.DictReader(stream)
            if row.get("month") in EXPECTED_MONTHS
        }
    if sorted(monthly_reference) != EXPECTED_MONTHS:
        raise ValueError(f"Could not read expected six-month counts from {reference_monthly}")
    labels: dict[str, dict[str, str]] = {}
    with zone_lookup_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        needed_lookup_columns = {"LocationID", "Borough", "Zone"}
        if not reader.fieldnames or not needed_lookup_columns.issubset(reader.fieldnames):
            raise ValueError(f"Taxi Zone lookup lacks required columns: {sorted(needed_lookup_columns)}")
        for row in reader:
            location_id = (row.get("LocationID") or "").strip()
            if not location_id:
                continue
            if location_id in labels:
                raise ValueError(f"Taxi Zone lookup has duplicate LocationID: {location_id}")
            labels[location_id] = {
                "borough": (row.get("Borough") or "").strip(),
                "zone": (row.get("Zone") or "").strip(),
            }

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    spark = None
    edge_agg = None
    started = utc_now()
    try:
        from pyspark import StorageLevel
        from pyspark.sql import SparkSession, functions as F

        spark = (
            SparkSession.builder.appName("HVFHS-Graph-Analytics")
            .master("local[2]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.default.parallelism", "2")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.adaptive.enabled", "true")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        trips = spark.read.parquet(str(input_path))
        required = {"PULocationID", "DOLocationID", "pickup_date"}
        missing_columns = sorted(required - set(trips.columns))
        if missing_columns:
            raise ValueError(f"Input schema lacks required columns: {missing_columns}")

        LOG.info("Input schema verified; selecting only graph keys and zone labels")
        edges = (
            trips.select(
                F.date_format(F.col("pickup_date"), "yyyy-MM").alias("month"),
                F.col("PULocationID").cast("string").alias("origin"),
                F.col("DOLocationID").cast("string").alias("destination"),
            )
            .groupBy("month", "origin", "destination")
            .agg(F.count(F.lit(1)).cast("long").alias("trip_count"))
            .persist(StorageLevel.DISK_ONLY)
        )

        # Count the compact aggregate first; do not bring it to the driver unless
        # the observed size and current memory make that safe.
        aggregate_edge_count = edges.count()
        memory_available = available_memory_bytes()
        estimate = aggregate_edge_count * ESTIMATED_BYTES_PER_EDGE
        memory_limit = min(2 * 1024**3, int(memory_available * 0.20)) if memory_available else 512 * 1024**2
        if aggregate_edge_count > MAX_AGGREGATED_EDGES or estimate > memory_limit:
            raise MemoryError(
                "Aggregated edge table exceeds safe driver limit: "
                f"{aggregate_edge_count:,} rows, estimated {estimate / 1024**2:.1f} MiB; "
                f"safe limit {memory_limit / 1024**2:.1f} MiB. No trip rows were collected."
            )
        LOG.info(
            "Aggregated edge rows=%s, estimated driver graph memory=%.1f MiB, available RAM=%.1f MiB",
            f"{aggregate_edge_count:,}", estimate / 1024**2,
            memory_available / 1024**2 if memory_available else -1,
        )

        graphs: dict[str, dict[str, Any]] = {month: make_graph() for month in EXPECTED_MONTHS}
        graphs["ALL_2026-01_2026-06"] = make_graph()
        month_trip_rows: defaultdict[str, int] = defaultdict(int)
        null_origin_trip_rows = 0
        null_destination_trip_rows = 0
        unknown_month_trip_rows = 0
        edge_path = analysis_dir / "edges_by_month.csv"
        with edge_path.open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["month", "PULocationID", "DOLocationID", "trip_count"])
            for row in edges.toLocalIterator():
                month = row["month"] or "UNKNOWN_DATE"
                origin = row["origin"]
                destination = row["destination"]
                weight = int(row["trip_count"])
                writer.writerow([month, origin or "", destination or "", weight])
                month_trip_rows[month] += weight
                if month not in EXPECTED_MONTHS:
                    unknown_month_trip_rows += weight
                if origin is None:
                    null_origin_trip_rows += weight
                if destination is None:
                    null_destination_trip_rows += weight
                if origin is None or destination is None or month not in EXPECTED_MONTHS:
                    continue
                add_edge(graphs[month], str(origin), str(destination), weight)
                add_edge(graphs["ALL_2026-01_2026-06"], str(origin), str(destination), weight)

        graph_summary_rows: list[dict[str, Any]] = []
        node_metric_rows: list[dict[str, Any]] = []
        top_origin_rows: list[dict[str, Any]] = []
        top_destination_rows: list[dict[str, Any]] = []
        for month in EXPECTED_MONTHS + ["ALL_2026-01_2026-06"]:
            graph = graphs[month]
            node_count = len(graph["nodes"])
            non_loop_edges = graph["edges"] - graph["self_loop_edges"]
            possible_non_loop_edges = node_count * (node_count - 1)
            density = non_loop_edges / possible_non_loop_edges if possible_non_loop_edges else 0.0
            graph_summary_rows.append({
                "period": month,
                "trip_rows_represented": graph["trips"],
                "active_nodes": node_count,
                "unique_directed_od_pairs": graph["edges"],
                "non_self_loop_od_pairs": non_loop_edges,
                "self_loop_od_pairs": graph["self_loop_edges"],
                "self_loop_trip_rows": graph["self_loop_trips"],
                "directed_density_excluding_self_loops": density,
                "weakly_connected_components": weak_component_count(graph),
            })

            zone_rows: list[dict[str, Any]] = []
            for location_id in sorted(graph["nodes"], key=id_sort_key):
                borough, zone = label_for(location_id, labels)
                out_strength = graph["out_strength"][location_id]
                in_strength = graph["in_strength"][location_id]
                row_data = {
                    "period": month,
                    "LocationID": location_id,
                    "borough": borough,
                    "zone": zone,
                    "out_degree": len(graph["out_neighbors"][location_id]),
                    "in_degree": len(graph["in_neighbors"][location_id]),
                    "out_strength_trips": out_strength,
                    "in_strength_trips": in_strength,
                    "in_plus_out_trip_rows": out_strength + in_strength,
                }
                node_metric_rows.append(row_data)
                zone_rows.append(row_data)
            for direction, field, target in (
                ("origin", "out_strength_trips", top_origin_rows),
                ("destination", "in_strength_trips", top_destination_rows),
            ):
                for rank, item in enumerate(sorted(zone_rows, key=lambda x: (-x[field], id_sort_key(x["LocationID"])))[:20], start=1):
                    target.append({"period": month, "rank": rank, "role": direction, **item})

        all_period_nodes = graphs["ALL_2026-01_2026-06"]["nodes"]
        unmapped_node_ids = sorted(all_period_nodes - set(labels), key=id_sort_key)
        expected_total = sum(monthly_reference.values())
        actual_total = sum(month_trip_rows.values())
        month_checks = {
            month: {
                "expected": monthly_reference[month],
                "actual": month_trip_rows.get(month, 0),
                "passed": monthly_reference[month] == month_trip_rows.get(month, 0),
            }
            for month in EXPECTED_MONTHS
        }
        all_months_present = all(month_trip_rows.get(month, 0) > 0 for month in EXPECTED_MONTHS)
        validation_passed = (
            actual_total == EXPECTED_TOTAL_ROWS
            and expected_total == EXPECTED_TOTAL_ROWS
            and all(check["passed"] for check in month_checks.values())
            and all_months_present
            and unknown_month_trip_rows == 0
            and null_origin_trip_rows == 0
            and null_destination_trip_rows == 0
        )

        validation_rows = [
            {"check": "total_trip_rows_from_aggregated_edges", "expected": EXPECTED_TOTAL_ROWS, "actual": actual_total, "status": "PASS" if actual_total == EXPECTED_TOTAL_ROWS else "FAIL"},
            {"check": "sum_monthly_reference_rows", "expected": EXPECTED_TOTAL_ROWS, "actual": expected_total, "status": "PASS" if expected_total == EXPECTED_TOTAL_ROWS else "FAIL"},
            {"check": "aggregated_edge_rows", "expected": "bounded aggregate; not trip-level collection", "actual": aggregate_edge_count, "status": "PASS"},
            {"check": "rows_with_null_PULocationID", "expected": 0, "actual": null_origin_trip_rows, "status": "PASS" if null_origin_trip_rows == 0 else "FAIL"},
            {"check": "rows_with_null_DOLocationID", "expected": 0, "actual": null_destination_trip_rows, "status": "PASS" if null_destination_trip_rows == 0 else "FAIL"},
            {"check": "rows_outside_expected_months_or_null_pickup_date", "expected": 0, "actual": unknown_month_trip_rows, "status": "PASS" if unknown_month_trip_rows == 0 else "FAIL"},
            {"check": "all_six_expected_months_present", "expected": ",".join(EXPECTED_MONTHS), "actual": ",".join(m for m in EXPECTED_MONTHS if month_trip_rows.get(m, 0) > 0), "status": "PASS" if all_months_present else "FAIL"},
            {"check": "observed_nodes_without_lookup_label", "expected": 0, "actual": len(unmapped_node_ids), "status": "PASS" if not unmapped_node_ids else "INFO"},
        ]
        for month, check in month_checks.items():
            validation_rows.append({"check": f"trip_rows_{month}", "expected": check["expected"], "actual": check["actual"], "status": "PASS" if check["passed"] else "FAIL"})

        write_csv(analysis_dir / "graph_summary.csv", list(graph_summary_rows[0].keys()), graph_summary_rows)
        write_csv(analysis_dir / "zone_metrics_by_month.csv", list(node_metric_rows[0].keys()), node_metric_rows)
        write_csv(analysis_dir / "top_origin_zones_by_month.csv", list(top_origin_rows[0].keys()), top_origin_rows)
        write_csv(analysis_dir / "top_destination_zones_by_month.csv", list(top_destination_rows[0].keys()), top_destination_rows)
        write_csv(quality_dir / "validation.csv", ["check", "expected", "actual", "status"], validation_rows)

        metadata = {
            "status": "VALIDATION_PASSED" if validation_passed else "VALIDATION_FAILED",
            "run_id": run_id,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "source": str(input_path),
            "spark_master": "local[2]",
            "spark_session_timezone": "UTC",
            "analysis": "directed weighted origin-destination Taxi Zone graph by pickup month, plus six-month aggregate",
            "nodes": "observed non-null Taxi Zone IDs at either endpoint; unresolved IDs retained by ID",
            "edges": "directed PULocationID -> DOLocationID; weight is number of trip rows; self-loops retained",
            "density_definition": "non-self-loop directed OD pairs / (n * (n - 1)); self-loop pairs reported separately",
            "weak_components_definition": "connected components after treating directed edges as undirected",
            "networkx_required": False,
            "spark_aggregated_edge_rows": aggregate_edge_count,
            "driver_memory_estimate_bytes": estimate,
            "available_memory_at_check_bytes": memory_available,
            "total_trip_rows_from_edges": actual_total,
            "monthly_trip_rows": dict(month_trip_rows),
            "monthly_reference_rows": monthly_reference,
            "monthly_checks": month_checks,
            "null_origin_trip_rows": null_origin_trip_rows,
            "null_destination_trip_rows": null_destination_trip_rows,
            "unknown_month_trip_rows": unknown_month_trip_rows,
            "observed_nodes": len(all_period_nodes),
            "observed_nodes_without_lookup_label": unmapped_node_ids,
        }
        with (quality_dir / "run_metadata.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
        report = [
            "# HVFHV Taxi Zone Graph Analytics",
            "",
            f"Run: `{run_id}` | Status: **{metadata['status']}**",
            "",
            "## Design",
            "- Directed weighted graph: origin `PULocationID` -> destination `DOLocationID`.",
            "- Edge weight: number of trip rows for each ordered pair.",
            "- Monthly graphs for January-June 2026, plus one six-month aggregate graph.",
            "- Self-loops are retained and reported separately. Density excludes self-loops and uses n*(n-1) as denominator.",
            "- Spark aggregates the trip data. Only the compact aggregated edge rows and distinct zone labels are streamed to the driver.",
            "- In/out degree count distinct neighboring zones; in/out strength sum trip counts.",
            "- Weak component count treats directed edges as undirected.",
            "",
            "## Validation",
            f"- Expected trip rows: {EXPECTED_TOTAL_ROWS:,}",
            f"- Trip rows represented by edge weights: {actual_total:,}",
            f"- Null pickup IDs: {null_origin_trip_rows:,}; null drop-off IDs: {null_destination_trip_rows:,}",
            f"- Rows outside the six expected months or with null pickup date: {unknown_month_trip_rows:,}",
            f"- Observed graph nodes without Taxi Zone lookup labels: {len(unmapped_node_ids)} ({', '.join(unmapped_node_ids) if unmapped_node_ids else 'none'}).",
            f"- Monthly row counts compared with the existing cleaning report; details are in `output/quality/graph_analytics/{run_id}/validation.csv`.",
            "",
            "## Graph summary",
            "| Period | Trips | Nodes | Directed OD pairs | Self-loop pairs | Density (no loops) | Weak components |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in graph_summary_rows:
            report.append(f"| {r['period']} | {r['trip_rows_represented']:,} | {r['active_nodes']:,} | {r['unique_directed_od_pairs']:,} | {r['self_loop_od_pairs']:,} | {r['directed_density_excluding_self_loops']:.6f} | {r['weakly_connected_components']:,} |")
        report += [
            "",
            "## Output files",
            "- `edges_by_month.csv`: compact edge list with monthly trip-count weights.",
            "- `graph_summary.csv`: network-level counts and topology summaries.",
            "- `zone_metrics_by_month.csv`: per-zone degree and trip strength by month and for the full period.",
            "- `top_origin_zones_by_month.csv` and `top_destination_zones_by_month.csv`: top 20 zones by outgoing/incoming trip volume.",
            "- Validation and run configuration are stored in the matching quality run directory.",
            "",
        ]
        with (analysis_dir / "README.md").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(report))
        print(f"Graph analytics status: {metadata['status']}")
        print(f"Trip rows represented: {actual_total:,} / {EXPECTED_TOTAL_ROWS:,}")
        print(f"Aggregated edges: {aggregate_edge_count:,}")
        print(f"Analysis output: {analysis_dir}")
        print(f"Quality output: {quality_dir}")
        if not validation_passed:
            LOG.error("Graph analytics completed but validation failed; inspect %s", quality_dir / "validation.csv")
            return 2
        return 0
    except Exception:
        error = traceback.format_exc()
        LOG.error("Graph analytics failed: %s", error)
        error_file = quality_dir / "error.txt"
        if quality_dir.exists() and not error_file.exists():
            error_file.write_text(error, encoding="utf-8")
        print(error, file=sys.stderr)
        return 1
    finally:
        if edge_agg is not None:
            try:
                edge_agg.unpersist(blocking=False)
            except Exception:
                pass
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
