"""Create paper-ready static visualizations from a validated graph run.

This reads only compact CSV reports and the local Taxi Zone GeoJSON; it does
not launch Spark or scan the trip Parquet. Requires Pillow (already available
in the project Windows Python environment).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROOT = PROJECT_ROOT / "output" / "analysis" / "graph_analytics"
QUALITY_ROOT = PROJECT_ROOT / "output" / "quality" / "graph_analytics"
ZONE_GEOJSON = PROJECT_ROOT / "data" / "reference" / "taxi_zones" / "nyc_taxi_zones.geojson"
ZONE_ISSUES = PROJECT_ROOT / "output" / "quality" / "taxi_zone_integration" / "geometry_issues.csv"
ZONE_LOOKUP = PROJECT_ROOT / "data" / "reference" / "taxi_zones" / "taxi_zone_lookup.csv"
MONTHS = [f"2026-{i:02d}" for i in range(1, 7)]

NAVY = "#17324D"
INK = "#243447"
MUTED = "#5A6C7D"
GRID = "#DCE3EA"
BLUE = "#3979B7"
TEAL = "#2B9C8D"
ORANGE = "#E69A2D"
PALE = "#F3F6FA"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf")
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def latest_valid_run() -> str:
    candidates: list[str] = []
    if not QUALITY_ROOT.is_dir():
        raise FileNotFoundError(f"Graph validation reports are missing: {QUALITY_ROOT}")
    for child in QUALITY_ROOT.iterdir():
        status_path = child / "run_metadata.json"
        if not status_path.is_file():
            continue
        try:
            if json.loads(status_path.read_text(encoding="utf-8")).get("status") == "VALIDATION_PASSED":
                candidates.append(child.name)
        except (OSError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError("No VALIDATION_PASSED graph run was found.")
    return sorted(candidates)[-1]


def verify_run(run_id: str, graph_dir: Path) -> None:
    meta = json.loads((QUALITY_ROOT / run_id / "run_metadata.json").read_text(encoding="utf-8"))
    if meta.get("status") != "VALIDATION_PASSED":
        raise ValueError(f"Refusing to visualize non-passing graph run {run_id}: {meta.get('status')}")
    expected = int(meta["total_trip_rows_from_edges"])
    summary = read_csv(graph_dir / "graph_summary.csv")
    all_row = next(r for r in summary if r["period"] == "ALL_2026-01_2026-06")
    if int(all_row["trip_rows_represented"]) != expected:
        raise ValueError("Graph summary row total does not match run metadata.")


def write_header(draw: ImageDraw.ImageDraw, width: int, title: str, subtitle: str) -> None:
    draw.text((70, 45), title, font=font(42, True), fill=NAVY)
    draw.text((73, 108), subtitle, font=font(22), fill=MUTED)


def graph_flowchart(destination: Path) -> None:
    """Draw the reproducible Spark-to-graph analysis flow for Chapter VI."""
    im = Image.new("RGB", (1500, 1580), WHITE)
    d = ImageDraw.Draw(im)
    write_header(d, 1500, "ALUR GRAPH ANALYTICS HVFHV", "Agregasi Spark dan analisis jaringan Taxi Zone")

    def box(rect: tuple[int, int, int, int], lines: list[str], fill: str, outline: str, size: int = 23) -> None:
        d.rounded_rectangle(rect, radius=14, fill=fill, outline=outline, width=4)
        fnt = font(size, len(lines) == 1)
        heights = []
        for line in lines:
            bb = d.textbbox((0, 0), line, font=fnt)
            heights.append((bb[2] - bb[0], bb[3] - bb[1]))
        gap = 9
        total = sum(h for _, h in heights) + gap * (len(lines) - 1)
        y = rect[1] + (rect[3] - rect[1] - total) // 2
        for line, (ww, hh) in zip(lines, heights):
            d.text(((rect[0] + rect[2] - ww) // 2, y), line, font=fnt, fill=NAVY if fill != BLUE else INK)
            y += hh + gap

    def arrow(a: tuple[int, int], b: tuple[int, int]) -> None:
        import math
        d.line([a, b], fill="#52677b", width=5)
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        length = 18
        d.polygon([b,
                   (b[0] - length * math.cos(angle - 0.58), b[1] - length * math.sin(angle - 0.58)),
                   (b[0] - length * math.cos(angle + 0.58), b[1] - length * math.sin(angle + 0.58))], fill="#52677b")

    box((250, 180, 1250, 320), ["Dataset Taxi Zone terintegrasi", "126.771.977 perjalanan | pickup_date, PULocationID, DOLocationID"], "#F6B94B", NAVY, 22)
    arrow((750, 320), (750, 390))
    box((250, 390, 1250, 545), ["Agregasi terdistribusi dengan Spark", "Group by bulan pickup + zona asal + zona tujuan", "Bobot sisi = jumlah perjalanan"], "#EAF2FA", "#67839D", 21)
    arrow((750, 545), (750, 615))
    # Branch into monthly graphs and a six-month aggregate graph.
    d.line([(750, 615), (405, 615), (1095, 615)], fill="#52677b", width=5)
    arrow((405, 615), (405, 660))
    arrow((1095, 615), (1095, 660))
    box((90, 660, 720, 820), ["Graph bulanan", "6 jaringan berarah dan berbobot", "Januari-Juni 2026"], "#E7F7F4", "#2B9C8D", 21)
    box((780, 660, 1410, 820), ["Graph gabungan", "1 jaringan berarah dan berbobot", "Januari-Juni 2026"], "#EEF3FB", "#3979B7", 21)
    arrow((405, 820), (405, 890)); arrow((1095, 820), (1095, 890))
    box((90, 890, 720, 1100), ["Metrik jaringan", "Jumlah node dan pasangan OD", "Density, self-loop, komponen lemah", "In/out degree dan in/out strength"], "#E7F7F4", "#2B9C8D", 20)
    box((780, 890, 1410, 1100), ["Metrik jaringan", "Jumlah node dan pasangan OD", "Density, self-loop, komponen lemah", "In/out degree dan in/out strength"], "#EEF3FB", "#3979B7", 20)
    arrow((405, 1100), (405, 1180)); arrow((1095, 1100), (1095, 1180))
    d.line([(405, 1180), (750, 1180), (1095, 1180)], fill="#52677b", width=5)
    arrow((750, 1180), (750, 1245))
    box((190, 1245, 1310, 1415), ["Output untuk Bab VI dan dashboard", "Ringkasan per bulan | top origin/destination | arus OD", "Peta strength zona dengan batas geometri yang tersedia"], "#FFF3DC", "#C28A24", 21)
    d.text((90, 1480), "Lokasi tanpa poligon valid tetap ada dalam tabel graph; hanya tidak tampil pada peta.", font=font(18), fill=MUTED)
    im.save(destination, dpi=(300, 300), optimize=True)


def monthly_chart(rows: list[dict[str, str]], destination: Path) -> None:
    monthly = [r for r in rows if r["period"] in MONTHS]
    im = Image.new("RGB", (1900, 1120), WHITE)
    d = ImageDraw.Draw(im)
    write_header(d, 1900, "Perubahan Struktur Graph per Bulan", "HVFHV OD graph | Januari-Juni 2026")

    panels = [
        ("Pasangan OD berarah unik", [int(r["unique_directed_od_pairs"]) for r in monthly], BLUE, 0, 70000, "jumlah pasangan"),
        ("Kepadatan berarah", [float(r["directed_density_excluding_self_loops"]) for r in monthly], TEAL, 0.84, 0.89, "rasio; sumbu dimulai dari 0,84"),
    ]
    for pidx, (title, values, color, ymin, ymax, ylabel) in enumerate(panels):
        x0 = 100 + pidx * 930
        plot = (x0 + 90, 260, x0 + 840, 850)
        d.text((x0 + 90, 190), title, font=font(27, True), fill=NAVY)
        d.text((x0 + 90, 225), ylabel, font=font(17), fill=MUTED)
        left, top, right, bottom = plot
        for tick in range(5):
            value = ymin + (ymax - ymin) * tick / 4
            y = bottom - (bottom - top) * tick / 4
            d.line((left, y, right, y), fill=GRID, width=2)
            label = f"{value:.2f}" if pidx else f"{int(value/1000)}k"
            d.text((left - 60, y - 12), label, font=font(16), fill=MUTED)
        slot = (right - left) / len(monthly)
        points = []
        for i, (row, value) in enumerate(zip(monthly, values)):
            x = left + slot * (i + 0.5)
            y = bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
            points.append((x, y))
            if pidx == 0:
                bar_w = 62
                d.rounded_rectangle((x - bar_w / 2, y, x + bar_w / 2, bottom), radius=5, fill=color)
                d.text((x - 33, y - 31), f"{value:,}", font=font(15, True), fill=INK)
            else:
                d.text((x - 29, y - 39), f"{value:.3f}", font=font(17, True), fill=INK)
            d.text((x - 37, bottom + 18), row["period"][5:], font=font(18), fill=INK)
        if pidx == 1:
            d.line(points, fill=color, width=6)
            for x, y in points:
                d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color, outline=WHITE, width=2)
    d.text((75, 1000), "Self-loop pasangan dilaporkan terpisah dan tidak masuk ke perhitungan kepadatan.", font=font(18), fill=MUTED)
    im.save(destination, dpi=(300, 300), optimize=True)


def top_flows_chart(edge_csv: Path, lookup: dict[str, dict[str, str]], destination: Path) -> None:
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for row in read_csv(edge_csv):
        if row["PULocationID"] and row["DOLocationID"]:
            counts[(row["PULocationID"], row["DOLocationID"])] += int(row["trip_count"])
    top = sorted(counts.items(), key=lambda item: (-item[1], int(item[0][0]), int(item[0][1])))[:15]
    im = Image.new("RGB", (2100, 1460), WHITE)
    d = ImageDraw.Draw(im)
    write_header(d, 2100, "15 Arus Asal-Tujuan Terpadat", "Bobot sisi = jumlah perjalanan selama Januari-Juni 2026; arah dibaca dari asal ke tujuan")
    left, right, top_y, bottom = 560, 1930, 220, 1300
    max_v = max(value for _, value in top)
    for i, ((origin, destination_id), value) in enumerate(top):
        y = top_y + i * 70
        o = lookup.get(origin, {}).get("Zone", f"LocationID {origin}")
        dest = lookup.get(destination_id, {}).get("Zone", f"LocationID {destination_id}")
        label = f"{o} ({origin})  ->  {dest} ({destination_id})"
        d.text((70, y + 8), label[:58], font=font(19), fill=INK)
        length = (right - left) * value / max_v
        d.rounded_rectangle((left, y, left + length, y + 42), radius=7, fill=BLUE)
        d.text((min(left + length + 12, right - 120), y + 8), f"{value:,}", font=font(18, True), fill=NAVY)
        d.line((left, y + 53, right, y + 53), fill=GRID, width=1)
    d.text((70, 1355), "Catatan: arus di luar 15 pasangan teratas tetap terdapat pada edges_by_month.csv.", font=font(18), fill=MUTED)
    im.save(destination, dpi=(300, 300), optimize=True)


def top_zones_chart(origin_rows: list[dict[str, str]], destination_rows: list[dict[str, str]], output: Path) -> None:
    period = "ALL_2026-01_2026-06"
    groups = [
        ("Zona asal teratas (out-strength)", [r for r in origin_rows if r["period"] == period][:10], "out_strength_trips", BLUE),
        ("Zona tujuan teratas (in-strength)", [r for r in destination_rows if r["period"] == period][:10], "in_strength_trips", TEAL),
    ]
    im = Image.new("RGB", (2200, 1180), WHITE)
    d = ImageDraw.Draw(im)
    write_header(d, 2200, "Zona dengan Volume Perjalanan Keluar dan Masuk Tertinggi", "Kekuatan berbobot (strength) pada graph gabungan Januari-Juni 2026")
    for idx, (title, rows, column, color) in enumerate(groups):
        x0 = 70 + idx * 1080
        d.text((x0 + 20, 190), title, font=font(25, True), fill=NAVY)
        max_v = max(int(r[column]) for r in rows)
        for i, row in enumerate(rows):
            y = 245 + i * 80
            label = f"{row['zone']} ({row['LocationID']})"
            d.text((x0 + 15, y + 4), label[:32], font=font(18), fill=INK)
            bx = x0 + 410
            bw = 560 * int(row[column]) / max_v
            d.rounded_rectangle((bx, y, bx + bw, y + 38), radius=6, fill=color)
            d.text((min(bx + bw + 9, x0 + 965), y + 6), f"{int(row[column]):,}", font=font(16, True), fill=NAVY)
    d.text((72, 1070), "Out-strength = perjalanan berangkat dari zona; in-strength = perjalanan tiba di zona.", font=font(19), fill=MUTED)
    im.save(output, dpi=(300, 300), optimize=True)


def geometry_polygons(geojson_path: Path, invalid_ids: set[str], metrics: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    id_frequency: defaultdict[str, int] = defaultdict(int)
    for feature in features:
        id_frequency[str(feature.get("properties", {}).get("locationid", ""))] += 1
    result = []
    for feature in features:
        props = feature.get("properties") or {}
        location_id = str(props.get("locationid", ""))
        geometry = feature.get("geometry") or {}
        if not location_id or id_frequency[location_id] != 1 or location_id in invalid_ids or location_id not in metrics:
            continue
        coords = geometry.get("coordinates")
        if not coords:
            continue
        if geometry.get("type") == "Polygon":
            polygons = [coords]
        elif geometry.get("type") == "MultiPolygon":
            polygons = coords
        else:
            continue
        outer_rings = []
        for polygon in polygons:
            if not polygon or not polygon[0]:
                continue
            ring = [(float(point[0]), float(point[1])) for point in polygon[0] if len(point) >= 2]
            if len(ring) >= 3 and all(math.isfinite(x) and math.isfinite(y) for x, y in ring):
                outer_rings.append(ring)
        if outer_rings:
            result.append({"id": location_id, "rings": outer_rings})
    return result


def color_scale(value: int, maximum: int) -> tuple[int, int, int]:
    # Log compression followed by a power curve preserves the ranking while
    # improving visible contrast across NYC zones with different trip volumes.
    t = (math.log1p(value) / math.log1p(maximum)) ** 2.5 if maximum > 0 else 0.0
    start = (235, 243, 250)
    end = (25, 73, 116)
    return tuple(round(a + (b - a) * t) for a, b in zip(start, end))


def flow_maps(node_rows: list[dict[str, str]], invalid_ids: set[str], output: Path) -> dict[str, Any]:
    all_period = "ALL_2026-01_2026-06"
    metrics: dict[str, dict[str, int]] = {}
    for row in node_rows:
        if row["period"] == all_period:
            metrics[row["LocationID"]] = {
                "out": int(row["out_strength_trips"]),
                "in": int(row["in_strength_trips"]),
            }
    polys = geometry_polygons(ZONE_GEOJSON, invalid_ids, metrics)
    if not polys:
        raise ValueError("No unique valid Taxi Zone polygons match graph node metrics.")
    im = Image.new("RGB", (2200, 1450), WHITE)
    d = ImageDraw.Draw(im)
    write_header(d, 2200, "Pola Arus Perjalanan per Taxi Zone", "Kiri: perjalanan keluar (out-strength) | kanan: perjalanan masuk (in-strength)")
    # Local equirectangular view of GeoJSON longitude/latitude (WGS84).
    reference_lat = 40.7
    cos_lat = math.cos(math.radians(reference_lat))
    all_points = [point for item in polys for ring in item["rings"] for point in ring]
    xs = [lon * cos_lat for lon, _ in all_points]
    ys = [lat for _, lat in all_points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xspan, yspan = xmax - xmin, ymax - ymin
    panels = [(70, 220, 1030, 1240, "out", "Perjalanan keluar per zona"), (1130, 220, 2090, 1240, "in", "Perjalanan masuk per zona")]
    for x1, y1, x2, y2, metric, label in panels:
        d.text((x1 + 50, y1 - 44), label, font=font(25, True), fill=NAVY)
        pad = 28
        avail_w, avail_h = x2 - x1 - 2 * pad, y2 - y1 - 2 * pad
        scale = min(avail_w / xspan, avail_h / yspan)
        offset_x = x1 + pad + (avail_w - xspan * scale) / 2
        offset_y = y1 + pad + (avail_h - yspan * scale) / 2
        panel_max = max(m[metric] for m in metrics.values())
        for item in polys:
            value = metrics[item["id"]][metric]
            fill = color_scale(value, panel_max)
            for ring in item["rings"]:
                pts = [(offset_x + (lon * cos_lat - xmin) * scale,
                        offset_y + (ymax - lat) * scale) for lon, lat in ring]
                d.polygon(pts, fill=fill, outline=WHITE)
    # Each map has its own logarithmic range. This is stated to avoid treating
    # equal shades as equal absolute volumes across the two panels.
    legend_y = 1305
    steps, step_w = 140, 2
    for legend_x, metric in ((430, "out"), (1490, "in")):
        panel_max = max(m[metric] for m in metrics.values())
        d.text((legend_x - 260, legend_y - 3), "Skala log volume trip", font=font(17), fill=MUTED)
        for i in range(steps):
            value = round(panel_max * i / (steps - 1))
            d.rectangle((legend_x + i * step_w, legend_y, legend_x + i * step_w + step_w, legend_y + 22), fill=color_scale(value, panel_max))
        d.text((legend_x, legend_y + 30), "0", font=font(15), fill=MUTED)
        d.text((legend_x + steps * step_w - 55, legend_y + 30), f"{panel_max:,}", font=font(15), fill=MUTED)
    active_ids = set(metrics)
    mapped_ids = {item["id"] for item in polys}
    omitted_ids = sorted(active_ids - mapped_ids, key=lambda x: int(x) if x.isdigit() else x)
    d.text((70, 1390), f"Poligon valid ditampilkan: {len(mapped_ids)} dari {len(active_ids)} zona aktif. Tidak terpetakan: {', '.join(omitted_ids) if omitted_ids else 'tidak ada'}.", font=font(16), fill=MUTED)
    im.save(output, dpi=(300, 300), optimize=True)
    return {"mapped_zone_count": len(mapped_ids), "active_zone_count": len(active_ids), "omitted_zone_ids": omitted_ids}


def main() -> int:
    run_id = latest_valid_run()
    graph_dir = GRAPH_ROOT / run_id
    verify_run(run_id, graph_dir)
    output_base = graph_dir / "visualizations_chapter_vi"
    output = output_base
    version = 2
    while output.exists():
        output = graph_dir / f"visualizations_chapter_vi_v{version}"
        version += 1
    if output.exists():
        raise FileExistsError(f"Visualization folder already exists; not overwriting: {output}")
    output.mkdir(parents=True)

    graph_summary = read_csv(graph_dir / "graph_summary.csv")
    edges_path = graph_dir / "edges_by_month.csv"
    origins = read_csv(graph_dir / "top_origin_zones_by_month.csv")
    destinations = read_csv(graph_dir / "top_destination_zones_by_month.csv")
    nodes = read_csv(graph_dir / "zone_metrics_by_month.csv")
    lookup = {r["LocationID"]: r for r in read_csv(ZONE_LOOKUP)}
    issue_ids = {r["LocationID"] for r in read_csv(ZONE_ISSUES)}

    monthly_chart(graph_summary, output / "monthly_graph_structure.png")
    graph_flowchart(output / "graph_analytics_flowchart.png")
    top_flows_chart(edges_path, lookup, output / "top_od_flows_jan_jun.png")
    top_zones_chart(origins, destinations, output / "top_origin_destination_zones.png")
    map_info = flow_maps(nodes, issue_ids, output / "zone_flow_strength_maps.png")

    readme = [
        f"# Graph Analytics Visualizations - {run_id}",
        "",
        "Generated from validated graph aggregation CSVs; no trip Parquet was read.",
        "",
        "- `monthly_graph_structure.png`: monthly unique OD pairs and directed density.",
        "- `graph_analytics_flowchart.png`: Spark aggregation, graph construction, metrics, and outputs.",
        "- `top_od_flows_jan_jun.png`: top 15 directed OD pairs by six-month trip count.",
        "- `top_origin_destination_zones.png`: top 10 zones by outgoing and incoming weighted strength.",
        f"- `zone_flow_strength_maps.png`: outgoing/incoming trip volume on {map_info['mapped_zone_count']} valid zone polygons; each map panel uses its own logarithmic color range.",
        f"- Active graph zones: {map_info['active_zone_count']}; active IDs omitted from map due to missing/invalid geometry: {', '.join(map_info['omitted_zone_ids']) if map_info['omitted_zone_ids'] else 'none'}.",
        "- Map projection is a local equirectangular display of GeoJSON WGS84 coordinates; use the Taxi Zone polygons as reference boundaries, not street-level vehicle paths.",
        "",
    ]
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Visualizations created: {output}")
    print(f"Valid polygons mapped: {map_info['mapped_zone_count']} / {map_info['active_zone_count']} active graph zones")
    print("Files: graph_analytics_flowchart.png, monthly_graph_structure.png, top_od_flows_jan_jun.png, top_origin_destination_zones.png, zone_flow_strength_maps.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
