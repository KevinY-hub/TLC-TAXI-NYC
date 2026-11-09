"""Create paper-ready SVG and interactive Plotly figures from saved ML results."""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = ROOT / "output" / "machine_learning"
CONTROLLED_ID = "20261106T204756Z"
CONTROLLED = ML_ROOT / "controlled_comparison" / CONTROLLED_ID
BASELINE_SPARK = ML_ROOT / "spark_rf" / "20261106T194743Z"
BASELINE_SKLEARN = ML_ROOT / "sklearn_rf" / "20261106T201549Z"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ML_ROOT / "visualizations" / "chapter_v" / RUN_ID

COLORS = {
    "spark": "#176B87",
    "sklearn": "#D97732",
    "training": "#6C8EBF",
    "validation": "#4D9B76",
    "testing": "#C45B60",
}
ZONE_KEY = "PULocationID"
HOUR_KEY = "pickup_hour_utc_epoch"


def require_inputs() -> None:
    required = [
        CONTROLLED / "predictions" / "spark" / "testing",
        CONTROLLED / "predictions" / "sklearn" / "testing.parquet",
        CONTROLLED / "reports" / "controlled_metrics.csv",
        CONTROLLED / "reports" / "feature_importance_by_source.csv",
        CONTROLLED / "reports" / "final_comparison_report.md",
        BASELINE_SPARK / "reports" / "model_metrics.csv",
        BASELINE_SPARK / "reports" / "feature_importance.csv",
        BASELINE_SKLEARN / "reports" / "model_metrics.csv",
        BASELINE_SKLEARN / "reports" / "feature_importance_by_source.csv",
        ML_ROOT / "reports" / "model_comparison.csv",
        ML_ROOT / "dataset",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required audit inputs: " + ", ".join(missing))
    if OUT.exists():
        raise FileExistsError(f"Visualization output exists; will not overwrite: {OUT}")
    if available_memory_gib() < 2.0:
        raise MemoryError("Less than 2 GiB RAM available; visualization generation stopped.")


def available_memory_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return 99.0


def load_predictions(path: Path) -> pd.DataFrame:
    return ds.dataset(str(path), format="parquet").to_table(
        columns=[ZONE_KEY, HOUR_KEY, "actual_trip_count", "prediction", "residual"]
    ).to_pandas()


def svg_header(title: str, desc: str, width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<title>{html.escape(title)}</title>',
        f'<desc>{html.escape(desc)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#243447}.title{font-size:23px;font-weight:700}.subtitle{font-size:13px;fill:#536579}.label{font-size:13px}.small{font-size:11px;fill:#536579}.axis{stroke:#738295;stroke-width:1}.grid{stroke:#e5eaf0;stroke-width:1}</style>',
        f'<text x="{width/2:.0f}" y="34" text-anchor="middle" class="title">{html.escape(title)}</text>',
        f'<text x="{width/2:.0f}" y="57" text-anchor="middle" class="subtitle">{html.escape(desc)}</text>',
    ]


def svg_finish(lines: list[str], path: Path) -> None:
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def nice_ticks(max_value: float, count: int = 5) -> list[float]:
    if max_value <= 0:
        return [0.0]
    return np.linspace(0, max_value, count + 1).tolist()


def color_for_density(value: int, maximum: int) -> str:
    if value <= 0:
        return "#f5f8fb"
    intensity = np.log1p(value) / np.log1p(maximum)
    low = np.array([225, 240, 246])
    high = np.array([18, 93, 125])
    rgb = (low + intensity * (high - low)).astype(int)
    return "#%02x%02x%02x" % tuple(rgb)


def svg_actual_predicted(spark: pd.DataFrame, sklearn: pd.DataFrame, path: Path) -> None:
    width, height = 1360, 680
    lines = svg_header(
        "Actual vs predicted hourly trip volume",
        "Controlled models, June 2026 testing set; cell color shows zone-hour frequency (log color scale)",
        width,
        height,
    )
    lines.insert(4, '<defs><linearGradient id="countScale" x1="0" x2="1"><stop offset="0%" stop-color="#e1f0f6"/><stop offset="100%" stop-color="#125d7d"/></linearGradient></defs>')
    max_value = float(max(spark.actual_trip_count.max(), spark.prediction.max(), sklearn.prediction.max()))
    max_value = max(1.0, np.ceil(max_value / 100) * 100)
    grid_n = 55
    bins = np.linspace(0, max_value, grid_n + 1)
    histograms = {
        "Spark MLlib": np.histogram2d(spark.actual_trip_count, spark.prediction, bins=(bins, bins))[0],
        "scikit-learn": np.histogram2d(sklearn.actual_trip_count, sklearn.prediction, bins=(bins, bins))[0],
    }
    shared_max_count = int(max(hist.max() for hist in histograms.values()))
    panel_y, panel_h = 115, 450
    panel_w, panel_xs = 535, [95, 730]
    for model, color, x0 in [
        ("Spark MLlib", COLORS["spark"], panel_xs[0]),
        ("scikit-learn", COLORS["sklearn"], panel_xs[1]),
    ]:
        x, y = panel_y, panel_w
        plot_x, plot_y, plot_w, plot_h = x0 + 55, panel_y + 30, panel_w - 80, panel_h - 80
        hist = histograms[model]
        max_count = shared_max_count
        cw, ch = plot_w / grid_n, plot_h / grid_n
        for ix in range(grid_n):
            for iy in range(grid_n):
                count = int(hist[ix, iy])
                if not count:
                    continue
                rx = plot_x + ix * cw
                ry = plot_y + plot_h - (iy + 1) * ch
                lines.append(f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{cw+0.3:.2f}" height="{ch+0.3:.2f}" fill="{color_for_density(count,max_count)}"/>')
        lines += [
            f'<line x1="{plot_x}" y1="{plot_y+plot_h}" x2="{plot_x+plot_w}" y2="{plot_y}" stroke="#1f2937" stroke-width="1.5" stroke-dasharray="7,5"/>',
            f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y+plot_h}" class="axis"/>',
            f'<line x1="{plot_x}" y1="{plot_y+plot_h}" x2="{plot_x+plot_w}" y2="{plot_y+plot_h}" class="axis"/>',
            f'<text x="{plot_x+plot_w/2:.0f}" y="{plot_y+plot_h+48}" text-anchor="middle" class="label">Actual trip_count (trips / zone-hour)</text>',
            f'<text x="{plot_x-48}" y="{plot_y+plot_h/2:.0f}" text-anchor="middle" transform="rotate(-90 {plot_x-48} {plot_y+plot_h/2:.0f})" class="label">Predicted trip_count</text>',
            f'<text x="{plot_x+plot_w/2:.0f}" y="{plot_y-12}" text-anchor="middle" class="label" font-weight="bold">{model}</text>',
        ]
        for t in nice_ticks(max_value):
            px = plot_x + t / max_value * plot_w
            py = plot_y + plot_h - t / max_value * plot_h
            lines += [
                f'<line x1="{px:.2f}" y1="{plot_y}" x2="{px:.2f}" y2="{plot_y+plot_h}" class="grid"/>',
                f'<line x1="{plot_x}" y1="{py:.2f}" x2="{plot_x+plot_w}" y2="{py:.2f}" class="grid"/>',
                f'<text x="{px:.2f}" y="{plot_y+plot_h+19}" text-anchor="middle" class="small">{t:.0f}</text>',
                f'<text x="{plot_x-8}" y="{py+4:.2f}" text-anchor="end" class="small">{t:.0f}</text>',
            ]
    lines += [
        '<text x="470" y="606" text-anchor="end" class="small">Cell frequency</text>',
        '<rect x="480" y="593" width="260" height="14" fill="url(#countScale)"/>',
        '<text x="750" y="606" class="small">low to high (log scale)</text>',
        '<text x="680" y="635" text-anchor="middle" class="small">Dashed diagonal indicates perfect prediction. All 180,689 test observations are included in the binned display.</text>',
    ]
    svg_finish(lines, path)


def svg_residuals(spark: pd.DataFrame, sklearn: pd.DataFrame, path: Path) -> dict[str, dict[str, float]]:
    width, height = 1200, 650
    lines = svg_header(
        "Residual distribution",
        "Controlled models, June 2026 testing set; residual = prediction - actual trip_count",
        width,
        height,
    )
    lo = float(min(spark.residual.min(), sklearn.residual.min()))
    hi = float(max(spark.residual.max(), sklearn.residual.max()))
    pad = (hi - lo) * 0.02
    lo, hi = lo - pad, hi + pad
    n_bins = 100
    edges = np.linspace(lo, hi, n_bins + 1)
    x0, y0, w, h = 100, 115, 1010, 410
    series = [("Spark MLlib", spark.residual.to_numpy(), COLORS["spark"]), ("scikit-learn", sklearn.residual.to_numpy(), COLORS["sklearn"])]
    maxima = []
    counts = {}
    summary = {}
    for name, residual, color in series:
        hist, _ = np.histogram(residual, bins=edges, density=True)
        counts[name] = hist
        maxima.append(float(hist.max()))
        summary[name] = {
            "mean": float(np.mean(residual)),
            "median": float(np.median(residual)),
            "p05": float(np.quantile(residual, 0.05)),
            "p95": float(np.quantile(residual, 0.95)),
            "below_zero_share": float(np.mean(residual < 0)),
            "above_zero_share": float(np.mean(residual > 0)),
        }
    ymax = max(maxima) * 1.08
    centers = (edges[:-1] + edges[1:]) / 2
    for name, residual, color in series:
        pts = []
        for xc, yc in zip(centers, counts[name]):
            px = x0 + (xc - lo) / (hi - lo) * w
            py = y0 + h - yc / ymax * h
            pts.append(f"{px:.2f},{py:.2f}")
        lines.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
    zero_x = x0 + (0 - lo) / (hi - lo) * w
    lines += [
        f'<line x1="{zero_x:.2f}" y1="{y0}" x2="{zero_x:.2f}" y2="{y0+h}" stroke="#111827" stroke-width="1.4" stroke-dasharray="6,4"/>',
        f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="axis"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" class="axis"/>',
        f'<text x="{x0+w/2}" y="{y0+h+50}" text-anchor="middle" class="label">Residual (predicted - actual trips)</text>',
        f'<text x="35" y="{y0+h/2}" text-anchor="middle" transform="rotate(-90 35 {y0+h/2})" class="label">Probability density</text>',
        f'<text x="{zero_x+5:.1f}" y="{y0+17}" class="small">zero error</text>',
    ]
    for t in np.linspace(lo, hi, 7):
        px = x0 + (t - lo) / (hi - lo) * w
        lines.append(f'<line x1="{px:.2f}" y1="{y0+h}" x2="{px:.2f}" y2="{y0+h+5}" class="axis"/><text x="{px:.2f}" y="{y0+h+22}" text-anchor="middle" class="small">{t:.0f}</text>')
    for i, (name, _, color) in enumerate(series):
        lx = 440 + i * 190
        lines += [f'<line x1="{lx}" y1="590" x2="{lx+30}" y2="590" stroke="{color}" stroke-width="3"/>', f'<text x="{lx+38}" y="594" class="label">{name}</text>']
    lines.append('<text x="600" y="625" text-anchor="middle" class="small">Negative residual = underprediction; positive residual = overprediction.</text>')
    svg_finish(lines, path)
    return summary


def svg_performance(metrics: pd.DataFrame, path: Path) -> None:
    width, height = 1500, 670
    lines = svg_header("Random Forest performance comparison", "Validation and testing metrics; controlled runs are the primary comparison", width, height)
    models = ["Baseline Spark MLlib", "Baseline scikit-learn", "Controlled Spark MLlib", "Controlled scikit-learn"]
    splits = ["validation", "testing"]
    metrics_to_plot = [("MAE", "MAE (trips / zone-hour)"), ("RMSE", "RMSE (trips / zone-hour)"), ("R2", "R-squared")]
    panel_w, panel_h = 440, 430
    xstarts = [75, 550, 1025]
    colors = {"validation": "#8BB7D9", "testing": "#285F8F"}
    for metric_index, (metric, label) in enumerate(metrics_to_plot):
        x0, y0 = xstarts[metric_index], 125
        w, h = panel_w, panel_h
        rows = metrics[metrics["split"].isin(splits)]
        max_v = float(rows[metric].max()) * 1.12
        group_w = w / len(models)
        bar_w = group_w * 0.27
        lines += [
            f'<text x="{x0+w/2}" y="{y0-24}" text-anchor="middle" class="label" font-weight="bold">{label}</text>',
            f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="axis"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" class="axis"/>',
        ]
        for tick in nice_ticks(max_v, 4):
            y = y0 + h - tick / max_v * h
            lines += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" class="grid"/>', f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="small">{tick:.2g}</text>']
        for mi, model in enumerate(models):
            center = x0 + group_w * (mi + 0.5)
            for si, split in enumerate(splits):
                vals = rows[(rows["implementation"] == model) & (rows["split"] == split)][metric]
                if vals.empty:
                    continue
                value = float(vals.iloc[0])
                bx = center + (si - 0.5) * bar_w - bar_w / 2
                bh = value / max_v * h
                by = y0 + h - bh
                lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w-3:.1f}" height="{bh:.1f}" fill="{colors[split]}"/>')
                lines.append(f'<text x="{bx+bar_w/2-1:.1f}" y="{by-5:.1f}" text-anchor="middle" class="small">{value:.3g}</text>')
            short = ["Base\nSpark", "Base\nsklearn", "Controlled\nSpark", "Controlled\nsklearn"][mi]
            for line_idx, part in enumerate(short.split("\\n")):
                lines.append(f'<text x="{center:.1f}" y="{y0+h+20+line_idx*13}" text-anchor="middle" class="small">{html.escape(part)}</text>')
    lines += [
        '<rect x="535" y="610" width="14" height="14" fill="#8BB7D9"/><text x="555" y="622" class="small">Validation</text>',
        '<rect x="660" y="610" width="14" height="14" fill="#285F8F"/><text x="680" y="622" class="small">Testing</text>',
        '<text x="750" y="650" text-anchor="middle" class="small">Baseline and controlled timing procedures differ; this chart compares prediction metrics only.</text>',
    ]
    svg_finish(lines, path)


def svg_importance(importance: pd.DataFrame, path: Path) -> None:
    width, height = 1200, 700
    lines = svg_header("Controlled Random Forest feature importance", "Importance is impurity-based and not causal evidence", width, height)
    features = sorted(importance.source_feature.unique().tolist())
    pivot = importance.pivot(index="source_feature", columns="implementation", values="importance").reindex(features)
    max_v = float(np.nanmax(pivot.to_numpy())) * 1.2
    x0, y0, w, row_h = 390, 110, 660, 49
    models = ["Controlled Spark MLlib", "Controlled scikit-learn"]
    fills = [COLORS["spark"], COLORS["sklearn"]]
    for i, feature in enumerate(features):
        y = y0 + i * row_h
        lines.append(f'<text x="{x0-15}" y="{y+25}" text-anchor="end" class="label">{html.escape(feature)}</text>')
        for j, model in enumerate(models):
            value = float(pivot.loc[feature, model])
            bx, by = x0, y + j * 18
            bw = value / max_v * w
            lines += [f'<rect x="{bx}" y="{by}" width="{bw:.2f}" height="13" fill="{fills[j]}"/>', f'<text x="{bx+bw+6:.2f}" y="{by+11}" class="small">{value:.4f}</text>']
    lines += [
        f'<line x1="{x0}" y1="{y0-8}" x2="{x0}" y2="{y0+row_h*len(features)}" class="axis"/>',
        f'<text x="{x0+w/2}" y="{height-37}" text-anchor="middle" class="label">Summed impurity importance</text>',
        f'<rect x="390" y="{height-25}" width="14" height="14" fill="{fills[0]}"/><text x="410" y="{height-13}" class="small">Spark MLlib</text>',
        f'<rect x="545" y="{height-25}" width="14" height="14" fill="{fills[1]}"/><text x="565" y="{height-13}" class="small">scikit-learn</text>',
    ]
    svg_finish(lines, path)


def svg_trip_distribution(dataset_path: Path, path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    data = ds.dataset(str(dataset_path), format="parquet", partitioning="hive")
    split_months = {
        "Training Jan-Apr": ["2026-01", "2026-02", "2026-03", "2026-04"],
        "Validation May": ["2026-05"],
        "Testing June": ["2026-06"],
    }
    arrays = {}
    rows = []
    all_values = []
    for split, months in split_months.items():
        arrow = data.to_table(columns=["trip_count"], filter=ds.field("pickup_month_local").isin(months))
        values = arrow.column("trip_count").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        if len(values) == 0 or int(values.min()) <= 0:
            raise ValueError(f"Expected positive observed trip_count rows for {split}")
        arrays[split] = values
        all_values.append(values)
        rows.append({"split": split, "zone_hour_rows": int(len(values)), "trip_count_sum": int(values.sum()), "min": int(values.min()), "median": float(np.median(values)), "mean": float(np.mean(values)), "max": int(values.max()), "zero_rows": int((values == 0).sum())})
    combined = np.concatenate(all_values)
    low = max(1, int(combined.min()))
    high = int(combined.max())
    edges = np.geomspace(low, high + 1, 45)
    width, height = 1200, 650
    lines = svg_header("Observed hourly trip volume distribution", "Actual trip_count by zone-hour; only observed positive-demand combinations are present", width, height)
    x0, y0, w, h = 100, 120, 1010, 410
    colors = {"Training Jan-Apr": COLORS["training"], "Validation May": COLORS["validation"], "Testing June": COLORS["testing"]}
    densities = {}
    max_density = 0.0
    for name, values in arrays.items():
        hist, _ = np.histogram(values, bins=edges, density=True)
        densities[name] = hist
        max_density = max(max_density, float(hist.max()))
    max_density *= 1.12
    for idx, (name, hist) in enumerate(densities.items()):
        pts = []
        centers = np.sqrt(edges[:-1] * edges[1:])
        for xc, yc in zip(centers, hist):
            px = x0 + (np.log(xc) - np.log(low)) / (np.log(high + 1) - np.log(low)) * w
            py = y0 + h - yc / max_density * h
            pts.append(f"{px:.2f},{py:.2f}")
        lines.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[name]}" stroke-width="2.4"/>')
    lines += [
        f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="axis"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" class="axis"/>',
        f'<text x="{x0+w/2}" y="{y0+h+50}" text-anchor="middle" class="label">trip_count (trips per observed pickup-zone hour; log scale)</text>',
        f'<text x="35" y="{y0+h/2}" text-anchor="middle" transform="rotate(-90 35 {y0+h/2})" class="label">Probability density</text>',
    ]
    for val in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, high]:
        if val < low or val > high:
            continue
        px = x0 + (np.log(val) - np.log(low)) / (np.log(high + 1) - np.log(low)) * w
        lines += [f'<line x1="{px:.2f}" y1="{y0+h}" x2="{px:.2f}" y2="{y0+h+5}" class="axis"/>', f'<text x="{px:.2f}" y="{y0+h+22}" text-anchor="middle" class="small">{val}</text>']
    for idx, (name, color) in enumerate(colors.items()):
        lx = 310 + idx * 230
        lines += [f'<line x1="{lx}" y1="600" x2="{lx+28}" y2="600" stroke="{color}" stroke-width="3"/>', f'<text x="{lx+36}" y="604" class="small">{name}</text>']
    lines.append('<text x="600" y="630" text-anchor="middle" class="small">No synthetic zero-demand zone-hours were added; this is a distribution of observed positive counts.</text>')
    svg_finish(lines, path)
    return pd.DataFrame(rows), arrays


def make_plotly_html(
    out: Path,
    spark: pd.DataFrame,
    sklearn: pd.DataFrame,
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
    volume: dict[str, np.ndarray],
) -> None:
    sample_n = min(25000, len(spark))
    idx = np.random.default_rng(42).choice(len(spark), sample_n, replace=False)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Spark MLlib", "scikit-learn"))
    for col, data, color, name in [(1, spark, COLORS["spark"], "Spark"), (2, sklearn, COLORS["sklearn"], "scikit-learn")]:
        fig.add_trace(go.Scattergl(x=data.actual_trip_count.iloc[idx], y=data.prediction.iloc[idx], mode="markers", marker={"size":4,"opacity":0.4,"color":color}, name=name), row=1, col=col)
        hi = float(max(data.actual_trip_count.max(), data.prediction.max()))
        fig.add_trace(go.Scatter(x=[0,hi],y=[0,hi],mode="lines",line={"color":"#333","dash":"dash"},name="Perfect prediction",showlegend=col==1),row=1,col=col)
        fig.update_xaxes(title_text="Actual trip_count", range=[0,hi], row=1,col=col)
        fig.update_yaxes(title_text="Predicted trip_count", range=[0,hi], row=1,col=col)
    fig.update_layout(title="Actual vs predicted trip_count — controlled testing set", template="plotly_white")
    fig.write_html(str(out / "actual_vs_predicted.html"), include_plotlyjs="cdn")

    res_fig = go.Figure()
    common_min = min(spark.residual.min(), sklearn.residual.min())
    common_max = max(spark.residual.max(), sklearn.residual.max())
    for data, color, name in [(spark, COLORS["spark"], "Spark MLlib"), (sklearn, COLORS["sklearn"], "scikit-learn")]:
        res_fig.add_trace(go.Histogram(x=data.residual, nbinsx=100, histnorm="probability density", opacity=0.56, marker_color=color, name=name))
    res_fig.add_vline(x=0,line_dash="dash",line_color="#222")
    res_fig.update_layout(title="Residual distribution — controlled testing set", xaxis_title="Residual (prediction - actual trip_count)", yaxis_title="Probability density", barmode="overlay", template="plotly_white")
    res_fig.write_html(str(out / "residual_distribution.html"), include_plotlyjs="cdn")

    perf = make_subplots(rows=1,cols=3,subplot_titles=("MAE","RMSE","R2"))
    models=["Baseline Spark MLlib","Baseline scikit-learn","Controlled Spark MLlib","Controlled scikit-learn"]
    for metric, col in [("MAE",1),("RMSE",2),("R2",3)]:
        for split,color in [("validation","#8BB7D9"),("testing","#285F8F")]:
            sub=metrics[metrics.split==split].set_index("implementation").reindex(models)
            perf.add_trace(go.Bar(x=models,y=sub[metric],name=split,marker_color=color,legendgroup=split,showlegend=col==1),row=1,col=col)
        perf.update_yaxes(title_text=metric,row=1,col=col)
    perf.update_layout(title="Model performance — validation and testing",barmode="group",template="plotly_white")
    perf.write_html(str(out / "model_performance_comparison.html"),include_plotlyjs="cdn")

    imp = importance[importance.implementation.isin(["Controlled Spark MLlib","Controlled scikit-learn"])]
    fig_imp=go.Figure()
    for model,color in [("Controlled Spark MLlib",COLORS["spark"]),("Controlled scikit-learn",COLORS["sklearn"])]:
        sub=imp[imp.implementation==model].sort_values("importance")
        fig_imp.add_trace(go.Bar(x=sub.importance,y=sub.source_feature,orientation="h",name=model,marker_color=color))
    fig_imp.update_layout(title="Controlled RF feature importance by source feature",xaxis_title="Summed impurity importance",yaxis_title="Source feature",barmode="group",template="plotly_white")
    fig_imp.write_html(str(out / "controlled_feature_importance.html"),include_plotlyjs="cdn")

    volfig=go.Figure()
    bins=np.geomspace(1,max(max(v.max() for v in volume.values()),2)+1,45)
    centers=np.sqrt(bins[:-1]*bins[1:])
    for name,vals in volume.items():
        density,_=np.histogram(vals,bins=bins,density=True)
        volfig.add_trace(go.Scatter(x=centers,y=density,mode="lines",name=name))
    volfig.update_layout(title="Observed trip_count distribution by chronological split",xaxis_title="trip_count per observed pickup-zone hour",yaxis_title="Probability density",xaxis_type="log",template="plotly_white")
    volfig.write_html(str(out / "trip_volume_distribution.html"),include_plotlyjs="cdn")


def main() -> None:
    require_inputs()
    OUT.mkdir(parents=True, exist_ok=False)
    reports = CONTROLLED / "reports"
    spark = load_predictions(CONTROLLED / "predictions" / "spark" / "testing")
    sklearn = load_predictions(CONTROLLED / "predictions" / "sklearn" / "testing.parquet")
    spark[ZONE_KEY] = spark[ZONE_KEY].astype(str)
    sklearn[ZONE_KEY] = sklearn[ZONE_KEY].astype(str)
    keys = [ZONE_KEY, HOUR_KEY]
    spark = spark.sort_values(keys).reset_index(drop=True)
    sklearn = sklearn.sort_values(keys).reset_index(drop=True)
    if len(spark) != 180_689 or len(sklearn) != 180_689:
        raise ValueError(f"Unexpected controlled test size: Spark={len(spark)}, sklearn={len(sklearn)}")
    if not np.array_equal(spark[keys].to_numpy(), sklearn[keys].to_numpy()):
        raise ValueError("Controlled model test prediction keys differ")
    if not np.array_equal(spark.actual_trip_count.to_numpy(), sklearn.actual_trip_count.to_numpy()):
        raise ValueError("Controlled model test actual target arrays differ")
    for frame in (spark, sklearn):
        if not np.allclose(frame.residual, frame.prediction - frame.actual_trip_count, rtol=0, atol=1e-8):
            raise ValueError("Residual column does not equal prediction - actual")

    baseline_spark = pd.read_csv(BASELINE_SPARK / "reports" / "model_metrics.csv")
    baseline_sklearn = pd.read_csv(BASELINE_SKLEARN / "reports" / "model_metrics.csv")
    consolidated_baseline = pd.read_csv(ML_ROOT / "reports" / "model_comparison.csv")
    baseline_individual = pd.concat(
        [
            baseline_spark.assign(implementation="Spark MLlib RandomForestRegressor"),
            baseline_sklearn.assign(implementation="scikit-learn RandomForestRegressor"),
        ],
        ignore_index=True,
    ).sort_values(["implementation", "split"]).reset_index(drop=True)
    consolidated_baseline = consolidated_baseline.sort_values(["implementation", "split"]).reset_index(drop=True)
    audit_cols = ["implementation", "split", "rows", "trip_count_sum", "MAE", "RMSE", "R2"]
    if baseline_individual[audit_cols][["implementation", "split"]].to_numpy().tolist() != consolidated_baseline[audit_cols][["implementation", "split"]].to_numpy().tolist():
        raise ValueError("Root baseline comparison report rows do not align with individual baseline metric CSVs")
    if not np.allclose(
        baseline_individual[["rows", "trip_count_sum", "MAE", "RMSE", "R2"]].to_numpy(dtype=float),
        consolidated_baseline[["rows", "trip_count_sum", "MAE", "RMSE", "R2"]].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError("Root baseline metrics differ from individual run metric files")
    controlled = pd.read_csv(reports / "controlled_metrics.csv")
    baseline_spark["implementation"] = "Baseline Spark MLlib"
    baseline_sklearn["implementation"] = "Baseline scikit-learn"
    metrics = pd.concat([baseline_spark, baseline_sklearn, controlled], ignore_index=True)
    importance = pd.read_csv(reports / "feature_importance_by_source.csv")
    if set(importance.implementation) != {"Baseline Spark MLlib","Baseline scikit-learn","Controlled Spark MLlib","Controlled scikit-learn"}:
        raise ValueError("Expected baseline and controlled feature importance rows")

    static_files = {
        "actual_vs_predicted.svg": svg_actual_predicted,
        "residual_distribution.svg": svg_residuals,
        "model_performance_comparison.svg": svg_performance,
        "controlled_feature_importance.svg": svg_importance,
    }
    svg_actual_predicted(spark, sklearn, OUT / "actual_vs_predicted.svg")
    residual_summary = svg_residuals(spark, sklearn, OUT / "residual_distribution.svg")
    svg_performance(metrics, OUT / "model_performance_comparison.svg")
    controlled_importance = importance[importance.implementation.isin(["Controlled Spark MLlib","Controlled scikit-learn"])].copy()
    svg_importance(controlled_importance, OUT / "controlled_feature_importance.svg")
    volume_summary, volume_arrays = svg_trip_distribution(ML_ROOT / "dataset", OUT / "trip_volume_distribution.svg")
    volume_summary.to_csv(OUT / "trip_volume_distribution_summary.csv", index=False)
    pd.DataFrame(
        [{"model": model, **values} for model, values in residual_summary.items()]
    ).to_csv(OUT / "residual_summary.csv", index=False)
    metrics.to_csv(OUT / "metrics_used.csv", index=False)
    importance.to_csv(OUT / "feature_importance_used.csv", index=False)
    make_plotly_html(OUT, spark, sklearn, metrics, importance, volume_arrays)

    description = [
        "# Chapter V machine learning visualizations",
        "",
        f"Generated from saved baseline runs and controlled run `{CONTROLLED_ID}`.",
        "",
        "## Files",
        "",
        "- `actual_vs_predicted.svg` and `.html`: controlled Spark/sklearn test predictions; SVG uses 2D bins over all test rows, HTML uses a deterministic sample for browser interaction.",
        "- `residual_distribution.svg` and `.html`: residual is prediction minus actual; negative values indicate underprediction, positive values overprediction.",
        "- `model_performance_comparison.svg` and `.html`: MAE, RMSE, and R2 for both validation and testing; baseline rows are shown separately from controlled rows.",
        "- `controlled_feature_importance.svg` and `.html`: controlled models, importance aggregated by source feature; impurity importance is not causal evidence.",
        "- `trip_volume_distribution.svg` and `.html`: actual trip_count per observed pickup-zone hour, grouped by chronological split and displayed with a logarithmic x-axis.",
        "- `metrics_used.csv`, `feature_importance_used.csv`, `residual_summary.csv`, `trip_volume_distribution_summary.csv`: source values and descriptive summaries used for the figures.",
        "",
        "The trip volume distribution contains observed positive-demand zone-hour combinations only. The ML dataset did not synthesize zero-demand combinations; no zeros were added for these plots.",
        "",
        "SVG files are vector graphics suitable for document insertion. Plotly HTML files are interactive. Static raster export was not attempted because matplotlib and Kaleido are not installed in the project environment.",
    ]
    (OUT / "README.md").write_text("\n".join(description) + "\n", encoding="utf-8")

    print(f"Visualizations: {OUT}")
    print(f"Controlled testing rows per model: {len(spark):,}; keys and actual targets matched exactly")
    print(f"Actual trip_count distribution rows: {int(volume_summary.zone_hour_rows.sum()):,}; zero rows added: {int(volume_summary.zero_rows.sum())}")
    print("Static SVG and interactive Plotly HTML outputs created; no dependencies installed.")


if __name__ == "__main__":
    main()
