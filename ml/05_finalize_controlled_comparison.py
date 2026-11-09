"""Build derived controlled-comparison reports without retraining or changing models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "output" / "machine_learning"


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def importance_source(name: str) -> str:
    if name.startswith("PULocationID=") or name == "PULocationID_index":
        return "PULocationID"
    if name.startswith("weather_code=") or name == "weather_code_index":
        return "weather_code"
    return name


def prediction_consistency(run_root: Path) -> dict[str, dict[str, int]]:
    results = {}
    for split in ("training", "validation", "testing"):
        spark_path = run_root / "predictions" / "spark" / split
        sklearn_path = run_root / "predictions" / "sklearn" / f"{split}.parquet"
        columns = ["PULocationID", "pickup_hour_utc_epoch", "actual_trip_count"]
        spark = ds.dataset(str(spark_path), format="parquet").to_table(columns=columns).to_pandas()
        sklearn = ds.dataset(str(sklearn_path), format="parquet").to_table(columns=columns).to_pandas()
        for frame in (spark, sklearn):
            frame["PULocationID"] = frame["PULocationID"].astype(str)
        keys = ["PULocationID", "pickup_hour_utc_epoch"]
        spark = spark.sort_values(keys).reset_index(drop=True)
        sklearn = sklearn.sort_values(keys).reset_index(drop=True)
        key_mismatches = int(
            (spark[keys].to_numpy() != sklearn[keys].to_numpy()).any(axis=1).sum()
        ) if len(spark) == len(sklearn) else abs(len(spark) - len(sklearn))
        target_mismatches = (
            int((spark["actual_trip_count"].to_numpy() != sklearn["actual_trip_count"].to_numpy()).sum())
            if len(spark) == len(sklearn) and key_mismatches == 0
            else -1
        )
        results[split] = {
            "spark_rows": len(spark),
            "sklearn_rows": len(sklearn),
            "key_mismatches": key_mismatches,
            "target_mismatches": target_mismatches,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overwrite-derived", action="store_true")
    args = parser.parse_args()
    run_root = BASELINE_ROOT / "controlled_comparison" / args.run_id
    reports = run_root / "reports"
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    report_path = reports / "final_comparison_report.md"
    grouped_path = reports / "feature_importance_by_source.csv"
    chart_path = reports / "controlled_feature_importance_by_source.html"
    if (report_path.exists() or grouped_path.exists() or chart_path.exists()) and not args.overwrite_derived:
        raise FileExistsError("Derived final report output already exists; no files were overwritten.")

    config = pd.read_json(reports / "experiment_configuration.json", typ="series").to_dict()
    encoding = config["encoding_verification"]
    if encoding["status"] != "IDENTICAL_SAMPLE_ENCODING" or encoding["max_absolute_difference"] != 0.0:
        raise ValueError(f"Encoding equivalence not verified: {encoding}")

    baseline_spark = read_csv(BASELINE_ROOT / "spark_rf" / "20261106T194743Z" / "reports" / "model_metrics.csv")
    baseline_sklearn = read_csv(BASELINE_ROOT / "sklearn_rf" / "20261106T201549Z" / "reports" / "model_metrics.csv")
    controlled = read_csv(reports / "controlled_metrics.csv")
    model_metric_rows = []
    for family, rows in (("Baseline Spark MLlib", baseline_spark), ("Baseline scikit-learn", baseline_sklearn), ("Controlled Spark MLlib", [r for r in controlled if r["implementation"] == "Controlled Spark MLlib"]), ("Controlled scikit-learn", [r for r in controlled if r["implementation"] == "Controlled scikit-learn"])):
        for row in rows:
            pred_seconds = row.get("prediction_seconds", row.get("prediction_materialization_seconds"))
            model_metric_rows.append({"implementation": family, "split": row["split"], **{k: row[k] for k in ("rows", "trip_count_sum", "MAE", "RMSE", "R2")}, "fit_seconds": row.get("fit_seconds"), "prediction_seconds": pred_seconds})
    metric_df = pd.DataFrame(model_metric_rows)

    importance_rows = []
    importance_sources = [
        ("Baseline Spark MLlib", BASELINE_ROOT / "spark_rf" / "20261106T194743Z" / "reports" / "feature_importance.csv"),
        ("Baseline scikit-learn", BASELINE_ROOT / "sklearn_rf" / "20261106T201549Z" / "reports" / "feature_importance_by_source.csv"),
        ("Controlled Spark MLlib", reports / "spark_feature_importance.csv"),
        ("Controlled scikit-learn", reports / "sklearn_feature_importance.csv"),
    ]
    for model_name, path in importance_sources:
        for row in read_csv(path):
            feature = row["feature"]
            importance_rows.append(
                {
                    "implementation": model_name,
                    "feature": feature,
                    "source_feature": importance_source(feature),
                    "importance": float(row["importance"]),
                }
            )
    importance_df = pd.DataFrame(importance_rows)
    grouped_df = (
        importance_df.groupby(["implementation", "source_feature"], as_index=False)["importance"]
        .sum()
        .sort_values(["implementation", "importance"], ascending=[True, False])
    )
    grouped_df.to_csv(grouped_path, index=False)

    consistency = prediction_consistency(run_root)
    if any(v["key_mismatches"] != 0 or v["target_mismatches"] != 0 for v in consistency.values()):
        raise ValueError(f"Prediction consistency validation failed: {consistency}")

    def metric(family: str, split: str, column: str) -> float:
        row = metric_df[(metric_df["implementation"] == family) & (metric_df["split"] == split)]
        if len(row) != 1:
            raise ValueError(f"Metric row missing/duplicated: {family} {split}")
        return float(row.iloc[0][column])

    timing = config["training_seconds"]
    controlled_spark_fit = float(timing["spark_including_preprocessing"])
    controlled_sklearn_fit = float(timing["sklearn_including_encoder_and_rf"])
    for row in model_metric_rows:
        if row["implementation"] == "Controlled Spark MLlib":
            row["fit_seconds"] = controlled_spark_fit
        elif row["implementation"] == "Controlled scikit-learn":
            row["fit_seconds"] = controlled_sklearn_fit
    metric_df = pd.DataFrame(model_metric_rows)
    metric_table = [
        "| Model | Split | Rows | MAE | RMSE | R2 | Fit (s) | Predict (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ("Baseline Spark MLlib", "Baseline scikit-learn", "Controlled Spark MLlib", "Controlled scikit-learn"):
        rows = metric_df[metric_df["implementation"] == family]
        for row in rows.to_dict(orient="records"):
            metric_table.append(
                f"| {family} | {row['split']} | {int(row['rows']):,} | {float(row['MAE']):.6f} | {float(row['RMSE']):.6f} | {float(row['R2']):.6f} | {float(row['fit_seconds']):.3f} | {float(row['prediction_seconds']):.3f} |"
            )

    lines = [
        "# Controlled Random Forest: Laporan Perbandingan Akhir",
        "",
        f"Run ID: `{args.run_id}`",
        "",
        "## Data dan encoding",
        "",
        "Kedua controlled model memakai label `trip_count`, fitur yang sama, dan split Jan-Apr training, Mei validation, Juni testing. Encoder kategori hanya menggunakan kategori dari training. `PULocationID` memiliki 258 kategori training dan `weather_code` 13. Kategori tak dikenal dipetakan ke token khusus yang menjadi satu kolom one-hot tambahan per variabel.",
        "",
        f"Spark dan scikit-learn menghasilkan matriks yang identik pada {encoding['checked_rows']} baris pemeriksaan (termasuk kasus kategori tak dikenal): {encoding['feature_dimension']} kolom dan perbedaan absolut maksimum {encoding['max_absolute_difference']:.1f}. Urutan: one-hot zona, one-hot weather code, lalu delapan fitur numerik.",
        "",
        "Jumlah observasi: 718.545 training, 186.481 validation, dan 180.689 testing. Jumlah `trip_count` masing-masing 83.569.154, 22.045.080, dan 20.695.307.",
        "",
        "## Metrik keempat eksperimen",
        "",
        *metric_table,
        "",
        "Baseline hanya memiliki metrik validation/testing; training metrics tidak tersedia untuk baseline.",
        "Waktu baseline ditampilkan sebagai catatan dari run lama, tetapi prosedur pengukuran dan cakupan transformasi/materialisasi berbeda dari controlled run; jangan membandingkan durasi baseline dan controlled secara langsung.",
        "",
        "## Perubahan baseline ke controlled",
        "",
    ]
    for base_family, controlled_family in (("Baseline Spark MLlib", "Controlled Spark MLlib"), ("Baseline scikit-learn", "Controlled scikit-learn")):
        lines += [f"### {base_family} to {controlled_family}", "", "| Split | Delta MAE (controlled - baseline) | Delta RMSE | Delta R2 |", "|---|---:|---:|---:|"]
        for split in ("validation", "testing"):
            deltas = [metric(controlled_family, split, c) - metric(base_family, split, c) for c in ("MAE", "RMSE", "R2")]
            lines.append(f"| {split} | {deltas[0]:+.6f} | {deltas[1]:+.6f} | {deltas[2]:+.6f} |")
        lines.append("")
    lines += [
        "Controlled Spark hasilnya berbeda nyata dari baseline Spark setelah encoding, feature sampling, dan maxBins berubah. Baseline memakai maxBins=259, sedangkan controlled memakai 32 karena fitur kategorikal sudah di-one-hot; binning fitur numerik Spark juga dapat terpengaruh. Baseline dan controlled scikit-learn menghasilkan metrik yang sangat dekat. Selisih baseline-controlled tidak dapat diatribusikan pada satu perubahan tunggal karena beberapa konfigurasi berubah bersama.",
        "",
        "## Training versus testing",
        "",
    ]
    for family in ("Controlled Spark MLlib", "Controlled scikit-learn"):
        train = {m: metric(family, "training", m) for m in ("MAE", "RMSE", "R2")}
        test = {m: metric(family, "testing", m) for m in ("MAE", "RMSE", "R2")}
        lines.append(
            f"- {family}: training MAE/RMSE/R2 = {train['MAE']:.4f}/{train['RMSE']:.4f}/{train['R2']:.4f}; testing = {test['MAE']:.4f}/{test['RMSE']:.4f}/{test['R2']:.4f}."
        )
    lines += [
        "",
        "Pada kedua controlled model, metrik test sedikit lebih baik daripada metrik training menurut ketiga angka tersebut. Ini bukan pola train-score yang lebih baik daripada test-score; tetapi perubahan distribusi antarbulan mungkin menjelaskan sebagian perbedaan. Mei dan Juni relatif dekat. Hasil enam bulan dan satu konfigurasi belum cukup untuk menyimpulkan ketiadaan overfitting secara umum.",
        "",
        "## Feature importance",
        "",
        "Feature importance untuk keempat eksperimen diringkas per fitur asal di `feature_importance_by_source.csv`; dummy one-hot dijumlahkan berdasarkan fitur asal. Importance encoded controlled tersedia pada `spark_feature_importance.csv` dan `sklearn_feature_importance.csv`. Angka ini impurity-based, bukan ukuran kausal.",
        "",
    ]
    for model in ("Baseline Spark MLlib", "Baseline scikit-learn", "Controlled Spark MLlib", "Controlled scikit-learn"):
        top = grouped_df[grouped_df["implementation"] == model].sort_values("importance", ascending=False).head(5)
        lines.append(f"Top 5 {model}: " + ", ".join(f"`{r.source_feature}` {r.importance:.4f}" for r in top.itertuples()))
    lines += [
        "",
        "Feature importance controlled lebih sebanding karena input one-hot dan urutannya sama. Feature importance baseline tidak dapat dibandingkan langsung karena unit kolom dan encoding berbeda. Pada semua model, importance impurity bukan efek kausal; implementasi impurity dan prosedur split juga masih berbeda.",
        "",
        "## Waktu, resource, dan batasan perbandingan",
        "",
        f"Controlled Spark fit termasuk fit preprocessing Spark dan RF: {controlled_spark_fit:.3f} detik; scikit-learn termasuk fit OHE dan RF: {controlled_sklearn_fit:.3f} detik. Waktu prediksi meliputi transformasi dan evaluasi seluruh prediksi: Spark memaksa distributed action agregasi; scikit-learn mematerialisasi array prediksi. Penulisan file prediksi dan penghitungan metrik tidak masuk ke interval tersebut.",
        "",
        "RAM tersedia sebelum run sekitar 15,8 GB; safety floor 6 GiB. Perkiraan CSR training matrix sekitar 118 MB; dataset Parquet sekitar 8,7 MB. Peak RSS Python dicatat, tetapi tidak mencakup JVM Spark, sehingga peak memory kedua model tidak dibandingkan.",
        "",
        "Kesamaan yang dicapai meliputi representasi dan urutan fitur, `50` trees, depth `8`, seed `42`, seluruh fitur pada setiap split, bootstrap aktif, dan analog parameter minimum leaf/gain. Kesamaan eksak tetap tidak mungkin: seed tidak menyamakan random stream lintas library, Spark `maxBins` tidak memiliki parameter identik di scikit-learn, dan implementasi pohon/criterion dapat berbeda. Jangan menyimpulkan satu library secara umum lebih unggul dari satu run ini.",
        "",
        "## Validasi prediction outputs",
        "",
    ]
    for split, values in consistency.items():
        lines.append(
            f"- {split}: Spark {values['spark_rows']:,}, scikit-learn {values['sklearn_rows']:,}; key mismatch {values['key_mismatches']}; target mismatch {values['target_mismatches']}."
        )
    lines += [
        "",
        "Kolom `PULocationID` tersimpan sebagai string pada prediksi Spark karena kebutuhan categorical indexer dan sebagai integer pada prediksi scikit-learn; key dicast ke string hanya untuk audit kesetaraan. Nilai ID dan observasi tetap sama.",
        "",
        "Baseline artifacts dan reports dibaca saja. Controlled model dan seluruh hasil berada di run folder ini.",
        "",
        "Parameter references: [Spark 4.2.0 RandomForestRegressor](https://spark.apache.org/docs/4.2.0/api/python/reference/api/pyspark.ml.regression.RandomForestRegressor.html) and [scikit-learn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for model in ("Baseline Spark MLlib", "Baseline scikit-learn", "Controlled Spark MLlib", "Controlled scikit-learn"):
            part = grouped_df[grouped_df["implementation"] == model].sort_values("importance", ascending=False)
            fig.add_trace(go.Bar(name=model, x=part["source_feature"], y=part["importance"]))
        fig.update_layout(title="Controlled Random Forest feature importance by source feature", xaxis_title="Feature", yaxis_title="Summed impurity importance", barmode="group")
        fig.write_html(chart_path, include_plotlyjs="cdn")
    except Exception as exc:
        raise RuntimeError(f"Could not create feature importance chart: {exc}") from exc

    print(f"Final report: {report_path}")
    print(f"Grouped importance: {grouped_path}")
    print(f"Prediction consistency: {consistency}")


if __name__ == "__main__":
    main()
