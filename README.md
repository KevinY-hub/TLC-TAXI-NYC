# NYC TLC HVFHV Trips and Weather Analysis (Jan–Jun 2026)

This repository contains the Python/PySpark pipeline and local Streamlit dashboard for an exploratory analysis of NYC TLC High Volume For-Hire Vehicle trips and Open-Meteo historical weather.

## Data and repository policy

Large/private/local data is intentionally excluded from Git. Place the six source Parquet files under `data/` (or at the project root only if a script supports that fallback), and follow the directory layout documented in the scripts. Generated Parquet outputs and reports belong under `output/`. The `.gitignore` excludes `data/`, `output/`, virtual environments, caches, and Parquet files. Download/reference sources and their manifests are not bundled.

Do not commit raw or integrated trip data, weather API response files, `.venv`, WSL's `ext4.vhdx`, credentials, or personal configuration files.

## Pipeline scripts

Run commands from the project root. Stages 1–4 inspect and profile the raw files; stages 5–8 create cleaned and integrated outputs; stage 9 creates validated exploratory aggregates. Existing outputs are not meant to be regenerated automatically—read each script's overwrite guard and check the target directory before rerunning it.

| Stage | Script | Purpose |
|---|---|---|
| 1 | `src/01_inspect_hvfhs.py` | Inspect Parquet metadata and schemas with PyArrow |
| 2 | `src/02_profile_hvfhs_sample.py` | Profile a bounded sample with Python/PyArrow/Pandas |
| 3 | `src/03_quality_check_spark.py` | Distributed quality counts on all raw months |
| 4 | `src/04_anomaly_investigation_spark.py` | Investigate suspicious records without cleaning |
| 5 | `src/05_clean_hvfhs_spark.py` | Apply the agreed timestamp and exact-duplicate rules |
| 6 | `src/06_integrate_taxi_zones_spark.py` | Add Taxi Zone names and representative coordinates |
| 7 | `src/07_collect_weather_openmeteo.py` | Collect hourly Open-Meteo historical weather |
| 8 | `src/08_integrate_weather_spark.py` | Left join weather by pickup zone and UTC hour |
| 9 | `src/09_exploratory_weather_analysis_spark.py` | Produce temporal, weather, and spatial aggregates |

The completed exploratory analysis recorded 126,771,977 trips. Weather was matched to 126,309,541 trips; 462,436 trips retained null weather values. Its report identifies 85 DST-gap pickups and trips in zones without representative geometry among the unmatched records. Weather comparisons are descriptive and do not establish causality.

## Environment

The tested runtime was WSL Ubuntu, Python 3.14, PySpark 4.2.0, and OpenJDK 21. The Spark scripts use local execution (`local[2]`); Hadoop/HDFS is not configured as part of this pipeline.

Install Java 21 separately in WSL, then create and activate the project environment:

```bash
python3 -m venv ~/hvfhs-venv
source ~/hvfhs-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm Java and PySpark:

```bash
java -version
python -c "import pyspark; print(pyspark.__version__)"
```

## Run the dashboard

From the project root in WSL:

```bash
source ~/hvfhs-venv/bin/activate
streamlit run app/dashboard.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open `http://localhost:8501` in the Windows browser. The dashboard reads only the Stage 3 aggregate tables in `output/analysis/weather_exploratory/`; it does not load the 126-million-row trip dataset.

## Main outputs

- `output/cleaned/` — cleaned trip Parquet
- `output/integrated/taxi_zones/` — trips with Taxi Zone attributes
- `output/integrated/taxi_zones_weather/` — trips with hourly weather attributes
- `output/analysis/weather_exploratory/` — compact analysis tables used by the dashboard
- `output/quality/` — quality and validation reports

These data products stay local or in a separate data backup. They are excluded from this source repository by design.
