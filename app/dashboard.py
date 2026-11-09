"""Local Streamlit dashboard for the validated HVFHV exploratory aggregates."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "output" / "analysis" / "weather_exploratory"
QUALITY_DIR = PROJECT_ROOT / "output" / "quality" / "weather_exploratory"
TAXI_ZONE_GEOJSON = PROJECT_ROOT / "data" / "reference" / "taxi_zones" / "nyc_taxi_zones.geojson"
GRAPH_ANALYSIS_DIR = PROJECT_ROOT / "output" / "analysis" / "graph_analytics" / "20261107T174911Z"

TABLES = {
    "daily": "temporal_trips_by_date",
    "hourly": "temporal_trips_by_hour",
    "weekday": "temporal_trips_by_weekday",
    "monthly": "temporal_trips_by_month",
    "date_hour": "temporal_trips_by_date_hour",
    "weather_monthly": "weather_trips_by_month_category",
    "locations": "pickup_location_volume",
    "boroughs": "pickup_borough_volume",
    "zone_hour_weather": "pickup_zone_hour_weather_volume",
}

CATEGORY_LABELS = {
    "no_precipitation": "Tanpa presipitasi",
    "rain": "Hujan",
    "snow": "Salju",
    "mixed_or_unclassified_precipitation": "Campuran / tidak terklasifikasi",
    "weather_unavailable": "Cuaca tidak tersedia",
}
CATEGORY_ORDER = list(CATEGORY_LABELS)
COLORS = {
    "Tanpa presipitasi": "#2f7d67",
    "Hujan": "#4386c6",
    "Salju": "#8c78c9",
    "Campuran / tidak terklasifikasi": "#e69f45",
    "Cuaca tidak tersedia": "#9aa4b2",
}
PLOT_TEMPLATE = "plotly_white"


st.set_page_config(
    page_title="NYC HVFHV | Mobility & Weather",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2.4rem; max-width: 1500px;}
      [data-testid="stMetric"] {background: #f5f8fa; border: 1px solid #e5ebef;
        padding: 16px 18px; border-radius: 12px; color: #172b3a !important;}
      [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
        color: #475569 !important;}
      [data-testid="stMetricValue"] {color: #172b3a !important; font-size: 1.65rem;}
      [data-testid="stMetricDelta"] {font-size: .82rem;}
      .hero {padding: 1rem 1.25rem; background: linear-gradient(115deg,#102c3a,#205b61);
        border-radius: 14px; color: white; margin-bottom: 1rem;}
      .hero p {margin-bottom: 0; color: #d4e3e5;}
      .note {padding: .8rem 1rem; background: #f3f6f8; border-left: 4px solid #4386c6;
        border-radius: 5px; color: #334155;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Membaca tabel agregasi...")
def load_parquet(table_key: str) -> pd.DataFrame:
    """Load only a named Stage 3 aggregate, never the trip-level dataset."""
    if table_key not in TABLES:
        raise ValueError(f"Unknown aggregate table key: {table_key}")
    path = ANALYSIS_DIR / f"{TABLES[table_key]}.parquet"
    if not path.is_dir():
        raise FileNotFoundError(f"Tabel agregasi tidak ditemukan: {path}")
    return pd.read_parquet(path, engine="pyarrow")


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    allowed = {
        "weather_category_summary.csv",
        "weather_category_definitions.csv",
        "analysis_summary.csv",
        "analysis_validation.csv",
        "top_pickup_zones.csv",
    }
    if name not in allowed:
        raise ValueError(f"Unknown report: {name}")
    path = QUALITY_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Laporan tidak ditemukan: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner="Membaca agregat Graph Analytics...")
def load_graph_csv(name: str) -> pd.DataFrame:
    """Read compact, precomputed graph aggregates only."""
    allowed = {"graph_summary.csv", "zone_metrics_by_month.csv", "edges_by_month.csv"}
    if name not in allowed:
        raise ValueError(f"Unknown graph aggregate: {name}")
    path = GRAPH_ANALYSIS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Agregat Graph Analytics tidak ditemukan: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner="Membaca geometri Taxi Zone...")
def load_taxi_zone_geojson() -> dict[str, Any]:
    """Load valid, uniquely keyed Taxi Zone polygons for the aggregate map."""
    if not TAXI_ZONE_GEOJSON.is_file():
        raise FileNotFoundError(f"GeoJSON Taxi Zone tidak ditemukan: {TAXI_ZONE_GEOJSON}")
    with TAXI_ZONE_GEOJSON.open(encoding="utf-8") as source:
        geojson = json.load(source)

    features = geojson.get("features", [])
    id_counts = Counter(
        str(feature.get("properties", {}).get("locationid", ""))
        for feature in features
    )
    valid_features = []
    for feature in features:
        location_id = str(feature.get("properties", {}).get("locationid", ""))
        geometry = feature.get("geometry") or {}
        if location_id and id_counts[location_id] == 1 and geometry.get("coordinates"):
            valid_features.append(feature)

    if not valid_features:
        raise ValueError("GeoJSON tidak memiliki poligon Taxi Zone dengan LocationID unik.")
    return {**geojson, "features": valid_features}


def metric_value(summary: pd.DataFrame, name: str, default: str = "—") -> str:
    rows = summary.loc[summary["metric"] == name, "value"]
    return str(rows.iloc[0]) if not rows.empty else default


def pretty_category(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["weather_category"] = result["weather_category"].map(CATEGORY_LABELS).fillna(result["weather_category"])
    return result


def filtered_dates(daily: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    frame = daily.copy()
    frame["pickup_date"] = pd.to_datetime(frame["pickup_date"])
    start = frame["pickup_date"].min().date()
    end = frame["pickup_date"].max().date()
    selected = st.sidebar.date_input(
        "Rentang tanggal pickup",
        value=(start, end),
        min_value=start,
        max_value=end,
        help="Filter ini memengaruhi visualisasi temporal yang mendukung rincian tanggal.",
    )
    if isinstance(selected, (tuple, list)) and len(selected) == 2:
        selected_start, selected_end = selected
    else:
        selected_start = selected_end = selected
    lower, upper = pd.Timestamp(selected_start), pd.Timestamp(selected_end)
    subset = frame[frame["pickup_date"].between(lower, upper)].copy()
    return lower, upper, subset


def chart_layout(fig: Any, height: int = 390) -> Any:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        font=dict(family="Arial, sans-serif", color="#263746"),
        legend_title_text="",
    )
    return fig


def render_overview(start: pd.Timestamp, end: pd.Timestamp, daily_all: pd.DataFrame, summary: pd.DataFrame, weather_summary: pd.DataFrame) -> None:
    st.markdown(
        '<div class="hero"><h2>Perjalanan HVFHV NYC dan kondisi cuaca</h2>'
        '<p>Ringkasan eksploratif Januari–Juni 2026 · sumber: agregasi tervalidasi Tahap 3</p></div>',
        unsafe_allow_html=True,
    )
    total = int(float(metric_value(summary, "source_trip_rows", "0")))
    matched = int(float(metric_value(summary, "weather_available_trip_rows", "0")))
    unavailable = int(float(metric_value(summary, "weather_unavailable_trip_rows", "0")))
    daily = daily_all.copy()
    daily["pickup_date"] = pd.to_datetime(daily["pickup_date"])
    chosen = daily[daily["pickup_date"].between(start, end)]
    period_total = int(chosen["trip_count"].sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Perjalanan (filter)", f"{period_total:,}")
    k2.metric("Perjalanan (semua)", f"{total:,}")
    k3.metric("Dengan cuaca", f"{matched:,}", f"{matched / total:.2%} dari total")
    k4.metric("Tanpa cuaca", f"{unavailable:,}", f"{unavailable / total:.2%} dari total")

    month = chosen.assign(pickup_month=chosen["pickup_date"].dt.to_period("M").astype(str)).groupby("pickup_month", as_index=False)["trip_count"].sum()
    fig = px.bar(month, x="pickup_month", y="trip_count", color_discrete_sequence=["#287b78"],
                 title="Perjalanan per bulan dalam rentang tanggal terpilih", labels={"pickup_month": "Bulan", "trip_count": "Jumlah perjalanan"})
    st.plotly_chart(chart_layout(fig, 350), width="stretch")

    dst = metric_value(summary, "dst_invalid_local_time_rows")
    weather_hours = metric_value(summary, "source_weather_zone_hour_observations")
    st.markdown(
        f'<div class="note"><b>Catatan kualitas:</b> {unavailable:,} perjalanan tidak memiliki pasangan cuaca; '
        f'{dst} perjalanan bertanda waktu DST lokal tidak valid. Sumber cuaca berisi {int(float(weather_hours)):,} '
        'observasi zona-jam. Zona tanpa geometri dan waktu DST dipertahankan, bukan dihapus.</div>',
        unsafe_allow_html=True,
    )


def render_temporal(start: pd.Timestamp, end: pd.Timestamp, daily_all: pd.DataFrame, date_hour_all: pd.DataFrame) -> None:
    st.title("Tren temporal")
    st.caption(f"Waktu pickup ditampilkan sebagai waktu lokal America/New_York · periode filter: {start:%Y-%m-%d} s.d. {end:%Y-%m-%d}")
    daily = daily_all.copy()
    daily["pickup_date"] = pd.to_datetime(daily["pickup_date"])
    daily = daily[daily["pickup_date"].between(start, end)]
    hourly_source = date_hour_all.copy()
    hourly_source["pickup_date"] = pd.to_datetime(hourly_source["pickup_date"])
    hourly_source = hourly_source[hourly_source["pickup_date"].between(start, end)]
    hourly = hourly_source.groupby("pickup_hour_local", as_index=False)["trip_count"].sum().sort_values("pickup_hour_local")
    weekday = hourly_source.assign(weekday=hourly_source["pickup_date"].dt.day_name()).groupby("weekday", as_index=False)["trip_count"].sum()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday["weekday"] = pd.Categorical(weekday["weekday"], categories=weekday_order, ordered=True)
    weekday = weekday.sort_values("weekday")

    left, right = st.columns([1.5, 1])
    with left:
        fig = px.line(daily, x="pickup_date", y="trip_count", title="Jumlah perjalanan per tanggal", markers=False,
                      labels={"pickup_date": "Tanggal pickup", "trip_count": "Jumlah perjalanan"})
        fig.update_traces(line_color="#287b78", line_width=2)
        st.plotly_chart(chart_layout(fig), width="stretch")
    with right:
        fig = px.bar(hourly, x="pickup_hour_local", y="trip_count", title="Perjalanan berdasarkan jam lokal",
                     labels={"pickup_hour_local": "Jam lokal (0–23)", "trip_count": "Jumlah perjalanan"}, color_discrete_sequence=["#4386c6"])
        st.plotly_chart(chart_layout(fig), width="stretch")
    fig = px.bar(weekday, x="weekday", y="trip_count", title="Perjalanan berdasarkan hari dalam minggu",
                 labels={"weekday": "Hari lokal", "trip_count": "Jumlah perjalanan"}, color_discrete_sequence=["#e69f45"])
    st.plotly_chart(chart_layout(fig, 330), width="stretch")
    st.caption("Grafik jam dan hari dihitung dari tabel tanggal-jam sehingga keduanya mengikuti filter tanggal. Jam 02:00 pada transisi DST musim semi memuat pickup yang dipertahankan dengan penanda kualitas.")


def render_weather(start: pd.Timestamp, end: pd.Timestamp, weather_monthly: pd.DataFrame, weather_summary: pd.DataFrame, definitions: pd.DataFrame) -> None:
    st.title("Perjalanan dan cuaca")
    st.markdown("Pola yang ditampilkan bersifat **deskriptif** dan tidak membuktikan bahwa cuaca menyebabkan perubahan volume perjalanan.")
    current = weather_summary.copy()
    current["category_label"] = current["weather_category"].map(CATEGORY_LABELS).fillna(current["weather_category"])
    valid = current[current["weather_category"] != "weather_unavailable"].copy()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(current, x="category_label", y="trip_count", color="category_label", color_discrete_map=COLORS,
                     title="Jumlah perjalanan menurut kategori", labels={"category_label": "Kategori cuaca", "trip_count": "Jumlah perjalanan"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_layout(fig), width="stretch")
    with c2:
        fig = px.bar(valid, x="category_label", y="mean_trips_per_zone_hour_observation", color="category_label", color_discrete_map=COLORS,
                     title="Rata-rata perjalanan per observasi zona-jam", labels={"category_label": "Kategori cuaca", "mean_trips_per_zone_hour_observation": "Perjalanan per observasi zona-jam"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_layout(fig), width="stretch")
    st.caption("Denominator rata-rata adalah jumlah observasi cuaca zona-jam valid (bukan jumlah baris perjalanan). Kategori tanpa cuaca tidak memiliki denominator observasi cuaca.")

    monthly = pretty_category(weather_monthly)
    monthly["month_start"] = pd.to_datetime(monthly["pickup_month"] + "-01")
    available_months = sorted(monthly["pickup_month"].unique())
    allowed = [m for m in available_months if pd.Timestamp(m + "-01") <= end and (pd.Timestamp(m + "-01") + pd.offsets.MonthEnd(0)) >= start]
    fig = px.line(monthly[monthly["pickup_month"].isin(allowed)], x="month_start", y="trip_count", color="weather_category",
                  color_discrete_map={v: COLORS[v] for v in COLORS}, markers=True, title="Tren bulanan menurut kategori cuaca",
                  labels={"month_start": "Bulan pickup", "trip_count": "Jumlah perjalanan", "weather_category": "Kategori cuaca"})
    st.plotly_chart(chart_layout(fig), width="stretch")
    if start.day != 1 or end.day != (end + pd.offsets.MonthEnd(0)).day:
        st.info("Tabel kategori cuaca hanya tersedia per bulan. Untuk bulan yang tersentuh filter tanggal sebagian, grafik tetap menampilkan agregat sebulan penuh.")
    with st.expander("Definisi kategori dan prioritas"):
        st.dataframe(definitions, width="stretch", hide_index=True)


def render_spatial(locations: pd.DataFrame, boroughs: pd.DataFrame, zone_weather: pd.DataFrame) -> None:
    st.title("Pola spasial pickup")
    st.caption("Zona dan borough berasal dari pemetaan Taxi Zone yang sudah terintegrasi; LocationID tidak diperlakukan sebagai borough.")
    locations = locations.copy()
    borough_names = sorted(x for x in locations["pickup_borough"].dropna().unique().tolist())
    selected_borough = st.selectbox("Borough", ["Semua borough", *borough_names])
    top_n = st.slider("Jumlah zona pada grafik", min_value=10, max_value=30, value=15, step=5)
    if selected_borough != "Semua borough":
        locations = locations[locations["pickup_borough"] == selected_borough]

    st.subheader("Peta volume pickup per Taxi Zone")
    st.caption("Warna menunjukkan jumlah pickup selama Januari–Juni 2026. Arahkan kursor ke zona untuk melihat jumlah perjalanan.")
    try:
        zone_geojson = load_taxi_zone_geojson()
        locations_for_map = locations.dropna(subset=["PULocationID"]).copy()
        locations_for_map["map_location_id"] = locations_for_map["PULocationID"].astype(int).astype(str)
        valid_geo_ids = {
            str(feature.get("properties", {}).get("locationid"))
            for feature in zone_geojson["features"]
        }
        mapped = locations_for_map[locations_for_map["map_location_id"].isin(valid_geo_ids)].copy()
        unmapped = locations_for_map[~locations_for_map["map_location_id"].isin(valid_geo_ids)]
        if mapped.empty:
            st.info("Tidak ada zona pada filter ini yang memiliki poligon valid untuk ditampilkan.")
        else:
            fig = px.choropleth(
                mapped,
                geojson=zone_geojson,
                locations="map_location_id",
                featureidkey="properties.locationid",
                color="trip_count",
                hover_name="pickup_zone",
                hover_data={
                    "map_location_id": False,
                    "PULocationID": True,
                    "pickup_borough": True,
                    "trip_count": ":,",
                },
                color_continuous_scale="YlOrRd",
                labels={
                    "trip_count": "Jumlah perjalanan",
                    "PULocationID": "Pickup LocationID",
                    "pickup_borough": "Borough",
                },
                title="Jumlah perjalanan pickup per zona",
            )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(coloraxis_colorbar_title="Trip", geo=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(chart_layout(fig, 610), width="stretch")
            st.caption(
                f"Poligon valid ditampilkan untuk {mapped['PULocationID'].nunique()} zona. "
                f"{int(unmapped['trip_count'].sum()):,} pickup pada "
                f"{unmapped['PULocationID'].nunique()} zona tidak dapat dipetakan karena poligon valid tidak tersedia; "
                "jumlah tersebut tetap tercakup pada grafik dan agregat lain."
            )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        st.info(f"Peta zona tidak tersedia: {exc}")

    top = locations.sort_values("trip_count", ascending=False).head(top_n).sort_values("trip_count")
    top["zone_label"] = top["pickup_zone"].fillna("Zona tidak diketahui") + " · " + top["PULocationID"].fillna(-1).astype(int).astype(str)
    fig = px.bar(top, x="trip_count", y="zone_label", orientation="h", color="pickup_borough",
                 title=f"{top_n} zona pickup dengan perjalanan terbanyak", labels={"trip_count": "Jumlah perjalanan", "zone_label": "Zona · LocationID", "pickup_borough": "Borough"})
    st.plotly_chart(chart_layout(fig, max(420, top_n * 28)), width="stretch")

    boroughs = boroughs.copy()
    boroughs["pickup_borough"] = boroughs["pickup_borough"].fillna("Tidak diketahui / tanpa pemetaan")
    boroughs = boroughs.sort_values("trip_count", ascending=False)
    fig = px.bar(boroughs, x="pickup_borough", y="trip_count", color_discrete_sequence=["#287b78"],
                 title="Volume perjalanan menurut borough", labels={"pickup_borough": "Borough", "trip_count": "Jumlah perjalanan"})
    st.plotly_chart(chart_layout(fig, 350), width="stretch")

    zone_weather = zone_weather.copy()
    if selected_borough != "Semua borough":
        zone_weather = zone_weather[zone_weather["pickup_borough"] == selected_borough]
    zone_weather["category_label"] = zone_weather["weather_category"].map(CATEGORY_LABELS).fillna(zone_weather["weather_category"])
    volume = zone_weather.groupby(["PULocationID", "pickup_borough", "pickup_zone", "category_label"], as_index=False)["trip_count"].sum()
    leading_ids = set(volume.groupby("PULocationID")["trip_count"].sum().nlargest(top_n).index)
    volume = volume[volume["PULocationID"].isin(leading_ids)]
    volume["zone_label"] = volume["pickup_zone"].fillna("Zona tidak diketahui") + " · " + volume["PULocationID"].fillna(-1).astype(int).astype(str)
    volume["category_label"] = pd.Categorical(volume["category_label"], categories=list(CATEGORY_LABELS.values()), ordered=True)
    fig = px.bar(volume, x="zone_label", y="trip_count", color="category_label", color_discrete_map=COLORS,
                 title="Perjalanan menurut zona dan kategori cuaca (zona teratas)", labels={"zone_label": "Zona · LocationID", "trip_count": "Jumlah perjalanan", "category_label": "Kategori cuaca"})
    fig.update_layout(barmode="stack", xaxis_tickangle=-45)
    st.plotly_chart(chart_layout(fig, 520), width="stretch")
    st.caption("Nilai kategori cuaca tidak tersedia tetap dipisahkan dari kategori cuaca teramati.")


def render_graph_analytics() -> None:
    st.title("Analisis jaringan perjalanan (Graph Analytics)")
    st.caption("Graf berarah berbobot: pickup zone ke drop-off zone; bobot adalah jumlah perjalanan.")
    summary = load_graph_csv("graph_summary.csv")
    zones = load_graph_csv("zone_metrics_by_month.csv")
    edges = load_graph_csv("edges_by_month.csv")

    available_periods = [p for p in summary["period"].astype(str) if p != "ALL_2026-01_2026-06"]
    selected_period = st.selectbox("Periode graf", ["Semua Jan-Jun 2026", *available_periods])
    period_key = "ALL_2026-01_2026-06" if selected_period == "Semua Jan-Jun 2026" else selected_period
    current = summary.loc[summary["period"].astype(str) == period_key]
    if current.empty:
        st.error(f"Ringkasan periode {period_key} tidak tersedia.")
        return
    row = current.iloc[0]

    cards = st.columns(5)
    cards[0].metric("Perjalanan terwakili", f"{int(row['trip_rows_represented']):,}")
    cards[1].metric("Zona aktif", f"{int(row['active_nodes']):,}")
    cards[2].metric("Pasangan OD unik", f"{int(row['unique_directed_od_pairs']):,}")
    cards[3].metric("Kepadatan berarah", f"{float(row['directed_density_excluding_self_loops']):.1%}")
    cards[4].metric("Perjalanan dalam zona sama", f"{int(row['self_loop_trip_rows']):,}")
    st.markdown(
        '<div class="note"><b>Cara membaca:</b> degree menghitung jumlah zona tetangga unik; '
        'strength menjumlahkan volume perjalanan. Satu weak component berarti seluruh zona terhubung '
        'jika arah perjalanan diabaikan, bukan berarti semua zona saling terjangkau dalam arah tertentu.</div>',
        unsafe_allow_html=True,
    )

    trend = summary[summary["period"].astype(str).str.match(r"^2026-\d{2}$")].copy()
    left, right = st.columns(2)
    with left:
        fig = px.line(trend, x="period", y="trip_rows_represented", markers=True,
                      title="Volume perjalanan pembentuk graf per bulan",
                      labels={"period": "Bulan", "trip_rows_represented": "Jumlah perjalanan"})
        st.plotly_chart(chart_layout(fig), width="stretch")
    with right:
        fig = px.line(trend, x="period", y="unique_directed_od_pairs", markers=True,
                      title="Pasangan OD berarah unik per bulan",
                      labels={"period": "Bulan", "unique_directed_od_pairs": "Pasangan OD unik"})
        st.plotly_chart(chart_layout(fig), width="stretch")

    edge_period = edges[edges["month"].astype(str) == period_key].copy() if period_key != "ALL_2026-01_2026-06" else edges.copy()
    if period_key == "ALL_2026-01_2026-06":
        edge_period = edge_period.groupby(["PULocationID", "DOLocationID"], as_index=False)["trip_count"].sum()
    zone_period = zones[zones["period"].astype(str) == period_key].copy()
    top_n = st.slider("Jumlah zona/rute pada grafik", min_value=5, max_value=20, value=10, step=5)
    metric_choice = st.selectbox("Metrik zona", ["Trip strength (volume)", "Degree (tetangga unik)"])
    if metric_choice.startswith("Trip"):
        origin_metric, destination_metric = "out_strength_trips", "in_strength_trips"
        origin_label, destination_label = "Perjalanan keluar", "Perjalanan masuk"
    else:
        origin_metric, destination_metric = "out_degree", "in_degree"
        origin_label, destination_label = "Tetangga tujuan unik", "Tetangga asal unik"

    a, b = st.columns(2)
    for container, column, title, label in [
        (a, origin_metric, f"{top_n} zona asal teratas", origin_label),
        (b, destination_metric, f"{top_n} zona tujuan teratas", destination_label),
    ]:
        top = zone_period.sort_values(column, ascending=False).head(top_n).sort_values(column)
        top["zone_label"] = top["zone"].fillna("Zona tidak diketahui") + " · " + top["LocationID"].astype(int).astype(str)
        fig = px.bar(top, x=column, y="zone_label", orientation="h", color="borough",
                     title=title, labels={column: label, "zone_label": "Taxi Zone · LocationID", "borough": "Borough"})
        container.plotly_chart(chart_layout(fig, max(380, top_n * 30)), width="stretch")

    labels = zone_period.set_index("LocationID")["zone"].to_dict()
    edge_period["route"] = edge_period.apply(
        lambda x: f"{labels.get(int(x['PULocationID']), 'Zona tidak diketahui')} ({int(x['PULocationID'])}) → "
                  f"{labels.get(int(x['DOLocationID']), 'Zona tidak diketahui')} ({int(x['DOLocationID'])})",
        axis=1,
    )
    top_edges = edge_period.nlargest(top_n, "trip_count").sort_values("trip_count")
    fig = px.bar(top_edges, x="trip_count", y="route", orientation="h",
                 title=f"{top_n} rute pickup → drop-off dengan volume tertinggi",
                 labels={"trip_count": "Jumlah perjalanan", "route": "Rute berarah"},
                 color_discrete_sequence=["#287b78"])
    st.plotly_chart(chart_layout(fig, max(420, top_n * 34)), width="stretch")

    st.subheader("Peta kekuatan aliran antarzona")
    map_metric = st.selectbox("Nilai pada peta", ["Volume pickup keluar", "Volume drop-off masuk"])
    map_column = "out_strength_trips" if map_metric == "Volume pickup keluar" else "in_strength_trips"
    try:
        geojson = load_taxi_zone_geojson()
        valid_ids = {str(f.get("properties", {}).get("locationid")) for f in geojson["features"]}
        map_frame = zone_period.copy()
        map_frame["map_location_id"] = map_frame["LocationID"].astype(int).astype(str)
        mapped = map_frame[map_frame["map_location_id"].isin(valid_ids)]
        missing_map = map_frame[~map_frame["map_location_id"].isin(valid_ids)]
        if mapped.empty:
            st.info("Tidak ada zona dengan poligon valid untuk periode ini.")
        else:
            fig = px.choropleth(
                mapped, geojson=geojson, locations="map_location_id", featureidkey="properties.locationid",
                color=map_column, hover_name="zone",
                hover_data={"map_location_id": False, "LocationID": True, "borough": True, map_column: ":,"},
                color_continuous_scale="YlOrRd", title=map_metric,
                labels={map_column: "Perjalanan", "LocationID": "LocationID", "borough": "Borough"},
            )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(coloraxis_colorbar_title="Trip", geo=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(chart_layout(fig, 610), width="stretch")
        st.caption(
            f"Peta menampilkan {mapped['LocationID'].nunique()} zona dengan poligon valid. "
            f"{missing_map['LocationID'].nunique()} zona aktif tidak dapat dipetakan pada periode ini; "
            "datanya tetap tercakup pada grafik dan tabel agregat."
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        st.info(f"Peta aliran zona tidak tersedia: {exc}")

    st.caption("Sumber: agregat Graph Analytics tervalidasi, run 20261107T174911Z. Data perjalanan tingkat baris tidak dibaca oleh halaman ini.")


def main() -> None:
    st.sidebar.title("NYC TLC · HVFHV")
    page = st.sidebar.radio("Bagian dashboard", ["Overview", "Temporal Trends", "Weather Analysis", "Spatial Analysis", "Graph Analytics"])
    if page == "Graph Analytics":
        try:
            render_graph_analytics()
        except Exception as exc:
            st.error(f"Dashboard Graph Analytics gagal membaca agregat: {exc}")
            st.info("Pastikan agregat graph run tervalidasi tersedia di output/analysis/graph_analytics/20261107T174911Z.")
        return
    try:
        daily = load_parquet("daily")
        if page == "Spatial Analysis":
            # Spatial output is full-period only; its precomputed tables do not
            # carry pickup_date, so avoid displaying an ineffective date filter.
            daily_dates = pd.to_datetime(daily["pickup_date"])
            start, end = daily_dates.min(), daily_dates.max()
            st.sidebar.caption("Analisis spasial memakai seluruh periode; tabel zona tidak menyimpan dimensi tanggal.")
        else:
            start, end, _ = filtered_dates(daily)
        summary = load_csv("analysis_summary.csv")
        weather_summary = load_csv("weather_category_summary.csv")
        if metric_value(summary, "validation_status") != "VALIDATION_PASSED":
            st.warning("Laporan Tahap 3 tidak berstatus VALIDATION_PASSED. Periksa laporan kualitas sebelum menggunakan dashboard.")
        else:
            st.sidebar.success("Validasi Tahap 3 lulus")

        if page == "Overview":
            render_overview(start, end, daily, summary, weather_summary)
        elif page == "Temporal Trends":
            render_temporal(start, end, daily, load_parquet("date_hour"))
        elif page == "Weather Analysis":
            render_weather(start, end, load_parquet("weather_monthly"), weather_summary, load_csv("weather_category_definitions.csv"))
        else:
            render_spatial(load_parquet("locations"), load_parquet("boroughs"), load_parquet("zone_hour_weather"))

        with st.sidebar.expander("Sumber dan kualitas data"):
            st.caption("Sumber: output/analysis/weather_exploratory dan output/quality/weather_exploratory")
            st.caption("Pickup time: America/New_York · kunci join cuaca menggunakan jam UTC.")
            st.caption("Nilai cuaca null dipertahankan dan bukan dianggap sebagai kondisi kering.")
    except Exception as exc:
        st.error(f"Dashboard gagal membaca agregasi: {exc}")
        st.info("Pastikan aplikasi dijalankan dari proyek yang memiliki output Tahap 3 di lokasi yang diharapkan.")


if __name__ == "__main__":
    main()
