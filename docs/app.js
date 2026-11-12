"use strict";

const DATA_FILES = [
  "daily.json",
  "date_hour.json",
  "weather_monthly.json",
  "locations.json",
  "boroughs.json",
  "zone_weather.json",
  "analysis_summary.json",
  "weather_category_summary.json",
  "weather_category_definitions.json",
  "taxi_zones.json",
  "graph_summary.json",
  "zone_metrics.json",
  "top_graph_edges.json",
];
const NUMBER = new Intl.NumberFormat("id-ID");
const CATEGORY_LABELS = {
  no_precipitation: "Tanpa presipitasi",
  rain: "Hujan",
  snow: "Salju",
  mixed_or_unclassified_precipitation: "Campuran / tidak terklasifikasi",
  weather_unavailable: "Cuaca tidak tersedia",
};
const COLORS = {
  "Tanpa presipitasi": "#2f7d67",
  Hujan: "#4386c6",
  Salju: "#8c78c9",
  "Campuran / tidak terklasifikasi": "#e69f45",
  "Cuaca tidak tersedia": "#9aa4b2",
};
const WEEKDAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"];
const PLOT_LAYOUT = {
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "DM Sans, Segoe UI, sans-serif", color: "#354b55", size: 12 },
  margin: { l: 58, r: 18, t: 55, b: 55 },
  legend: { orientation: "h", y: -0.22 },
  xaxis: { gridcolor: "#edf1f1", automargin: true },
  yaxis: { gridcolor: "#edf1f1", automargin: true, tickformat: ",d" },
};
const PLOT_CONFIG = { responsive: true, displaylogo: false };
let data;
let currentPage = "overview";

function sum(rows, field) {
  return rows.reduce((total, row) => total + Number(row[field] || 0), 0);
}

function formatNumber(value) {
  return NUMBER.format(Number(value) || 0);
}

function dateValue(value) {
  return typeof value === "string" ? value.slice(0, 10) : "";
}

function dateRange() {
  return {
    start: document.getElementById("start-date").value,
    end: document.getElementById("end-date").value,
  };
}

function inRange(date, range) {
  const day = dateValue(date);
  return day >= range.start && day <= range.end;
}

function labelCategory(value) {
  return CATEGORY_LABELS[value] || value;
}

function recordsBy(rows, key) {
  const result = new Map();
  for (const row of rows) {
    const name = String(row[key]);
    result.set(name, (result.get(name) || 0) + Number(row.trip_count || 0));
  }
  return result;
}

function setMetrics(elementId, cards) {
  const container = document.getElementById(elementId);
  container.replaceChildren();
  for (const [label, value, note = ""] of cards) {
    const card = document.createElement("article");
    card.className = "metric";
    const title = document.createElement("p");
    title.className = "metric-label";
    title.textContent = label;
    const amount = document.createElement("p");
    amount.className = "metric-value";
    amount.textContent = value;
    const detail = document.createElement("p");
    detail.className = "metric-note";
    detail.textContent = note;
    card.append(title, amount, detail);
    container.append(card);
  }
}

function plot(elementId, traces, title, options = {}) {
  const layout = {
    ...PLOT_LAYOUT,
    title: { text: title, x: 0.02, xanchor: "left", font: { size: 15, color: "#203a47" } },
    ...options,
  };
  return Plotly.react(elementId, traces, layout, PLOT_CONFIG);
}

function metricValue(name, fallback = "0") {
  const row = data.analysisSummary.find((item) => item.metric === name);
  return row ? row.value : fallback;
}

function renderOverview(range) {
  const total = Number(metricValue("source_trip_rows"));
  const matched = Number(metricValue("weather_available_trip_rows"));
  const unavailable = Number(metricValue("weather_unavailable_trip_rows"));
  const daily = data.daily.filter((row) => inRange(row.pickup_date, range));
  const periodTotal = sum(daily, "trip_count");
  setMetrics("overview-metrics", [
    ["Perjalanan (filter)", formatNumber(periodTotal), `${range.start} s.d. ${range.end}`],
    ["Perjalanan (semua)", formatNumber(total), "Januari–Juni 2026"],
    ["Dengan data cuaca", formatNumber(matched), `${(matched / total * 100).toFixed(2)}% dari total`],
    ["Data cuaca tidak tersedia", formatNumber(unavailable), `${(unavailable / total * 100).toFixed(2)}% dari total`],
  ]);
  const byMonth = new Map();
  for (const row of daily) {
    const month = dateValue(row.pickup_date).slice(0, 7);
    byMonth.set(month, (byMonth.get(month) || 0) + Number(row.trip_count));
  }
  const months = [...byMonth.keys()].sort();
  plot("overview-monthly", [{
    type: "bar",
    x: months,
    y: months.map((month) => byMonth.get(month)),
    marker: { color: "#287b78", borderRadius: 4 },
    hovertemplate: "%{x}<br>%{y:,} perjalanan<extra></extra>",
  }], "Perjalanan per bulan dalam rentang tanggal terpilih", {
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Bulan pickup" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Jumlah perjalanan" },
  });
  const dst = metricValue("dst_invalid_local_time_rows");
  const weatherHours = Number(metricValue("source_weather_zone_hour_observations"));
  document.getElementById("quality-note").textContent =
    `Catatan kualitas: ${formatNumber(unavailable)} perjalanan tidak memiliki pasangan cuaca; ` +
    `${formatNumber(dst)} pickup memiliki waktu lokal DST yang tidak valid. Sumber cuaca berisi ` +
    `${formatNumber(weatherHours)} observasi zona-jam. Zona tanpa geometri dan waktu DST dipertahankan, bukan dihapus.`;
}

function renderTemporal(range) {
  const daily = data.daily.filter((row) => inRange(row.pickup_date, range))
    .sort((a, b) => dateValue(a.pickup_date).localeCompare(dateValue(b.pickup_date)));
  plot("daily-chart", [{
    type: "scatter",
    mode: "lines",
    x: daily.map((row) => dateValue(row.pickup_date)),
    y: daily.map((row) => Number(row.trip_count)),
    line: { color: "#287b78", width: 2 },
    hovertemplate: "%{x}<br>%{y:,} perjalanan<extra></extra>",
  }], "Jumlah perjalanan per tanggal", {
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Tanggal pickup" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Jumlah perjalanan" },
  });

  const hourlyRows = data.dateHour.filter((row) => inRange(row.pickup_date, range));
  const hourly = Array.from({ length: 24 }, (_, hour) => ({
    hour,
    trips: sum(hourlyRows.filter((row) => Number(row.pickup_hour_local) === hour), "trip_count"),
  }));
  plot("hour-chart", [{
    type: "bar",
    x: hourly.map((row) => String(row.hour).padStart(2, "0")),
    y: hourly.map((row) => row.trips),
    marker: { color: "#4386c6" },
    hovertemplate: "Jam %{x}:00<br>%{y:,} perjalanan<extra></extra>",
  }], "Perjalanan berdasarkan jam lokal", {
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Jam lokal" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Jumlah perjalanan" },
  });

  const weekdayCounts = Array(7).fill(0);
  for (const row of hourlyRows) {
    const [year, month, day] = dateValue(row.pickup_date).split("-").map(Number);
    const dayIndex = (new Date(Date.UTC(year, month - 1, day)).getUTCDay() + 6) % 7;
    weekdayCounts[dayIndex] += Number(row.trip_count);
  }
  plot("weekday-chart", [{
    type: "bar",
    x: WEEKDAYS,
    y: weekdayCounts,
    marker: { color: "#e69f45" },
    hovertemplate: "%{x}<br>%{y:,} perjalanan<extra></extra>",
  }], "Perjalanan berdasarkan hari dalam minggu", {
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Hari pickup lokal" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Jumlah perjalanan" },
  });
}

function renderWeather(range) {
  const summary = data.weatherSummary.map((row) => ({
    ...row,
    label: labelCategory(row.weather_category),
  }));
  const valid = summary.filter((row) => row.weather_category !== "weather_unavailable");
  plot("weather-count-chart", [{
    type: "bar",
    x: summary.map((row) => row.label),
    y: summary.map((row) => Number(row.trip_count)),
    marker: { color: summary.map((row) => COLORS[row.label] || "#287b78") },
    hovertemplate: "%{x}<br>%{y:,} perjalanan<extra></extra>",
  }], "Jumlah perjalanan menurut kategori cuaca", {
    showlegend: false,
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Kategori cuaca" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Jumlah perjalanan" },
  });
  plot("weather-mean-chart", [{
    type: "bar",
    x: valid.map((row) => row.label),
    y: valid.map((row) => Number(row.mean_trips_per_zone_hour_observation)),
    marker: { color: valid.map((row) => COLORS[row.label] || "#287b78") },
    hovertemplate: "%{x}<br>%{y:.2f} perjalanan per observasi<extra></extra>",
  }], "Rata-rata perjalanan per observasi zona-jam", {
    showlegend: false,
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Kategori cuaca" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Perjalanan per observasi" },
  });

  const startMonth = range.start.slice(0, 7);
  const endMonth = range.end.slice(0, 7);
  const visible = data.weatherMonthly.filter((row) => (
    row.pickup_month >= startMonth && row.pickup_month <= endMonth
  ));
  const categories = [...new Set(visible.map((row) => labelCategory(row.weather_category)))];
  const months = [...new Set(visible.map((row) => row.pickup_month))].sort();
  plot("weather-month-chart", categories.map((category) => ({
    type: "scatter",
    mode: "lines+markers",
    name: category,
    x: months,
    y: months.map((month) => {
      const row = visible.find((item) => (
        item.pickup_month === month && labelCategory(item.weather_category) === category
      ));
      return row ? Number(row.trip_count) : null;
    }),
    line: { color: COLORS[category] || "#287b78" },
    hovertemplate: `${category}<br>%{x}<br>%{y:,} perjalanan<extra></extra>`,
  })), "Tren bulanan menurut kategori cuaca", {
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Bulan pickup" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Jumlah perjalanan" },
  });
  renderDefinitions();
}

function renderDefinitions() {
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const label of ["Kategori", "Aturan", "Prioritas"]) {
    const cell = document.createElement("th");
    cell.textContent = label;
    headerRow.append(cell);
  }
  head.append(headerRow);
  const body = document.createElement("tbody");
  for (const row of data.weatherDefinitions) {
    const tableRow = document.createElement("tr");
    for (const value of [labelCategory(row.category), row.rule, row.priority || "—"]) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      tableRow.append(cell);
    }
    body.append(tableRow);
  }
  table.append(head, body);
  document.getElementById("weather-definitions").replaceChildren(table);
}

function selectedLocations() {
  const borough = document.getElementById("borough-filter").value;
  return data.locations.filter((row) => borough === "Semua borough" || row.pickup_borough === borough);
}

function selectedZoneWeather() {
  const borough = document.getElementById("borough-filter").value;
  return data.zoneWeather.filter((row) => borough === "Semua borough" || row.pickup_borough === borough);
}

function renderSpatial() {
  const locations = selectedLocations();
  const zoneWeather = selectedZoneWeather();
  const limit = Number(document.getElementById("zone-limit").value);
  const features = data.taxiZones.features;
  const geoIds = new Set(features.map((feature) => String(feature.properties.locationid)));
  const mapped = locations.filter((row) => (
    row.PULocationID != null && geoIds.has(String(Math.trunc(Number(row.PULocationID))))
  ));
  const unmapped = locations.filter((row) => !mapped.includes(row));

  if (mapped.length) {
    plot("pickup-map", [{
      type: "choropleth",
      geojson: data.taxiZones,
      featureidkey: "properties.locationid",
      locations: mapped.map((row) => String(Math.trunc(Number(row.PULocationID)))),
      z: mapped.map((row) => Number(row.trip_count)),
      text: mapped.map((row) => row.pickup_zone || "Zona tidak diketahui"),
      customdata: mapped.map((row) => [row.PULocationID, row.pickup_borough, row.trip_count]),
      colorscale: "YlOrRd",
      marker: { line: { color: "white", width: 0.45 } },
      colorbar: { title: "Perjalanan" },
      hovertemplate: "%{text}<br>LocationID %{customdata[0]}<br>%{customdata[1]}<br>%{z:,} perjalanan<extra></extra>",
    }], "Jumlah perjalanan pickup per zona", {
      height: 560,
      geo: { fitbounds: "locations", visible: false, bgcolor: "rgba(0,0,0,0)" },
      margin: { l: 5, r: 5, t: 45, b: 5 },
      showlegend: false,
    });
  } else {
    document.getElementById("pickup-map").textContent = "Tidak ada zona pada filter ini yang memiliki poligon valid.";
  }
  document.getElementById("pickup-map-note").textContent =
    `Poligon valid ditampilkan untuk ${mapped.length} zona. ` +
    `${formatNumber(sum(unmapped, "trip_count"))} pickup pada ${unmapped.length} zona ` +
    "tidak dapat dipetakan; jumlah tersebut tetap tercakup pada grafik dan agregat.";

  const top = [...locations].sort((a, b) => Number(b.trip_count) - Number(a.trip_count))
    .slice(0, limit).reverse();
  const zoneLabels = top.map((row) => `${row.pickup_zone || "Zona tidak diketahui"} · ${row.PULocationID ?? -1}`);
  const boroughs = [...data.boroughs].sort((a, b) => Number(b.trip_count) - Number(a.trip_count));
  const boroughLabels = boroughs.map((row) => row.pickup_borough || "Tidak diketahui / tanpa pemetaan");
  plot("top-zones-chart", [{
    type: "bar",
    orientation: "h",
    y: zoneLabels,
    x: top.map((row) => Number(row.trip_count)),
    marker: { color: top.map((row) => COLORS[row.pickup_borough] || "#287b78") },
    customdata: top.map((row) => row.pickup_borough || "Tidak diketahui"),
    hovertemplate: "%{y}<br>%{x:,} perjalanan<br>%{customdata}<extra></extra>",
  }], `${limit} zona pickup dengan perjalanan terbanyak`, {
    height: Math.max(380, limit * 27),
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Jumlah perjalanan" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "" },
    showlegend: false,
  });
  plot("borough-chart", [{
    type: "bar",
    x: boroughLabels,
    y: boroughs.map((row) => Number(row.trip_count)),
    marker: { color: "#287b78" },
    hovertemplate: "%{x}<br>%{y:,} perjalanan<extra></extra>",
  }], "Volume perjalanan menurut borough", {
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Borough" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Jumlah perjalanan" },
    showlegend: false,
  });

  const volume = new Map();
  for (const row of zoneWeather) {
    const id = String(row.PULocationID);
    volume.set(id, (volume.get(id) || 0) + Number(row.trip_count));
  }
  const topIds = new Set([...volume.entries()]
    .sort((a, b) => b[1] - a[1]).slice(0, limit).map(([id]) => id));
  const categoryOrder = Object.keys(CATEGORY_LABELS).map((key) => CATEGORY_LABELS[key]);
  const topZoneRows = zoneWeather.filter((row) => topIds.has(String(row.PULocationID)));
  const topZoneLabels = [...new Map(topZoneRows.map((row) => [
    String(row.PULocationID),
    `${row.pickup_zone || "Zona tidak diketahui"} · ${row.PULocationID}`,
  ]))].sort((a, b) => (volume.get(b[0]) || 0) - (volume.get(a[0]) || 0)).map((item) => item[1]);
  plot("zone-weather-chart", categoryOrder.map((category) => {
    const rows = topZoneRows.filter((row) => labelCategory(row.weather_category) === category);
    const byZone = new Map(rows.map((row) => [String(row.PULocationID), Number(row.trip_count)]));
    return {
      type: "bar",
      name: category,
      x: topZoneLabels,
      y: [...new Map(topZoneRows.map((row) => [
        String(row.PULocationID),
        `${row.pickup_zone || "Zona tidak diketahui"} · ${row.PULocationID}`,
      ]))].sort((a, b) => (volume.get(b[0]) || 0) - (volume.get(a[0]) || 0)).map(([id]) => byZone.get(id) || 0),
      marker: { color: COLORS[category] },
      hovertemplate: `${category}<br>%{x}<br>%{y:,} perjalanan<extra></extra>`,
    };
  }), "Perjalanan per zona dan kategori cuaca", {
    barmode: "stack",
    height: 510,
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Zona · LocationID", tickangle: -35 },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Jumlah perjalanan" },
  });
}

function graphPeriod() {
  return document.getElementById("graph-period").value;
}

function renderGraph() {
  const period = graphPeriod();
  const allPeriod = "ALL_2026-01_2026-06";
  const row = data.graphSummary.find((item) => item.period === period);
  if (!row) {
    throw new Error(`Ringkasan periode graf ${period} tidak tersedia.`);
  }
  setMetrics("graph-metrics", [
    ["Perjalanan terwakili", formatNumber(row.trip_rows_represented), "Trip rows dalam graf"],
    ["Zona aktif", formatNumber(row.active_nodes), "Node pada periode terpilih"],
    ["Pasangan OD unik", formatNumber(row.unique_directed_od_pairs), "Pickup → drop-off"],
    ["Kepadatan berarah", `${(Number(row.directed_density_excluding_self_loops) * 100).toFixed(1)}%`, "Tanpa self-loop"],
    ["Perjalanan dalam zona sama", formatNumber(row.self_loop_trip_rows), "Self-loop"],
  ]);

  const trend = data.graphSummary.filter((item) => /^\d{4}-\d{2}$/.test(item.period))
    .sort((a, b) => a.period.localeCompare(b.period));
  plot("graph-trips-chart", [{
    type: "scatter", mode: "lines+markers",
    x: trend.map((item) => item.period), y: trend.map((item) => Number(item.trip_rows_represented)),
    line: { color: "#287b78" }, marker: { color: "#287b78" },
    hovertemplate: "%{x}<br>%{y:,} perjalanan<extra></extra>",
  }], "Volume perjalanan pembentuk graf per bulan", {
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Bulan" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Jumlah perjalanan" },
  });
  plot("graph-pairs-chart", [{
    type: "scatter", mode: "lines+markers",
    x: trend.map((item) => item.period), y: trend.map((item) => Number(item.unique_directed_od_pairs)),
    line: { color: "#4386c6" }, marker: { color: "#4386c6" },
    hovertemplate: "%{x}<br>%{y:,} pasangan OD unik<extra></extra>",
  }], "Pasangan OD berarah unik per bulan", {
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Bulan" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Pasangan OD unik" },
  });

  const limit = Number(document.getElementById("graph-limit").value);
  const metric = document.getElementById("graph-metric").value;
  const originField = metric === "strength" ? "out_strength_trips" : "out_degree";
  const destinationField = metric === "strength" ? "in_strength_trips" : "in_degree";
  const originTitle = metric === "strength" ? "Perjalanan keluar" : "Tetangga tujuan unik";
  const destinationTitle = metric === "strength" ? "Perjalanan masuk" : "Tetangga asal unik";
  const zoneRows = data.zoneMetrics.filter((item) => item.period === period);
  function drawZoneRanking(elementId, field, title) {
    const top = [...zoneRows].sort((a, b) => Number(b[field]) - Number(a[field]))
      .slice(0, limit).reverse();
    plot(elementId, [{
      type: "bar",
      orientation: "h",
      y: top.map((item) => `${item.zone || "Zona tidak diketahui"} · ${item.LocationID}`),
      x: top.map((item) => Number(item[field])),
      marker: { color: "#287b78" },
      customdata: top.map((item) => item.borough || "Tidak diketahui"),
      hovertemplate: "%{y}<br>%{x:,}<br>%{customdata}<extra></extra>",
    }], title, {
      height: Math.max(370, limit * 29),
      xaxis: { ...PLOT_LAYOUT.xaxis, title: metric === "strength" ? "Jumlah perjalanan" : "Jumlah tetangga" },
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "" },
      showlegend: false,
    });
  }
  drawZoneRanking("graph-origins-chart", originField, `${limit} zona asal teratas · ${originTitle}`);
  drawZoneRanking("graph-destinations-chart", destinationField, `${limit} zona tujuan teratas · ${destinationTitle}`);

  const routes = data.topGraphEdges.filter((item) => item.period === period)
    .sort((a, b) => Number(b.trip_count) - Number(a.trip_count))
    .slice(0, limit).reverse();
  const labels = new Map(zoneRows.map((item) => [String(item.LocationID), item.zone || "Zona tidak diketahui"]));
  plot("top-routes-chart", [{
    type: "bar",
    orientation: "h",
    y: routes.map((item) => (
      `${labels.get(String(item.PULocationID)) || "Zona tidak diketahui"} (${item.PULocationID}) → ` +
      `${labels.get(String(item.DOLocationID)) || "Zona tidak diketahui"} (${item.DOLocationID})`
    )),
    x: routes.map((item) => Number(item.trip_count)),
    marker: { color: "#287b78" },
    hovertemplate: "%{y}<br>%{x:,} perjalanan<extra></extra>",
  }], `${limit} rute pickup → drop-off dengan volume tertinggi`, {
    height: Math.max(400, limit * 34),
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Jumlah perjalanan" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "" },
    showlegend: false,
  });

  const mapField = document.getElementById("graph-map-metric").value;
  const mapTitle = mapField === "out_strength_trips" ? "Volume pickup keluar" : "Volume drop-off masuk";
  const featureIds = new Set(data.taxiZones.features.map((feature) => String(feature.properties.locationid)));
  const mapped = zoneRows.filter((item) => featureIds.has(String(Math.trunc(Number(item.LocationID)))));
  plot("graph-map", [{
    type: "choropleth",
    geojson: data.taxiZones,
    featureidkey: "properties.locationid",
    locations: mapped.map((item) => String(Math.trunc(Number(item.LocationID)))),
    z: mapped.map((item) => Number(item[mapField])),
    text: mapped.map((item) => item.zone || "Zona tidak diketahui"),
    customdata: mapped.map((item) => [item.LocationID, item.borough]),
    colorscale: "YlOrRd",
    marker: { line: { color: "white", width: 0.45 } },
    colorbar: { title: "Perjalanan" },
    hovertemplate: "%{text}<br>LocationID %{customdata[0]}<br>%{customdata[1]}<br>%{z:,} perjalanan<extra></extra>",
  }], mapTitle, {
    height: 560,
    geo: { fitbounds: "locations", visible: false, bgcolor: "rgba(0,0,0,0)" },
    margin: { l: 5, r: 5, t: 45, b: 5 },
    showlegend: false,
  });
}

function renderActivePage() {
  const range = dateRange();
  if (!range.start || !range.end || range.start > range.end) {
    throw new Error("Rentang tanggal tidak valid. Pastikan tanggal mulai tidak melewati tanggal akhir.");
  }
  if (currentPage === "overview") renderOverview(range);
  if (currentPage === "temporal") renderTemporal(range);
  if (currentPage === "weather") renderWeather(range);
  if (currentPage === "spatial") renderSpatial();
  if (currentPage === "graph") renderGraph();
}

function refreshDashboard() {
  const status = document.getElementById("status");
  try {
    renderActivePage();
    status.hidden = true;
    status.classList.remove("error");
  } catch (error) {
    status.hidden = false;
    status.classList.add("error");
    status.textContent = `Dashboard tidak dapat diperbarui.\n${error instanceof Error ? error.message : String(error)}`;
  }
}

function showPage(name) {
  currentPage = name;
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.page === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-current", active ? "page" : "false");
  });
  document.querySelectorAll(".page").forEach((page) => {
    const active = page.id === `page-${name}`;
    page.classList.toggle("active", active);
    page.hidden = !active;
  });
  document.querySelector(".date-filter").hidden = name === "spatial" || name === "graph";
  refreshDashboard();
}

function setupFilters() {
  const dates = data.daily.map((row) => dateValue(row.pickup_date)).sort();
  document.getElementById("start-date").min = dates[0];
  document.getElementById("start-date").max = dates[dates.length - 1];
  document.getElementById("end-date").min = dates[0];
  document.getElementById("end-date").max = dates[dates.length - 1];
  document.getElementById("start-date").value = dates[0];
  document.getElementById("end-date").value = dates[dates.length - 1];

  const boroughs = [...new Set(data.locations.map((row) => row.pickup_borough).filter(Boolean))].sort();
  const boroughSelect = document.getElementById("borough-filter");
  for (const borough of ["Semua borough", ...boroughs]) {
    const option = document.createElement("option");
    option.value = borough;
    option.textContent = borough;
    boroughSelect.append(option);
  }

  const periodSelect = document.getElementById("graph-period");
  const periods = data.graphSummary.filter((row) => row.period !== "ALL_2026-01_2026-06");
  for (const row of periods) {
    const option = document.createElement("option");
    option.value = row.period;
    option.textContent = row.period === "2026-01" ? "Januari 2026"
      : row.period === "2026-02" ? "Februari 2026"
        : row.period === "2026-03" ? "Maret 2026"
          : row.period === "2026-04" ? "April 2026"
            : row.period === "2026-05" ? "Mei 2026" : "Juni 2026";
    periodSelect.append(option);
  }
  const allOption = document.createElement("option");
  allOption.value = "ALL_2026-01_2026-06";
  allOption.textContent = "Semua Jan–Jun 2026";
  periodSelect.append(allOption);
  periodSelect.value = allOption.value;
}

async function loadData() {
  const entries = await Promise.all(DATA_FILES.map(async (filename) => {
    const response = await fetch(`./data/${filename}`);
    if (!response.ok) {
      throw new Error(`File data ${filename} tidak dapat dimuat (HTTP ${response.status}). Jalankan docs/export_data.py, lalu push folder docs/data/ ke GitHub.`);
    }
    return [filename, await response.json()];
  }));
  return Object.fromEntries(entries);
}

async function initialize() {
  const status = document.getElementById("status");
  try {
    if (!window.Plotly) {
      throw new Error("Pustaka grafik Plotly tidak termuat. Periksa koneksi internet atau pemblokir konten, lalu muat ulang halaman.");
    }
    const loaded = await loadData();
    data = {
      daily: loaded["daily.json"],
      dateHour: loaded["date_hour.json"],
      weatherMonthly: loaded["weather_monthly.json"],
      locations: loaded["locations.json"],
      boroughs: loaded["boroughs.json"],
      zoneWeather: loaded["zone_weather.json"],
      analysisSummary: loaded["analysis_summary.json"],
      weatherSummary: loaded["weather_category_summary.json"],
      weatherDefinitions: loaded["weather_category_definitions.json"],
      taxiZones: loaded["taxi_zones.json"],
      graphSummary: loaded["graph_summary.json"],
      zoneMetrics: loaded["zone_metrics.json"],
      topGraphEdges: loaded["top_graph_edges.json"],
    };
    setupFilters();
    document.getElementById("dashboard").hidden = false;
    status.hidden = true;

    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => showPage(tab.dataset.page));
    });
    for (const id of ["start-date", "end-date", "borough-filter", "zone-limit", "graph-period", "graph-limit", "graph-metric", "graph-map-metric"]) {
      document.getElementById(id).addEventListener("change", refreshDashboard);
    }
    refreshDashboard();
  } catch (error) {
    status.classList.add("error");
    status.textContent = `Dashboard tidak dapat dimuat.\n${error instanceof Error ? error.message : String(error)}`;
  }
}

initialize();
