# GitHub Pages dashboard

This folder contains a browser-only version of the NYC HVFHV dashboard. It does not need Python, Streamlit, or a server after GitHub Pages publishes it.

## Prepare the data

The site uses only small, pre-aggregated tables and Taxi Zone geometry. It never exports the raw trip files. From the project root, run the exporter in the project's Python environment:

```powershell
.\.venv\Scripts\python.exe .\docs\export_data.py
```

Or, from WSL:

```bash
python docs/export_data.py
```

The exporter writes the JSON files into `docs/data/`. Check the generated record counts, then add the `docs/` folder—including `docs/data/`—to the repository. The website will show a clear error if those generated data files are missing.

Because `.gitignore` excludes the source `data/`, `output/`, Parquet, and GeoJSON files, the exported JSON inside `docs/data/` is the shareable, static copy. Review the generated files before publishing; do not export or add raw trip-level data.

## Publish from GitHub

1. Push the repository, including `docs/index.html`, the other site files, and `docs/data/*.json`, to GitHub.
2. Open the repository's **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Choose branch `main` and folder `/docs`, then save.
5. Wait for the Pages deployment to finish. Open the URL shown in the Pages settings and share it.

Whenever the source aggregates change, run the exporter again and push the updated `docs/data/` files.

## Local preview

From the project root, start a simple static server:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/docs/`. The page loads Plotly.js and its fonts from public CDNs, so those chart and font resources require an internet connection.
