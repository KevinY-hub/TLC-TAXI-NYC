"""Append project-result slides to the supplied Big Data PowerPoint.

The source deck is read-only. New slides are rendered as high-resolution PNG
slides, matching its dark navy/yellow visual style, and inserted before its
existing final slide. Output is never overwritten.
"""

from __future__ import annotations

import csv
import re
import shutil
import zipfile
from pathlib import Path
from textwrap import wrap
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\USER\Downloads\Big Data.pptx")
OUTPUT_DIR = ROOT / "output" / "presentation"
OUTPUT = OUTPUT_DIR / "Big Data - continued.pptx"
ASSET_DIR = OUTPUT_DIR / "continued_slides"
GRAPH_DIR = ROOT / "output" / "analysis" / "graph_analytics" / "20261107T174911Z"
GRAPH_IMAGES = GRAPH_DIR / "visualizations_chapter_vi_v4"
ML_DIR = ROOT / "output" / "machine_learning" / "controlled_comparison" / "20261106T204756Z" / "reports"
WEATHER_SUMMARY = ROOT / "output" / "quality" / "weather_exploratory" / "analysis_summary.csv"
GRAPH_SUMMARY = GRAPH_DIR / "graph_summary.csv"
ML_METRICS = ML_DIR / "controlled_metrics.csv"

WIDTH, HEIGHT = 1920, 1080
EMU_W, EMU_H = 18_288_000, 10_287_000
BG = "111820"
PANEL = "202C38"
PANEL_LIGHT = "F1F3F4"
YELLOW = "FFB900"
TEAL = "35B7A5"
BLUE = "68A9E8"
WHITE = "FFFFFF"
MUTED = "B9C4CF"
INK = "172533"

FONT_DIR = Path(r"C:\Windows\Fonts")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / ("segoeuib.ttf" if bold else "segoeui.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def make_canvas(section: str, title: str, subtitle: str, page: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), f"#{BG}")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 16), fill=f"#{YELLOW}")
    draw.text((76, 52), section.upper(), font=font(24, True), fill=f"#{YELLOW}")
    draw.text((76, 95), title, font=font(53, True), fill=f"#{WHITE}")
    draw.text((80, 169), subtitle, font=font(25), fill=f"#{MUTED}")
    draw.line((76, 1030, WIDTH - 76, 1030), fill=f"#{PANEL}", width=2)
    draw.text((78, 1042), "NYC TLC HVFHV · JANUARI–JUNI 2026", font=font(16, True), fill=f"#{MUTED}")
    draw.text((WIDTH - 120, 1040), page, font=font(18, True), fill=f"#{YELLOW}")
    return image, draw


def round_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: str, radius: int = 22, outline: str | None = None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=f"#{color}", outline=f"#{outline}" if outline else None, width=width)


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], max_width: int, face: ImageFont.FreeTypeFont, color: str, line_gap: int = 8) -> int:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=face)[2] <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    y = xy[1]
    for line in lines:
        draw.text((xy[0], y), line, font=face, fill=f"#{color}")
        y += face.size + line_gap
    return y


def draw_bullets(draw: ImageDraw.ImageDraw, items: list[str], x: int, y: int, max_width: int, face: ImageFont.FreeTypeFont, color: str, bullet_color: str = YELLOW, gap: int = 15) -> int:
    for item in items:
        draw.ellipse((x, y + 9, x + 13, y + 22), fill=f"#{bullet_color}")
        y = draw_wrapped(draw, item, (x + 28, y), max_width - 28, face, color, 5) + gap
    return y


def card_header(draw: ImageDraw.ImageDraw, x: int, y: int, number: str, title: str) -> None:
    draw.ellipse((x, y, x + 54, y + 54), fill=f"#{YELLOW}")
    draw.text((x + 9, y + 7), number, font=font(24, True), fill=f"#{INK}")
    draw.text((x + 72, y + 7), title, font=font(27, True), fill=f"#{WHITE}")


def create_pipeline_slide(path: Path) -> None:
    image, draw = make_canvas("Hasil implementasi", "Dari data mentah ke analisis", "Alur aktual proyek dan cakupan data yang telah divalidasi", "07")
    cards = [
        ("01", "DATA TLC", ["6 file Parquet, Januari–Juni 2026", "126.771.982 baris awal", "25 atribut perjalanan"]),
        ("02", "KUALITAS & CLEANING", ["126.771.977 baris dipertahankan", "4 timestamp invalid dan 1 duplikat identik dihapus", "Parquet asli tidak ditimpa"]),
        ("03", "INTEGRASI ZONA & CUACA", ["1.120.494 observasi cuaca untuk 258 zona", "126.309.541 perjalanan berpasangan cuaca (99,64%)", "462.436 tanpa pasangan; 85 kasus DST dipertahankan"]),
        ("04", "ANALISIS & APLIKASI", ["Analisis temporal, spasial, dan Graph Analytics", "Random Forest untuk estimasi trip_count", "Dashboard Streamlit untuk eksplorasi hasil"]),
    ]
    x0, y0, w, h, gap = 76, 285, 426, 530, 27
    for index, (number, title, bullets) in enumerate(cards):
        x = x0 + index * (w + gap)
        round_rect(draw, (x, y0, x + w, y0 + h), PANEL, 22, outline="344454", width=2)
        card_header(draw, x + 26, y0 + 30, number, title)
        draw.line((x + 26, y0 + 104, x + w - 26, y0 + 104), fill=f"#{YELLOW}", width=3)
        draw_bullets(draw, bullets, x + 28, y0 + 138, w - 52, font(25), WHITE, gap=22)
        if index < 3:
            draw.polygon([(x + w + 3, y0 + h // 2 - 14), (x + w + 17, y0 + h // 2), (x + w + 3, y0 + h // 2 + 14)], fill=f"#{YELLOW}")
    round_rect(draw, (76, 853, WIDTH - 76, 980), "1B2935", 18)
    draw.text((104, 875), "KONFIGURASI PEMROSESAN", font=font(20, True), fill=f"#{YELLOW}")
    draw_wrapped(draw, "Apache Spark dijalankan lokal dengan local[2]. Hadoop/HDFS belum digunakan sebagai lingkungan eksekusi; proyek ini tidak mengklaim pemrosesan cluster.", (430, 870), 1390, font(23), WHITE, 5)
    image.save(path)


def paste_contained(canvas: Image.Image, source: Path, box: tuple[int, int, int, int], padding: int = 18) -> None:
    panel = Image.open(source).convert("RGB")
    left, top, right, bottom = box
    available = (right - left - padding * 2, bottom - top - padding * 2)
    panel.thumbnail(available, Image.Resampling.LANCZOS)
    x = left + ((right - left) - panel.width) // 2
    y = top + ((bottom - top) - panel.height) // 2
    canvas.paste(panel, (x, y))


def create_graph_slide(path: Path, graph_summary: list[dict[str, str]]) -> None:
    all_row = next(row for row in graph_summary if row["period"] == "ALL_2026-01_2026-06")
    image, draw = make_canvas("Graph Analytics", "Pola arus perjalanan antarzona", "Graf berarah berbobot: pickup zone → drop-off zone; bobot = jumlah perjalanan", "08")
    metrics = [
        (f"{int(all_row['active_nodes']):,}", "zona aktif"),
        (f"{int(all_row['unique_directed_od_pairs']):,}", "pasangan OD unik"),
        (f"{float(all_row['directed_density_excluding_self_loops']):.1%}", "kepadatan (tanpa self-loop)"),
        (f"{int(all_row['self_loop_trip_rows']):,}", "perjalanan dalam zona sama"),
    ]
    x0, y, w, gap = 76, 225, 414, 28
    for i, (value, label) in enumerate(metrics):
        x = x0 + i * (w + gap)
        round_rect(draw, (x, y, x + w, y + 112), PANEL_LIGHT, 18)
        draw.text((x + 25, y + 14), value, font=font(39, True), fill=f"#{INK}")
        draw_wrapped(draw, label, (x + 27, y + 65), w - 50, font(19), "536170", 2)
    left_box = (76, 370, 1065, 865)
    right_box = (1090, 370, 1844, 865)
    round_rect(draw, left_box, PANEL_LIGHT, 18)
    round_rect(draw, right_box, PANEL_LIGHT, 18)
    paste_contained(image, GRAPH_IMAGES / "monthly_graph_structure.png", left_box, 18)
    paste_contained(image, GRAPH_IMAGES / "top_od_flows_jan_jun.png", right_box, 16)
    round_rect(draw, (76, 890, WIDTH - 76, 990), "1B2935", 18)
    draw_wrapped(draw, "Pasangan OD unik agregat: 64.701. Satu weak component berarti seluruh zona terhubung jika arah diabaikan. Hasil ini menjelaskan pola pergerakan, bukan dampak cuaca atau rute jalan yang dilalui.", (101, 908), WIDTH - 205, font(22), WHITE, 5)
    image.save(path)


def create_ml_slide(path: Path, metrics: list[dict[str, str]]) -> None:
    def metric(implementation: str, split: str, col: str) -> float:
        return float(next(row[col] for row in metrics if row["implementation"] == implementation and row["split"] == split))

    spark = "Controlled Spark MLlib"
    sklearn = "Controlled scikit-learn"
    image, draw = make_canvas("Machine Learning", "Estimasi volume perjalanan per zona-jam", "Random Forest Regression · evaluasi kronologis dan controlled comparison", "09")
    round_rect(draw, (76, 225, 900, 420), PANEL, 22, outline="344454", width=2)
    draw.text((108, 250), "UNIT & SPLIT DATA", font=font(24, True), fill=f"#{YELLOW}")
    draw_bullets(draw, ["Target: trip_count pada PULocationID × jam lokal", "Train Jan–Apr · validation Mei · test Juni", "Fitur waktu, zona pickup, dan cuaca aktual"], 112, 295, 740, font(22), WHITE, gap=5)

    round_rect(draw, (930, 225, 1844, 420), PANEL, 22, outline="344454", width=2)
    draw.text((962, 250), "PARAMETER UTAMA", font=font(24, True), fill=f"#{YELLOW}")
    draw_wrapped(draw, "50 trees · depth 8 · seed 42 · one-hot untuk zona dan weather_code · 281 kolom fitur", (965, 300), 825, font(24), WHITE, 8)

    draw.text((76, 466), "HASIL TESTING JUNI 2026", font=font(27, True), fill=f"#{WHITE}")
    columns = ["Model", "MAE ↓", "RMSE ↓", "R² ↑", "Waktu fit"]
    table_x = [76, 722, 1000, 1280, 1550]
    header_y, row_h = 520, 100
    round_rect(draw, (76, header_y, 1844, header_y + 65), YELLOW, 12)
    for i, label in enumerate(columns):
        draw.text((table_x[i] + 15, header_y + 16), label, font=font(22, True), fill=f"#{INK}")
    rows = [
        (spark, metric(spark, "testing", "MAE"), metric(spark, "testing", "RMSE"), metric(spark, "testing", "R2"), 113.05),
        (sklearn, metric(sklearn, "testing", "MAE"), metric(sklearn, "testing", "RMSE"), metric(sklearn, "testing", "R2"), 51.60),
    ]
    for idx, values in enumerate(rows):
        y = header_y + 79 + idx * row_h
        round_rect(draw, (76, y, 1844, y + 82), PANEL, 12)
        formatted = [values[0], f"{values[1]:.2f}", f"{values[2]:.2f}", f"{values[3]:.3f}", f"{values[4]:.1f} dtk"]
        for i, value in enumerate(formatted):
            draw.text((table_x[i] + 15, y + 20), value, font=font(22 if i == 0 else 25, i > 0), fill=f"#{WHITE}")

    round_rect(draw, (76, 830, 1844, 986), "1B2935", 18)
    draw.text((103, 849), "INSIGHT & BATAS INTERPRETASI", font=font(20, True), fill=f"#{YELLOW}")
    draw_wrapped(draw, "Kedua hasil controlled berdekatan. Lokasi pickup dan jam lokal paling dominan; importance fitur cuaca relatif lebih kecil. Model memakai cuaca aktual pada jam target, sehingga ini estimasi retrospektif—bukan forecasting operasional. Importance bukan hubungan sebab-akibat.", (103, 886), 1700, font(21), WHITE, 5)
    image.save(path)


def create_conclusion_slide(path: Path, weather: dict[str, str], graph_summary: list[dict[str, str]]) -> None:
    all_row = next(row for row in graph_summary if row["period"] == "ALL_2026-01_2026-06")
    available = int(weather["weather_available_trip_rows"])
    total = int(weather["source_trip_rows"])
    image, draw = make_canvas("Rangkuman", "Kesimpulan dan arah pengembangan", "Temuan utama, batas interpretasi, dan langkah berikutnya", "10")
    columns = [
        (76, "TEMUAN", [
            f"Cuaca tersedia untuk {available:,} dari {total:,} perjalanan ({available / total:.2%}).",
            f"Graf menghubungkan {int(all_row['active_nodes'])} zona dan {int(all_row['unique_directed_od_pairs']):,} pasangan OD unik.",
            "Model mengidentifikasi pola volume berdasarkan lokasi, waktu, dan fitur cuaca.",
        ]),
        (676, "KETERBATASAN", [
            "Cakupan analisis hanya Januari–Juni 2026.",
            "Spark dijalankan secara lokal (local[2]); HDFS belum diuji.",
            "Analisis cuaca bersifat deskriptif; model memakai cuaca aktual pada jam target.",
        ]),
        (1276, "LANGKAH BERIKUTNYA", [
            "Untuk forecasting, gunakan lag perjalanan dan prakiraan cuaca yang tersedia sebelum jam prediksi.",
            "Uji data pada periode lebih panjang dan lakukan validasi temporal berulang.",
            "Benchmark cluster Hadoop/Spark jika infrastruktur tersedia.",
        ]),
    ]
    for x, title, bullets in columns:
        round_rect(draw, (x, 270, x + 568, 875), PANEL, 22, outline="344454", width=2)
        draw.rectangle((x + 28, 305, x + 95, 313), fill=f"#{YELLOW}")
        draw.text((x + 28, 335), title, font=font(25, True), fill=f"#{WHITE}")
        draw_bullets(draw, bullets, x + 30, 405, 510, font(23), WHITE, gap=23)
    round_rect(draw, (76, 915, WIDTH - 76, 990), "1B2935", 16)
    draw_wrapped(draw, "Kesimpulan penelitian dibatasi pada pola yang teramati dan konfigurasi eksperimen yang diuji; hasil tidak ditafsirkan sebagai bukti kausal atau keunggulan umum suatu library.", (102, 935), WIDTH - 205, font(20), MUTED, 4)
    image.save(path)


def slide_xml(image_rid: str, name: str) -> bytes:
    xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
 <p:cSld><p:spTree>
  <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
  <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
  <p:pic><p:nvPicPr><p:cNvPr id="2" name="{name}" descr="Slide visual lanjutan proyek HVFHV"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
   <p:blipFill><a:blip r:embed="{image_rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
   <p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{EMU_W}" cy="{EMU_H}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
  </p:pic>
 </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''
    return xml.encode("utf-8")


def add_slides_to_pptx(source: Path, destination: Path, slide_images: list[Path]) -> None:
    ns_p = "http://schemas.openxmlformats.org/presentationml/2006/main"
    ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ns_rel = "http://schemas.openxmlformats.org/package/2006/relationships"
    ns_ct = "http://schemas.openxmlformats.org/package/2006/content-types"
    ET.register_namespace("p", ns_p)
    ET.register_namespace("r", ns_r)
    ET.register_namespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
    with zipfile.ZipFile(source, "r") as zin:
        parts = {name: zin.read(name) for name in zin.namelist()}

    pres = ET.fromstring(parts["ppt/presentation.xml"])
    rels = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    sld_list = pres.find(f"{{{ns_p}}}sldIdLst")
    if sld_list is None:
        raise ValueError("Presentation slide list is missing")
    slide_rels = [rel for rel in rels if rel.attrib.get("Type", "").endswith("/slide")]
    existing_nums = [int(m.group(1)) for name in parts for m in [re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)] if m]
    start_slide = max(existing_nums, default=0) + 1
    rid_nums = [int(m.group(1)) for rel in rels for m in [re.fullmatch(r"rId(\d+)", rel.attrib.get("Id", ""))] if m]
    next_rid = max(rid_nums, default=0) + 1
    ids = [int(item.attrib["id"]) for item in sld_list]
    next_sld_id = max(ids, default=255) + 1
    media_nums = [int(m.group(1)) for item in parts for m in [re.fullmatch(r"ppt/media/image(\d+)\.png", item)] if m]
    next_media_num = max(media_nums, default=0) + 1
    content_types = ET.fromstring(parts["[Content_Types].xml"])

    insert_index = max(0, len(sld_list) - 1)  # keep original closing slide last
    for index, image_path in enumerate(slide_images):
        slide_no = start_slide + index
        rid = f"rId{next_rid + index}"
        slide_id = str(next_sld_id + index)
        image_name = f"image{next_media_num + index}.png"
        image_target = f"ppt/media/{image_name}"
        slide_name = f"ppt/slides/slide{slide_no}.xml"
        rel_name = f"ppt/slides/_rels/slide{slide_no}.xml.rels"

        new_id = ET.Element(f"{{{ns_p}}}sldId", {"id": slide_id, f"{{{ns_r}}}id": rid})
        sld_list.insert(insert_index + index, new_id)
        ET.SubElement(rels, f"{{{ns_rel}}}Relationship", {
            "Id": rid,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
            "Target": f"slides/slide{slide_no}.xml",
        })
        ET.SubElement(content_types, f"{{{ns_ct}}}Override", {
            "PartName": f"/{slide_name}",
            "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
        })
        parts[slide_name] = slide_xml("rId2", f"Project results slide {index + 1}")
        slide_rels = ET.Element(f"{{{ns_rel}}}Relationships")
        ET.SubElement(slide_rels, f"{{{ns_rel}}}Relationship", {
            "Id": "rId1",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout",
            "Target": "../slideLayouts/slideLayout1.xml",
        })
        ET.SubElement(slide_rels, f"{{{ns_rel}}}Relationship", {
            "Id": "rId2",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            "Target": f"../media/{image_name}",
        })
        parts[rel_name] = ET.tostring(slide_rels, encoding="utf-8", xml_declaration=True)
        parts[image_target] = image_path.read_bytes()

    parts["ppt/presentation.xml"] = ET.tostring(pres, encoding="utf-8", xml_declaration=True)
    parts["ppt/_rels/presentation.xml.rels"] = ET.tostring(rels, encoding="utf-8", xml_declaration=True)
    parts["[Content_Types].xml"] = ET.tostring(content_types, encoding="utf-8", xml_declaration=True)

    app_xml = parts.get("docProps/app.xml")
    if app_xml:
        try:
            app = ET.fromstring(app_xml)
            for element in app.iter():
                if element.tag.endswith("}Slides") or element.tag == "Slides":
                    element.text = str(len(sld_list))
            parts["docProps/app.xml"] = ET.tostring(app, encoding="utf-8", xml_declaration=True)
        except ET.ParseError:
            pass

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for name, content in parts.items():
            zout.writestr(name, content)


def validate_deck(path: Path, expected_slides: int) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member in generated PPTX: {bad}")
        for name in archive.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(archive.read(name))
        root = ET.fromstring(archive.read("ppt/presentation.xml"))
        slide_list = root.find("{http://schemas.openxmlformats.org/presentationml/2006/main}sldIdLst")
        count = len(slide_list) if slide_list is not None else 0
        if count != expected_slides:
            raise ValueError(f"Slide count mismatch: expected {expected_slides}, got {count}")


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"PowerPoint source not found: {SOURCE}")
    if not GRAPH_SUMMARY.is_file() or not ML_METRICS.is_file() or not WEATHER_SUMMARY.is_file():
        raise FileNotFoundError("Required validated Graph Analytics, ML, or weather report is missing")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing presentation: {OUTPUT}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    graph_summary = read_csv(GRAPH_SUMMARY)
    metrics = read_csv(ML_METRICS)
    weather_rows = read_csv(WEATHER_SUMMARY)
    weather = {row["metric"]: row["value"] for row in weather_rows}
    slide_images = [
        ASSET_DIR / "07_pipeline_and_coverage.png",
        ASSET_DIR / "08_graph_analytics_results.png",
        ASSET_DIR / "09_machine_learning_results.png",
        ASSET_DIR / "10_conclusions_and_limitations.png",
    ]
    create_pipeline_slide(slide_images[0])
    create_graph_slide(slide_images[1], graph_summary)
    create_ml_slide(slide_images[2], metrics)
    create_conclusion_slide(slide_images[3], weather, graph_summary)
    add_slides_to_pptx(SOURCE, OUTPUT, slide_images)
    validate_deck(OUTPUT, expected_slides=7 + len(slide_images))

    readme = OUTPUT_DIR / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Presentasi proyek HVFHV\n\n"
            "`Big Data - continued.pptx` mempertahankan tujuh slide sumber dari Downloads dan menambahkan empat slide hasil proyek sebelum slide Thank You. "
            "Empat slide baru memakai visual PNG penuh agar gaya desain konsisten; teks slide tambahan tidak dapat diedit per objek di PowerPoint. "
            "Sumber hasil aktual diambil dari laporan Graph Analytics, controlled ML, dan validasi cuaca yang tercantum dalam script pembuat.\n",
            encoding="utf-8",
        )
    print(f"Created: {OUTPUT}")
    print(f"Slides: 7 source + {len(slide_images)} added = {7 + len(slide_images)}")
    print("New slides: pipeline/coverage, graph analytics, machine learning, conclusion/limitations")
    print("Validation: ZIP integrity, all XML parts parse, slide count passed")


if __name__ == "__main__":
    main()
