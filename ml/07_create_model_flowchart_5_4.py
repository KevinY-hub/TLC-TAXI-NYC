"""Create a concise, paper-ready flowchart for Chapter 5.4."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "machine_learning" / "visualizations" / "chapter_v" / "flowchart_model_rf_5_4_v2.png"
W, H = 1800, 1125

BG = "FFFFFF"
INK = "172B3A"
MUTED = "5D6B78"
AMBER = "F5AE32"
TEAL = "38B9A9"
TEAL_PALE = "E8F7F4"
BLUE = "4A82C3"
BLUE_PALE = "EEF4FB"
GRAY = "929BA5"
GRAY_PALE = "F4F5F6"
LINE = "596875"

FONT_DIR = Path(r"C:\Windows\Fonts")


def fnt(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    file = FONT_DIR / ("segoeuib.ttf" if bold else "segoeui.ttf")
    return ImageFont.truetype(str(file), size) if file.exists() else ImageFont.load_default()


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, face: ImageFont.FreeTypeFont, color: str = INK, spacing: int = 7) -> None:
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    sizes = [draw.textbbox((0, 0), line, font=face) for line in lines]
    heights = [b[3] - b[1] for b in sizes]
    total = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total) / 2
    for line, bounds, height in zip(lines, sizes, heights):
        width = bounds[2] - bounds[0]
        draw.text((x1 + (x2 - x1 - width) / 2, y), line, font=face, fill=f"#{color}")
        y += height + spacing


def box(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], fill: str, outline: str, text: str, size: int = 25, bold: bool = False, radius: int = 18) -> None:
    draw.rounded_rectangle(rect, radius=radius, fill=f"#{fill}", outline=f"#{outline}", width=3)
    centered_text(draw, rect, text, fnt(size, bold))


def arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str = LINE, width: int = 5) -> None:
    draw.line(points, fill=f"#{color}", width=width, joint="curve")
    x, y = points[-1]
    draw.polygon([(x, y), (x - 12, y - 20), (x + 12, y - 20)], fill=f"#{color}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (W, H), f"#{BG}")
    draw = ImageDraw.Draw(image)

    draw.text((82, 45), "5.4  PROSES PENGEMBANGAN MODEL", font=fnt(27, True), fill=f"#{BLUE}")
    draw.text((82, 88), "Random Forest Regression: dua implementasi", font=fnt(48, True), fill=f"#{INK}")
    draw.text((84, 151), "Unit observasi: PULocationID × jam pickup lokal  |  Target: trip_count", font=fnt(25), fill=f"#{MUTED}")

    box(draw, (360, 225, 1440, 345), "FFF2D8", AMBER,
        "DATASET ZONA–JAM + FITUR WAKTU, LOKASI, DAN CUACA\nTrain Jan–Apr  ·  Validation Mei  ·  Test Juni", 25, True)

    # Branch connector from the common chronological split to both implementations.
    draw.line([(900, 345), (900, 390), (450, 390), (450, 425)], fill=f"#{LINE}", width=5)
    draw.line([(900, 390), (1350, 390), (1350, 425)], fill=f"#{LINE}", width=5)
    arrow(draw, [(450, 390), (450, 425)])
    arrow(draw, [(1350, 390), (1350, 425)])

    left_lane = (70, 430, 875, 925)
    right_lane = (925, 430, 1730, 925)
    draw.rounded_rectangle(left_lane, radius=22, fill=f"#{TEAL_PALE}", outline=f"#{TEAL}", width=4)
    draw.rounded_rectangle(right_lane, radius=22, fill=f"#{BLUE_PALE}", outline=f"#{BLUE}", width=4)
    centered_text(draw, (95, 445, 850, 500), "SCIKIT-LEARN  ·  TANPA SPARK MLlib", fnt(27, True), INK)
    centered_text(draw, (950, 445, 1705, 500), "APACHE SPARK MLlib", fnt(27, True), INK)

    box(draw, (160, 520, 785, 650), "FFFFFF", TEAL,
        "Preprocessing fit pada training\nOne-hot kategori · 281 fitur", 23, True)
    box(draw, (160, 710, 785, 845), "FFFFFF", TEAL,
        "RandomForestRegressor\n50 trees · depth 8 · seed 42", 24, True)
    arrow(draw, [(472, 650), (472, 710)], TEAL)

    box(draw, (1015, 520, 1640, 650), "FFFFFF", BLUE,
        "Spark DataFrame + preprocessing\nKategori dari training · 281 fitur", 23, True)
    box(draw, (1015, 710, 1640, 845), "FFFFFF", BLUE,
        "RandomForestRegressor\n50 trees · maxDepth 8 · seed 42", 24, True)
    arrow(draw, [(1327, 650), (1327, 710)], BLUE)

    # Converge both branches into the shared evaluation stage.
    draw.line([(472, 845), (472, 885), (900, 885), (900, 950)], fill=f"#{LINE}", width=5)
    draw.line([(1327, 845), (1327, 885), (900, 885)], fill=f"#{LINE}", width=5)
    arrow(draw, [(900, 885), (900, 950)])
    box(draw, (275, 955, 1525, 1070), "F1F3F5", "7C8995",
        "EVALUASI VALIDATION & TEST  ·  MAE  ·  RMSE  ·  R²\nBandingkan prediksi dan feature importance", 24, True)
    draw.text((84, 1090), "Cuaca aktual pada jam target digunakan: hasil merupakan estimasi retrospektif, bukan forecasting operasional.", font=fnt(18), fill=f"#{MUTED}")

    image.save(OUTPUT, dpi=(300, 300))
    print(f"Created: {OUTPUT}")
    print(f"Dimensions: {W} x {H} px at 300 DPI")


if __name__ == "__main__":
    main()
