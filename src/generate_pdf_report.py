"""Generate PDF reports of train samples grouped by State.

Usage:
    uv run python src/generate_pdf_report.py
"""

import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
import rootutils
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
)
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Setup root directory
ROOT_DIR = rootutils.setup_root(".", indicator="pyproject.toml", cwd=True, dotenv=True)

# Paths
INPUT_DIR = ROOT_DIR / "data" / "input" / "csiro-biomass"
OUTPUT_DIR = ROOT_DIR / "data" / "output" / "pdf"
TRAIN_CSV = INPUT_DIR / "train.csv"

# Target columns
TARGET_COLS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]


def convert_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Long format CSV to Wide format (1 row per image).

    Long format (train.csv):
        sample_id, image_path, ..., target_name, target

    Wide format:
        image_id, image_path, ..., Dry_Total_g, GDM_g, Dry_Green_g, ...

    Args:
        df: DataFrame in Long format

    Returns:
        DataFrame in Wide format
    """
    df = df.copy()
    df["image_id"] = df["sample_id"].str.split("__").str[0]

    if "target" not in df.columns:
        df_wide = df[["image_id", "image_path"]].drop_duplicates().reset_index(drop=True)
        return df_wide

    meta_cols = [c for c in df.columns if c not in ["sample_id", "target_name", "target", "image_id"]]

    df_wide = df.pivot_table(
        index=["image_id"] + meta_cols,
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()

    df_wide.columns.name = None

    return df_wide


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load train.csv and convert to wide format."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df_wide = convert_long_to_wide(df)
    logger.info(f"Loaded {len(df_wide)} samples")
    return df_wide


def get_image_for_reportlab(image_path: Path, max_width: float, max_height: float) -> RLImage:
    """Load image at half resolution and prepare for reportlab.

    Args:
        image_path: Path to the image file
        max_width: Maximum width in points
        max_height: Maximum height in points

    Returns:
        reportlab Image object
    """
    img = Image.open(image_path)

    # Resize to half resolution
    new_size = (img.width // 2, img.height // 2)
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

    # Calculate scale to fit within bounds while maintaining aspect ratio
    aspect = img_resized.width / img_resized.height
    if max_width / aspect <= max_height:
        width = max_width
        height = max_width / aspect
    else:
        height = max_height
        width = max_height * aspect

    # Save to BytesIO buffer
    buffer = BytesIO()
    img_resized.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return RLImage(buffer, width=width, height=height)


def create_metadata_table(row: pd.Series, styles: dict) -> Table:
    """Create formatted metadata table for a sample.

    Args:
        row: DataFrame row with sample data
        styles: reportlab styles

    Returns:
        reportlab Table object
    """

    # Format values safely
    def fmt_float(val, decimals=4):
        try:
            return f"{float(val):.{decimals}f}"
        except (ValueError, TypeError):
            return str(val)

    data = [
        ["Image ID", str(row["image_id"])],
        ["Sampling Date", str(row["Sampling_Date"])],
        ["State", str(row["State"])],
        ["Species", str(row["Species"])],
        ["NDVI", fmt_float(row["Pre_GSHH_NDVI"], 4)],
        ["Height (cm)", fmt_float(row["Height_Ave_cm"], 2)],
        ["", ""],
        ["Dry_Clover_g", fmt_float(row.get("Dry_Clover_g", "N/A"), 4)],
        ["Dry_Dead_g", fmt_float(row.get("Dry_Dead_g", "N/A"), 4)],
        ["Dry_Green_g", fmt_float(row.get("Dry_Green_g", "N/A"), 4)],
        ["Dry_Total_g", fmt_float(row.get("Dry_Total_g", "N/A"), 4)],
        ["GDM_g", fmt_float(row.get("GDM_g", "N/A"), 4)],
    ]

    table = Table(data, colWidths=[28 * mm, 27 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                ("ALIGN", (1, 0), (1, -1), "LEFT"),
                ("LINEBELOW", (0, 5), (-1, 5), 0.5, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return table


def create_sample_block(
    row: pd.Series,
    image_dir: Path,
    styles: dict,
    available_width: float,
) -> list:
    """Create a sample block with image and metadata side by side.

    Layout: [Image (left)] [Metadata Table (right)]

    Args:
        row: DataFrame row with sample data
        image_dir: Directory containing images
        styles: reportlab styles
        available_width: Available width for content

    Returns:
        List of reportlab flowables
    """
    # Layout dimensions
    metadata_width = 55 * mm
    image_width = available_width - metadata_width - 5 * mm  # 5mm gap
    image_height = 60 * mm  # Aspect ratio 2:1

    # Create metadata table (left side)
    metadata_table = create_metadata_table(row, styles)

    # Create image (right side)
    image_path = image_dir / row["image_path"]

    if image_path.exists():
        try:
            img = get_image_for_reportlab(image_path, image_width, image_height)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            img = Paragraph("[Image not available]", styles["Normal"])
    else:
        logger.warning(f"Image not found: {image_path}")
        img = Paragraph("[Image not found]", styles["Normal"])

    # Create outer table to place image and metadata side by side
    outer_table = Table(
        [[img, metadata_table]],
        colWidths=[image_width, metadata_width],
    )
    outer_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    return [outer_table, Spacer(1, 8 * mm)]


def generate_pdf_for_state(
    df: pd.DataFrame,
    state: str,
    output_path: Path,
    image_dir: Path,
) -> None:
    """Generate PDF for a single state.

    Args:
        df: DataFrame with samples for this state (already sorted by date)
        state: State name
        output_path: Output PDF path
        image_dir: Directory containing images
    """
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    elements = []

    # Calculate available width
    page_width, page_height = A4
    available_width = page_width - 30 * mm  # 15mm margins on each side

    # Title
    title_style = styles["Title"]
    elements.append(Paragraph(f"Train Samples - {state}", title_style))
    elements.append(Paragraph(f"Total: {len(df)} samples", styles["Normal"]))
    elements.append(Spacer(1, 10 * mm))

    # Process samples 3 at a time
    samples_list = [row for _, row in df.iterrows()]

    for i in range(0, len(samples_list), 3):
        batch = samples_list[i : i + 3]

        for row in batch:
            block = create_sample_block(row, image_dir, styles, available_width)
            elements.extend(block)

        # Page break if more samples remain
        if i + 3 < len(samples_list):
            elements.append(PageBreak())

    doc.build(elements)
    logger.info(f"Generated: {output_path} ({len(df)} samples, {(len(df) + 2) // 3} pages)")


def main() -> None:
    """Main entry point."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Load data
    df = load_and_prepare_data(TRAIN_CSV)

    # Parse and sort by Sampling_Date
    df["Sampling_Date_parsed"] = pd.to_datetime(df["Sampling_Date"], format="%Y/%m/%d")
    df = df.sort_values("Sampling_Date_parsed")

    # Get unique states
    states = df["State"].unique()
    logger.info(f"Found {len(states)} states: {list(states)}")

    # Generate PDF for each state
    for state in states:
        df_state = df[df["State"] == state].reset_index(drop=True)
        output_path = OUTPUT_DIR / f"train_samples_{state}.pdf"

        logger.info(f"Generating PDF for {state} ({len(df_state)} samples)...")
        generate_pdf_for_state(
            df=df_state,
            state=state,
            output_path=output_path,
            image_dir=INPUT_DIR,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
