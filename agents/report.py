import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle


def _clean_table_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes empty/None column names and unnamed index issues before PDF export.
    """
    df = df.copy()

    # Make index name safe before reset
    if df.index.name is None:
        df.index.name = "index"

    # Replace None/empty column labels
    df.columns = [
        ("Unnamed" if (c is None or str(c).strip() == "") else str(c))
        for c in df.columns
    ]

    return df

def _render_markdown_like_text(text: str, styles):
    """
    Converts simple markdown headings into real PDF headings.
    Supports:
      # Heading1
      ## Heading2
      ### Heading3
    """
    story = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.12 * inch))
            continue

        if line.startswith("### "):
            story.append(Paragraph(f"<b>{line[4:]}</b>", styles["Heading3"]))
        elif line.startswith("## "):
            story.append(Paragraph(f"<b>{line[3:]}</b>", styles["Heading2"]))
        elif line.startswith("# "):
            story.append(Paragraph(f"<b>{line[2:]}</b>", styles["Heading1"]))
        else:
            # bullet support
            if line.startswith(("-", "*")):
                story.append(Paragraph(f"• {line[1:].strip()}", styles["Normal"]))
            else:
                story.append(Paragraph(line, styles["Normal"]))

    return story


def _save_figures(figures, output_dir="reports/charts"):
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for i, fig in enumerate(figures):
        path = f"{output_dir}/chart_{i}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

def _df_to_reportlab_table(df, max_rows=25, font_size=8, page_width=None):
    """
    Convert DataFrame to ReportLab Table fitted inside page margins.
    """

    df = df.copy()
    df.reset_index(drop=True, inplace=True)

    if len(df) > max_rows:
        df = df.head(max_rows)

    df = df.fillna("").astype(str)

    data = [df.columns.tolist()] + df.values.tolist()

    # ✅ page width inside margins
    if page_width is None:
        page_width = A4[0] - 80   # 40 left + 40 right margin

    n_cols = len(df.columns)

    # ✅ equal widths (simple + stable)
    col_width = page_width / max(1, n_cols)
    col_widths = [col_width] * n_cols

    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")

    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9d9d9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightcyan]),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ])

    table.setStyle(style)
    return table



def generate_pdf(
    insights,
    llm_text,
    assumptions=None,
    charts=None,
    eda_tables=None,
    output_path="reports/EDA_Report.pdf"
):
    os.makedirs("reports", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

    # ---------------- TITLE ----------------
    story.append(Paragraph("<b>Exploratory Data Analysis (EDA) Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"<i>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # ---------------- CLEANING ----------------
    story.append(Paragraph("<b>Data Loading and Cleaning</b>", styles["Heading2"]))
    for i in insights:
        story.append(Paragraph(f"- {i}", styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    # ---------------- ASSUMPTIONS ----------------
    if assumptions:
        story.append(Paragraph("<b>EDA Assumptions</b>", styles["Heading2"]))
        for a in assumptions:
            story.append(Paragraph(f"- {a}", styles["Normal"]))
        story.append(Spacer(1, 0.25 * inch))

    # ---------------- EDA TABLES ----------------
    if eda_tables:
        story.append(Paragraph("<b>Data Summary and Descriptive Statistics</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.15 * inch))

        if "missing_table" in eda_tables and not eda_tables["missing_table"].empty:
            story.append(Paragraph("<b>Missing Values</b>", styles["Heading3"]))

            df_missing = eda_tables["missing_table"].copy()
            df_missing.index.name = "column"
            df_missing = df_missing.reset_index()
            df_missing = _clean_table_headers(df_missing)

            story.append(_df_to_reportlab_table(df_missing))
            story.append(Spacer(1, 0.25 * inch))

        if "dtypes_table" in eda_tables and not eda_tables["dtypes_table"].empty:
            story.append(Paragraph("<b>Column Types</b>", styles["Heading3"]))

            df_types = _clean_table_headers(eda_tables["dtypes_table"])
            story.append(_df_to_reportlab_table(df_types))
            story.append(Spacer(1, 0.25 * inch))

        if "numeric_summary_table" in eda_tables and "numeric_summary_table" in eda_tables:
            story.append(Paragraph("<b>Summary Statistics (Numeric)</b>", styles["Heading3"]))

            df_num = eda_tables["numeric_summary_table"].copy()
            df_num.index.name = "feature"
            df_num = df_num.reset_index()
            df_num = _clean_table_headers(df_num)

            story.append(_df_to_reportlab_table(df_num))
            story.append(Spacer(1, 0.25 * inch))

        if "correlation_table" in eda_tables and not eda_tables["correlation_table"].empty:
            story.append(Paragraph("<b>Correlation Matrix</b>", styles["Heading2"]))
            story.append(Spacer(1, 0.2 * inch))

            corr_df = eda_tables["correlation_table"].copy()

            # round for readability
            corr_df = corr_df.round(2)

            # keep matrix small enough for PDF
            corr_df = corr_df.iloc[:10, :10]

            corr_df = corr_df.reset_index().rename(columns={"index": "feature"})
            corr_df = _clean_table_headers(corr_df)

            story.append(_df_to_reportlab_table(corr_df))
            story.append(Spacer(1, 0.25 * inch))


        if "top_correlations_table" in eda_tables and not eda_tables["top_correlations_table"].empty:
            story.append(Paragraph("<b>Top Correlations</b>", styles["Heading3"]))

            df_corr = _clean_table_headers(eda_tables["top_correlations_table"])
            story.append(_df_to_reportlab_table(df_corr))
            story.append(Spacer(1, 0.25 * inch))


    # ---------------- VISUALS ----------------
    if charts:
        story.append(Paragraph("<b>Visual Analysis</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.15 * inch))

        chart_paths = _save_figures(charts)

        for path in chart_paths:
            story.append(Image(path, width=5.5 * inch, height=3.5 * inch))
            story.append(Spacer(1, 0.25 * inch))

    # ---------------- AI REPORT TEXT ----------------
    if llm_text:
        story.append(Paragraph("<b>4. AI Narrative Insights</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.15 * inch))

        # render nicely (no markdown ## in pdf)
        story.extend(_render_markdown_like_text(llm_text, styles))


    doc.build(story)
    return output_path
