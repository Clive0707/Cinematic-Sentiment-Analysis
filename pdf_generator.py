"""
PDF Report Generator for MarathiSense AI.

Generates a professional, structured PDF report containing dual model predictions,
confidence scores, model agreement status, top 3 similar dataset reviews,
model descriptions, and evaluation metrics.
"""

from datetime import datetime
import io
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Register Devanagari font for Marathi text rendering
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

NIR_PATH = "C:/Windows/Fonts/Nirmala.ttc"
if os.path.exists(NIR_PATH):
    try:
        pdfmetrics.registerFont(TTFont("Nirmala", NIR_PATH, subfontIndex=0))
        pdfmetrics.registerFont(TTFont("Nirmala-Bold", NIR_PATH, subfontIndex=1))
        FONT_REGULAR = "Nirmala"
        FONT_BOLD = "Nirmala-Bold"
    except Exception as e:
        print(f"Warning: Could not register Nirmala font: {e}")


def generate_sentiment_pdf(result: dict) -> bytes:
    """
    Generate a binary PDF report for a given sentiment prediction result.

    Args:
        result: Dictionary returned by predict_sentiment().

    Returns:
        Bytes object containing the generated PDF file.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName=FONT_BOLD,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName=FONT_BOLD,
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )

    review_text_style = ParagraphStyle(
        "ReviewText",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=9,
        leading=11,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # Document Header
    story.append(Paragraph("MARATHI MOVIE REVIEW SENTIMENT ANALYSIS", title_style))
    story.append(
        Paragraph(
            f"Sentiment Analysis Report &bull; Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#6366f1"), spaceAfter=14))

    # 1. User Review Section
    story.append(Paragraph("1. User Review", section_heading))
    review_content = result.get("review_text", "")
    review_table_data = [
        [Paragraph(f"<b>Original Review (Marathi):</b><br/>{review_content}", review_text_style)]
    ]
    review_table = Table(review_table_data, colWidths=[540])
    review_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(review_table)
    story.append(Spacer(1, 10))

    # 2 & 3. Model Results Section
    nb = result.get("naive_bayes", {})
    lr = result.get("logistic_regression", {})

    story.append(Paragraph("2. Supervised Model Predictions", section_heading))

    nb_sent = nb.get("sentiment", "N/A")
    nb_conf = nb.get("confidence", 0.0)
    nb_color = "#16a34a" if nb_sent == "Positive" else "#dc2626"

    lr_sent = lr.get("sentiment", "N/A")
    lr_conf = lr.get("confidence", 0.0)
    lr_color = "#16a34a" if lr_sent == "Positive" else "#dc2626"

    model_table_data = [
        [
            Paragraph("Algorithm", table_header_style),
            Paragraph("Role", table_header_style),
            Paragraph("Prediction", table_header_style),
            Paragraph("Confidence Score", table_header_style),
        ],
        [
            Paragraph("<b>Multinomial Naive Bayes</b>", table_cell_bold),
            Paragraph("Primary Classifier", table_cell_style),
            Paragraph(f"<font color='{nb_color}'><b>{nb_sent}</b></font>", table_cell_bold),
            Paragraph(f"<b>{nb_conf:.1f}%</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Logistic Regression</b>", table_cell_bold),
            Paragraph("Secondary Classifier", table_cell_style),
            Paragraph(f"<font color='{lr_color}'><b>{lr_sent}</b></font>", table_cell_bold),
            Paragraph(f"<b>{lr_conf:.1f}%</b>", table_cell_bold),
        ],
    ]
    model_table = Table(model_table_data, colWidths=[160, 120, 130, 130])
    model_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#ffffff")),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(model_table)
    story.append(Spacer(1, 10))

    # 4. Model Agreement Section
    story.append(Paragraph("3. Model Agreement Analysis", section_heading))
    agreement = result.get("agreement", {})
    is_agree = agreement.get("is_agree", True)
    status_label = agreement.get("status_label", "N/A")
    msg = agreement.get("message", "")

    bg_color = "#f0fdf4" if is_agree else "#fffbeb"
    border_color = "#86efac" if is_agree else "#fde68a"
    text_color = "#15803d" if is_agree else "#b45309"

    agree_html = f"<font color='{text_color}'><b>{status_label}</b> &bull; {msg}</font>"
    agree_table_data = [[Paragraph(agree_html, body_style)]]
    agree_table = Table(agree_table_data, colWidths=[540])
    agree_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_color)),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(border_color)),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(agree_table)
    story.append(Spacer(1, 10))

    # 5. Top 3 Similar Reviews (Cosine Similarity)
    story.append(Paragraph("4. Top 3 Similar Reviews (TF-IDF Cosine Similarity)", section_heading))
    sim_reviews = result.get("similar_reviews", [])

    sim_table_data = [
        [
            Paragraph("Rank", table_header_style),
            Paragraph("Dataset Sentiment", table_header_style),
            Paragraph("Similarity %", table_header_style),
            Paragraph("Dataset Review Text", table_header_style),
        ]
    ]

    for idx, item in enumerate(sim_reviews, start=1):
        sent = item.get("sentiment", "N/A")
        sim_val = item.get("similarity", 0.0)
        text = item.get("review", "")
        scolor = "#16a34a" if sent == "Positive" else "#dc2626"

        sim_table_data.append(
            [
                Paragraph(f"#{idx}", table_cell_bold),
                Paragraph(f"<font color='{scolor}'><b>{sent}</b></font>", table_cell_style),
                Paragraph(f"<b>{sim_val:.1f}%</b>", table_cell_style),
                Paragraph(text, table_cell_style),
            ]
        )

    sim_table = Table(sim_table_data, colWidths=[40, 110, 80, 310])
    sim_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(sim_table)
    story.append(Spacer(1, 10))

    # 6. Model Architecture Overview
    story.append(Paragraph("5. System Architecture Overview", section_heading))
    arch_data = [
        [
            Paragraph("<b>Multinomial Naive Bayes:</b>", table_cell_bold),
            Paragraph("Primary supervised classification algorithm.", table_cell_style),
        ],
        [
            Paragraph("<b>Logistic Regression:</b>", table_cell_bold),
            Paragraph("Secondary supervised classification algorithm for validation.", table_cell_style),
        ],
        [
            Paragraph("<b>TF-IDF Vectorizer:</b>", table_cell_bold),
            Paragraph("Converts Marathi text into numerical feature representations.", table_cell_style),
        ],
        [
            Paragraph("<b>Cosine Similarity:</b>", table_cell_bold),
            Paragraph("NLP similarity metric to retrieve top matching dataset reviews (Not a classifier).", table_cell_style),
        ],
    ]
    arch_table = Table(arch_data, colWidths=[160, 380])
    arch_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(arch_table)
    story.append(Spacer(1, 10))

    # 7. Model Performance Evaluation Metrics
    story.append(Paragraph("6. Evaluation Metrics Comparison", section_heading))
    metrics = result.get("metrics", {}) or {}
    nb_m = metrics.get("naive_bayes", {})
    lr_m = metrics.get("logistic_regression", {})

    metrics_table_data = [
        [
            Paragraph("Metric", table_header_style),
            Paragraph("Multinomial Naive Bayes", table_header_style),
            Paragraph("Logistic Regression", table_header_style),
        ],
        [
            Paragraph("<b>Accuracy</b>", table_cell_bold),
            Paragraph(f"{nb_m.get('accuracy', 0.0):.2f}%", table_cell_style),
            Paragraph(f"{lr_m.get('accuracy', 0.0):.2f}%", table_cell_style),
        ],
        [
            Paragraph("<b>Precision</b>", table_cell_bold),
            Paragraph(f"{nb_m.get('precision', 0.0):.2f}%", table_cell_style),
            Paragraph(f"{lr_m.get('precision', 0.0):.2f}%", table_cell_style),
        ],
        [
            Paragraph("<b>Recall</b>", table_cell_bold),
            Paragraph(f"{nb_m.get('recall', 0.0):.2f}%", table_cell_style),
            Paragraph(f"{lr_m.get('recall', 0.0):.2f}%", table_cell_style),
        ],
        [
            Paragraph("<b>F1 Score</b>", table_cell_bold),
            Paragraph(f"{nb_m.get('f1_score', 0.0):.2f}%", table_cell_style),
            Paragraph(f"{lr_m.get('f1_score', 0.0):.2f}%", table_cell_style),
        ],
    ]
    metrics_table = Table(metrics_table_data, colWidths=[180, 180, 180])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(metrics_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
