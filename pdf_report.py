from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import io


def generate_pdf_report(
    user_name    : str,
    user_age     : int,
    sessions     : list,
    latest_result: dict,
    phq9_data    : dict,
    trajectory   : str
) -> bytes:
    """Generate a PDF mental health report"""

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch,   bottomMargin=0.75*inch
    )

    styles  = getSampleStyleSheet()
    content = []

    # ── Title ──────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title",
        parent    = styles["Heading1"],
        fontSize  = 20,
        textColor = colors.HexColor("#1a237e"),
        alignment = TA_CENTER,
        spaceAfter= 6
    )
    sub_style = ParagraphStyle(
        "Sub",
        parent    = styles["Normal"],
        fontSize  = 11,
        textColor = colors.HexColor("#546e7a"),
        alignment = TA_CENTER,
        spaceAfter= 4
    )

    content.append(Paragraph("🧠 MindCare AI", title_style))
    content.append(Paragraph("Mental Health Screening Report", sub_style))
    content.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        sub_style
    ))
    content.append(HRFlowable(width="100%", thickness=2,
                               color=colors.HexColor("#1a237e")))
    content.append(Spacer(1, 0.2*inch))

    # ── Patient Info ───────────────────────────────────────────────
    heading_style = ParagraphStyle(
        "Heading",
        parent    = styles["Heading2"],
        fontSize  = 13,
        textColor = colors.HexColor("#1a237e"),
        spaceAfter= 6
    )
    normal_style = ParagraphStyle(
        "Normal",
        parent    = styles["Normal"],
        fontSize  = 10,
        spaceAfter= 4
    )

    content.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ["Name",         user_name],
        ["Age",          str(user_age)],
        ["Total Sessions", str(len(sessions))],
        ["Report Date",  datetime.now().strftime("%Y-%m-%d")]
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), colors.HexColor("#e8eaf6")),
        ("TEXTCOLOR",   (0,0), (-1,-1), colors.black),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("PADDING",     (0,0), (-1,-1), 8),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#c5cae9")),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
    ]))
    content.append(patient_table)
    content.append(Spacer(1, 0.2*inch))

    # ── Latest Screening Result ────────────────────────────────────
    content.append(Paragraph("Latest Screening Result", heading_style))

    risk_score = latest_result.get("risk_score", 0)
    risk_level = latest_result.get("risk_level", "Unknown")
    confidence = latest_result.get("confidence", 0)

    # Color based on risk
    risk_color = colors.HexColor(phq9_data.get("color", "#4CAF50"))

    result_data = [
        ["Risk Score",      f"{risk_score:.2f} / 1.00"],
        ["Risk Level",      risk_level],
        ["Confidence",      f"{confidence:.1%}"],
        ["PHQ-9 Equivalent",f"{phq9_data['phq9_score']} / 27"],
        ["Severity",        phq9_data["severity"]],
        ["Trajectory",      trajectory.replace("_", " ").title()],
    ]
    result_table = Table(result_data, colWidths=[2.5*inch, 3.5*inch])
    result_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), colors.HexColor("#e8eaf6")),
        ("TEXTCOLOR",   (0,0), (-1,-1), colors.black),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("PADDING",     (0,0), (-1,-1), 8),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#c5cae9")),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (1,1), (1,1), risk_color),
        ("FONTNAME",    (1,1), (1,1), "Helvetica-Bold"),
    ]))
    content.append(result_table)
    content.append(Spacer(1, 0.2*inch))

    # ── Clinical Advice ────────────────────────────────────────────
    content.append(Paragraph("Clinical Recommendation", heading_style))
    content.append(Paragraph(phq9_data["advice"], normal_style))
    content.append(Spacer(1, 0.1*inch))

    # ── Session History ────────────────────────────────────────────
    if sessions:
        content.append(Paragraph("Session History", heading_style))
        history_data = [["Date", "Risk Score", "Risk Level", "Modalities"]]
        for s in sessions[-10:]:  # last 10 sessions
            history_data.append([
                s.get("date",       ""),
                f"{s.get('risk_score', 0):.2f}",
                s.get("risk_level", ""),
                s.get("modalities", "")
            ])
        history_table = Table(
            history_data,
            colWidths=[1.5*inch, 1.2*inch, 1.5*inch, 2.3*inch]
        )
        history_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1a237e")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("PADDING",     (0,0), (-1,-1), 6),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#c5cae9")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.white, colors.HexColor("#f5f5f5")]),
        ]))
        content.append(history_table)
        content.append(Spacer(1, 0.2*inch))

    # ── Disclaimer ────────────────────────────────────────────────
    content.append(HRFlowable(width="100%", thickness=1,
                               color=colors.HexColor("#c5cae9")))
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent    = styles["Normal"],
        fontSize  = 8,
        textColor = colors.HexColor("#9e9e9e"),
        spaceAfter= 4
    )
    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph(
        "⚠️ DISCLAIMER: This report is generated by an AI screening tool "
        "and is NOT a medical diagnosis. Please consult a qualified mental "
        "health professional for proper evaluation and treatment. "
        "If you are in crisis, contact a mental health helpline immediately.",
        disclaimer_style
    ))
    content.append(Paragraph(
        "MindCare AI | Powered by Multimodal Depression Detection Research",
        disclaimer_style
    ))

    # Build PDF
    doc.build(content)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
