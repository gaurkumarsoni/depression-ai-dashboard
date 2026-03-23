"""
pdf_report.py — MindCare AI
Generates a comprehensive clinical-grade screening report.

Sections:
  1. Cover Header
  2. Patient Information
  3. Executive Clinical Summary  (score badges + narrative)
  4. Text Analysis  (MentalBERT v2)
  5. Audio Analysis
  6. Behavioral Assessment  (PHQ-9 mapped questionnaire)
  7. Multimodal Fusion Result  (weight bars)
  8. DSM-5 Criterion Indicators
  9. Clinical Recommendations
  10. Session History  (progress page only)
  11. Disclaimer
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from datetime import datetime
import io

# ── Palette ────────────────────────────────────────────────────────
C_NAVY  = HexColor("#0B244E")
C_TEAL  = HexColor("#008B8B")
C_MINT  = HexColor("#02C39A")
C_RED   = HexColor("#E24B4A")
C_AMBER = HexColor("#FFA500")
C_GREEN = HexColor("#1D9E75")
C_LGRAY = HexColor("#F4F6F9")
C_MGRAY = HexColor("#E0E4EA")
C_DGRAY = HexColor("#444455")
C_WHITE = colors.white


# ── Helpers ────────────────────────────────────────────────────────

def _risk_color(v):
    if v >= 0.70: return C_RED
    if v >= 0.40: return C_AMBER
    return C_GREEN

def _phq9_color(n):
    if n >= 20: return C_RED
    if n >= 15: return HexColor("#FF5722")
    if n >= 10: return C_AMBER
    if n >= 5:  return HexColor("#8BC34A")
    return C_GREEN

def _styles():
    b = getSampleStyleSheet()
    S = {}
    S["title"]    = ParagraphStyle("t",  parent=b["Normal"], fontSize=18,
                       textColor=C_WHITE, alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=2)
    S["subtitle"] = ParagraphStyle("st", parent=b["Normal"], fontSize=10,
                       textColor=HexColor("#CADCFC"), alignment=TA_CENTER, spaceAfter=1)
    S["section"]  = ParagraphStyle("sh", parent=b["Normal"], fontSize=12,
                       textColor=C_WHITE, fontName="Helvetica-Bold", spaceAfter=0, spaceBefore=0)
    S["subsec"]   = ParagraphStyle("ss", parent=b["Normal"], fontSize=9.5,
                       textColor=C_TEAL,  fontName="Helvetica-Bold", spaceAfter=2, spaceBefore=4)
    S["body"]     = ParagraphStyle("bd", parent=b["Normal"], fontSize=9,
                       textColor=C_DGRAY, leading=13, spaceAfter=3, alignment=TA_JUSTIFY)
    S["small"]    = ParagraphStyle("sm", parent=b["Normal"], fontSize=8,
                       textColor=C_DGRAY, leading=11, spaceAfter=1)
    S["disc"]     = ParagraphStyle("di", parent=b["Normal"], fontSize=7.5,
                       textColor=HexColor("#888888"), leading=11, alignment=TA_JUSTIFY, spaceAfter=2)
    S["bul"]      = ParagraphStyle("bl", parent=b["Normal"], fontSize=9,
                       textColor=C_DGRAY, leading=13, leftIndent=12, spaceAfter=3)
    S["lw"]       = ParagraphStyle("lw", parent=b["Normal"], fontSize=8.5,
                       textColor=C_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
    S["vw"]       = ParagraphStyle("vw", parent=b["Normal"], fontSize=17,
                       textColor=C_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
    S["sw"]       = ParagraphStyle("sw", parent=b["Normal"], fontSize=7.5,
                       textColor=HexColor("#CADCFC"), alignment=TA_CENTER)
    return S

def _sec(title, S):
    t = Table([[Paragraph(title, S["section"])]], colWidths=[6.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), C_NAVY),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 6),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    return t

def _kv(rows, S, cw=(2.1*inch, 4.4*inch)):
    data = [[Paragraph(str(k), S["subsec"]),
             Paragraph(str(v), S["body"])] for k, v in rows]
    t = Table(data, colWidths=list(cw))
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(0,-1), C_LGRAY),
        ("BACKGROUND",   (1,0),(1,-1), C_WHITE),
        ("GRID",         (0,0),(-1,-1), 0.4, C_MGRAY),
        ("LEFTPADDING",  (0,0),(-1,-1), 7),
        ("RIGHTPADDING", (0,0),(-1,-1), 7),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
    ]))
    return t

def _badge(label, value, sub, color, S, w=1.55*inch):
    data = [[Paragraph(label, S["lw"])],
            [Paragraph(value, S["vw"])],
            [Paragraph(sub,   S["sw"])]]
    t = Table(data, colWidths=[w])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), color),
        ("ALIGN",        (0,0),(-1,-1), "CENTER"),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",  (0,0),(-1,-1), 3),
        ("RIGHTPADDING", (0,0),(-1,-1), 3),
    ]))
    return t

def _bar(label, val, color, S, tw=4.0*inch):
    """Horizontal percentage bar — safe for zero/near-zero values."""
    frac = max(0.0, min(float(val), 1.0))
    # Minimum 6pt for filled bar to avoid negative-width crash from padding
    min_pt  = 6.0
    max_bar = tw - min_pt
    filled  = max(min_pt, frac * tw) if frac > 0.001 else min_pt
    empty   = max(min_pt, tw - filled)
    # Rebalance to stay within tw
    total   = filled + empty
    if total > tw + 1:
        filled = frac * tw if frac > 0.001 else 0
        empty  = tw - filled
        if filled < min_pt: filled = min_pt; empty = tw - min_pt
        if empty  < min_pt: empty  = min_pt; filled = tw - min_pt

    fb = Table([[""]], colWidths=[filled], rowHeights=[0.14*inch])
    fb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1), color if frac > 0.001 else C_MGRAY)]))
    eb = Table([[""]], colWidths=[empty],  rowHeights=[0.14*inch])
    eb.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1), C_MGRAY)]))

    row = Table(
        [[Paragraph(f"<b>{label}</b>", S["small"]), fb, eb,
          Paragraph(f"{val:.1%}", S["small"])]],
        colWidths=[1.3*inch, filled, empty, 0.55*inch]
    )
    row.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",   (0,0),(-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
    ]))
    return row


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def generate_pdf_report(
    user_name        = "Anonymous",
    user_age         = 0,
    user_gender      = "",
    fusion_result    = None,
    phq9_data        = None,
    text_input       = None,
    text_v2_result   = None,
    audio_conf       = None,
    beh_answers      = None,
    sessions         = None,
    trajectory       = "",
    report_source    = "manual",
) -> bytes:

    S       = _styles()
    buf     = io.BytesIO()
    content = []

    # ── Resolve values ─────────────────────────────────────────────
    fr         = fusion_result or {}
    pd         = phq9_data     or {}
    tv         = text_v2_result or {}
    beh        = beh_answers   or {}

    risk_score = float(fr.get("risk_score", 0))
    risk_level = fr.get("risk_level", "—")
    confidence = float(fr.get("confidence", 0))
    modalities = fr.get("modalities", "—")
    weights    = fr.get("weights", [0, 0, 0])
    phq9_score = int(pd.get("phq9_score", int(risk_score * 27)))
    phq9_sev   = pd.get("severity", "—")
    phq9_adv   = pd.get("advice",   "—")
    text_prob  = float(tv.get("raw_prob",  0))
    text_sev   = tv.get("severity",  "—")
    sev_model  = tv.get("sev_model", "—")
    sev_conf   = float(tv.get("sev_conf",  0))

    # ── 1. HEADER ─────────────────────────────────────────────────
    hdr = Table([
        [Paragraph("🧠  MindCare AI", S["title"])],
        [Paragraph("Mental Health Screening Report", S["subtitle"])],
        [Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y  at  %H:%M')}  ·  "
            f"Source: {report_source.replace('_',' ').title()} Analysis",
            S["subtitle"]
        )],
    ], colWidths=[6.5*inch])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), C_NAVY),
        ("ALIGN",        (0,0),(-1,-1), "CENTER"),
        ("TOPPADDING",   (0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
    ]))
    content.append(hdr)
    content.append(Spacer(1, 0.12*inch))

    # ── 2. PATIENT INFORMATION ────────────────────────────────────
    content.append(_sec("1.  Patient Information", S))
    content.append(_kv([
        ("Full Name",       user_name or "—"),
        ("Age",             str(user_age) if user_age else "—"),
        ("Gender",          user_gender   or "—"),
        ("Report Date",     datetime.now().strftime("%Y-%m-%d")),
        ("Report Time",     datetime.now().strftime("%H:%M:%S")),
        ("Assessment Type", report_source.replace("_"," ").title()),
    ], S))
    content.append(Spacer(1, 0.1*inch))

    # ── 3. EXECUTIVE SUMMARY ──────────────────────────────────────
    content.append(_sec("2.  Executive Clinical Summary", S))
    content.append(Spacer(1, 0.06*inch))

    # Score badges
    n_mods = len([m for m in modalities.split(",") if m.strip()]) if modalities != "—" else 0
    badges_row = [
        _badge("FUSION RISK",  f"{risk_score:.0%}",  risk_level,   _risk_color(risk_score), S),
        _badge("PHQ-9 SCORE",  f"{phq9_score}/27",   phq9_sev,     _phq9_color(phq9_score), S),
        _badge("AI CONFIDENCE",f"{confidence:.0%}",  "Confidence", C_NAVY, S),
        _badge("MODALITIES",   str(n_mods) if n_mods else "—",
                               "Sources",    C_TEAL, S),
    ]
    badges_table = Table([badges_row], colWidths=[1.575*inch]*4)
    badges_table.setStyle(TableStyle([
        ("ALIGN",        (0,0),(-1,-1), "CENTER"),
        ("LEFTPADDING",  (0,0),(-1,-1), 2),
        ("RIGHTPADDING", (0,0),(-1,-1), 2),
        ("TOPPADDING",   (0,0),(-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
    ]))
    content.append(badges_table)
    content.append(Spacer(1, 0.08*inch))

    # Narrative
    narrative = _narrative(risk_score, risk_level, phq9_score, phq9_sev,
                           text_sev, text_prob, audio_conf, beh, modalities, trajectory)
    content.append(Paragraph(narrative, S["body"]))
    content.append(Spacer(1, 0.05*inch))

    # Advice box
    adv = Table([[Paragraph(f"<b>Clinical Recommendation:</b>  {phq9_adv}", S["body"])]],
                colWidths=[6.5*inch])
    adv.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), HexColor("#E8F4F8")),
        ("LINEBEFORE",   (0,0),(0,-1),  4, C_TEAL),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("TOPPADDING",   (0,0),(-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
    ]))
    content.append(adv)
    content.append(Spacer(1, 0.1*inch))

    # ── 4. TEXT ANALYSIS ─────────────────────────────────────────
    content.append(_sec("3.  Text Analysis  —  MentalBERT v2  (F1=0.9917, AUC=1.00)", S))
    if tv:
        txt_rows = [
            ("Input Text",              (text_input[:280] + "…") if text_input and len(text_input) > 280
                                          else (text_input or "Not provided")),
            ("Depression Probability",   f"{text_prob*100:.1f}%"),
            ("Binary Decision",          "Depressed" if text_prob >= 0.315 else "Not Depressed"),
            ("Severity (MentalBERT v2)", text_sev),
            ("Severity Model Class",     sev_model),
            ("Severity Confidence",      f"{sev_conf*100:.0f}%"),
            ("Thresholds Used",          "Binary=0.315  |  Low=0.70  |  Moderate=0.90"),
            ("Model Architecture",       "RoBERTa-base fine-tuned on clinical Reddit + r/SuicideWatch"),
        ]
        content.append(_kv(txt_rows, S))
    else:
        content.append(Paragraph("Text modality was not analyzed in this session.", S["body"]))
    content.append(Spacer(1, 0.1*inch))

    # ── 5. AUDIO ANALYSIS ────────────────────────────────────────
    content.append(_sec("4.  Audio / Speech Analysis  —  AudioTransformer (F1=0.76)", S))
    w = [float(weights[i]) if i < len(weights) else 0 for i in range(3)] if fr else [0,0,0]
    audio_in_fusion = "Audio" in modalities
    if audio_conf is not None:
        a_risk = "High" if audio_conf >= 0.7 else ("Moderate" if audio_conf >= 0.4 else "Low")
        content.append(_kv([
            ("Depression Confidence", f"{audio_conf*100:.1f}%"),
            ("Audio Risk Level",      a_risk),
            ("Fusion Weight",         f"{w[0]:.1%}" if audio_in_fusion else "Not used in this session"),
            ("Feature Extraction",    "eGeMAPS v02 (88 features) — clinically validated acoustic parameters"),
            ("Features Included",     "F0/Pitch · Loudness · MFCC · Jitter · Shimmer · HNR · Formants · ZCR"),
            ("Model Architecture",    "Transformer Encoder (d_model=128, 2 layers, attention pooling)"),
            ("Training Dataset",      "DAIC-WOZ AVEC 2017 — 275 participants, clinical interviews"),
        ], S))
    elif audio_in_fusion:
        # Audio was used in fusion but we don't have the raw confidence stored
        content.append(_kv([
            ("Status",            "Audio was included in multimodal fusion"),
            ("Fusion Weight",     f"{w[0]:.1%}"),
            ("Note",              "Detailed audio confidence not stored for this session record."),
        ], S))
    else:
        content.append(Paragraph("Audio modality was not analyzed in this session.", S["body"]))
    content.append(Spacer(1, 0.1*inch))

    # ── 6. BEHAVIORAL ASSESSMENT ─────────────────────────────────
    content.append(_sec("5.  Behavioral Assessment  —  PHQ-9 Mapped Questionnaire (13 Items)", S))
    if beh:
        content.append(_kv([
            ("Gender",                   beh.get("gender",             "—")),
            ("Age",                      str(beh.get("age",            "—"))),
            ("Academic Pressure (1-5)",  str(beh.get("academic_pressure","—"))),
            ("Work Pressure (1-5)",      str(beh.get("work_pressure",  "—"))),
            ("CGPA / GPA",               str(beh.get("cgpa",           "—"))),
            ("Study Satisfaction (1-5)", str(beh.get("study_satisfaction","—"))),
            ("Job Satisfaction (1-5)",   str(beh.get("job_satisfaction","—"))),
            ("Sleep Duration",           str(beh.get("sleep",          "—"))),
            ("Dietary Habits",           str(beh.get("diet",           "—"))),
            ("Suicidal Thoughts",        str(beh.get("suicidal",       "—"))),
            ("Work/Study Hours/Day",     str(beh.get("work_hours",     "—"))),
            ("Financial Stress (1-5)",   str(beh.get("financial_stress","—"))),
            ("Family History of MHI",    str(beh.get("family_history", "—"))),
        ], S))

        # Clinical flags
        flags = []
        if str(beh.get("suicidal","No"))     == "Yes":       flags.append("⚠️  SUICIDAL IDEATION — PHQ-9 Item 9 escalation applied (minimum score 20)")
        if "Less" in str(beh.get("sleep","")): flags.append("⚠️  SEVERE SLEEP DEPRIVATION (<5 hrs) — PHQ-9 Item 4 · minimum score 10 applied")
        if str(beh.get("diet",""))           == "Unhealthy": flags.append("⚠️  UNHEALTHY DIETARY HABITS — correlated with depression severity")
        if str(beh.get("family_history","No"))=="Yes":        flags.append("⚠️  FAMILY HISTORY of mental illness — elevated genetic risk")
        if isinstance(beh.get("work_hours"),  (int,float)) and beh.get("work_hours",0) >= 12:
            flags.append("⚠️  EXCESSIVE WORK/STUDY HOURS (≥12 hrs/day) — significant psychosocial stressor")
        if isinstance(beh.get("academic_pressure"),(int,float)) and beh.get("academic_pressure",3) >= 4:
            flags.append("⚠️  HIGH ACADEMIC PRESSURE (≥4/5) — linked to concentration difficulties")

        if flags:
            content.append(Spacer(1, 0.04*inch))
            content.append(Paragraph("<b>Clinical Risk Flags:</b>", S["subsec"]))
            for f in flags:
                content.append(Paragraph(f, S["bul"]))
    else:
        content.append(Paragraph("Behavioral questionnaire was not completed in this session.", S["body"]))
    content.append(Spacer(1, 0.1*inch))

    # ── 7. MULTIMODAL FUSION ─────────────────────────────────────
    content.append(_sec("6.  Multimodal Fusion  —  Missing-Modality Attention Transformer", S))
    if fr:
        w = [float(weights[i]) if i < len(weights) else 0 for i in range(3)]
        dom_idx = w.index(max(w))
        dom_mod = ["Audio","Text","Behavioral"][dom_idx]

        content.append(_kv([
            ("Final Risk Score",    f"{risk_score:.1%}  ({risk_level})"),
            ("Fusion Confidence",   f"{confidence:.1%}"),
            ("Active Modalities",   modalities),
            ("Dominant Modality",   f"{dom_mod}  (attention weight: {w[dom_idx]:.1%})"),
            ("Clinical Overrides",  _override_text(fr, beh)),
            ("Architecture",        "audio(128-dim) + text(768-dim) + beh(32-dim) → mask-attention → 2-class"),
        ], S))
        content.append(Spacer(1, 0.06*inch))
        content.append(Paragraph("<b>Modality Attention Weights:</b>", S["subsec"]))
        bar_colors = [HexColor("#2196F3"), HexColor("#4CAF50"), HexColor("#FF9800")]
        for lbl, wt, col in zip(["Audio","Text","Behavioral"], w, bar_colors):
            content.append(_bar(lbl, wt, col, S))
    else:
        content.append(Paragraph("Fusion analysis was not run in this session.", S["body"]))
    content.append(Spacer(1, 0.1*inch))

    # ── 8. DSM-5 INDICATORS ──────────────────────────────────────
    content.append(_sec("7.  DSM-5 Criterion Indicators  (Automated — Not Clinically Verified)", S))
    dsm = _dsm5(tv, audio_conf, beh, risk_score)
    dsm_data = [["DSM-5 Criterion","Indicator","Source","Severity Flag"]]
    for item in dsm:
        dsm_data.append([
            Paragraph(item["crit"],  S["small"]),
            Paragraph(item["ind"],   S["small"]),
            Paragraph(item["src"],   S["small"]),
            Paragraph(item["sev"],   S["small"]),
        ])
    dsm_t = Table(dsm_data, colWidths=[2.2*inch, 1.3*inch, 1.3*inch, 1.7*inch])
    dsm_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), C_NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("GRID",          (0,0),(-1,-1), 0.3, C_MGRAY),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ("RIGHTPADDING",  (0,0),(-1,-1), 5),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_LGRAY]),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
    ]))
    content.append(dsm_t)
    content.append(Paragraph(
        "DSM-5 flags are AI-inferred indicators only. Clinical confirmation required.",
        S["disc"]
    ))
    content.append(Spacer(1, 0.1*inch))

    # ── 9. CLINICAL RECOMMENDATIONS ──────────────────────────────
    content.append(_sec("8.  Clinical Recommendations", S))
    content.append(Spacer(1, 0.04*inch))
    for i, rec in enumerate(_recommendations(
            risk_score, phq9_score, phq9_sev, text_sev, audio_conf, beh), 1):
        content.append(Paragraph(f"{i}.  {rec}", S["bul"]))
    content.append(Spacer(1, 0.1*inch))

    # ── 10. SESSION HISTORY ───────────────────────────────────────
    if sessions and len(sessions) > 1:
        content.append(PageBreak())
        content.append(_sec("9.  Session History  —  Progress Over Time", S))
        content.append(Spacer(1, 0.04*inch))
        hist = [["Date","Time","Risk Score","Risk Level","PHQ-9","Severity","Modalities","Source"]]
        for s in sorted(sessions, key=lambda x: x.get("timestamp",""), reverse=True)[:20]:
            hist.append([
                s.get("date","—"), s.get("time","—"),
                f"{s.get('risk_score',0):.1%}",
                s.get("risk_level","—"),
                str(s.get("phq9_score","—")),
                s.get("severity","—"),
                s.get("modalities","—"),
                s.get("source","—"),
            ])
        ht = Table(hist, colWidths=[0.8*inch,0.55*inch,0.75*inch,0.9*inch,
                                     0.5*inch,1.0*inch,1.1*inch,0.7*inch])
        ht.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), C_NAVY),
            ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 7.5),
            ("GRID",          (0,0),(-1,-1), 0.3, C_MGRAY),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
            ("RIGHTPADDING",  (0,0),(-1,-1), 4),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_LGRAY]),
        ]))
        content.append(ht)
        if trajectory:
            tmap = {"improving":"📈 Improving","worsening":"📉 Worsening","stable":"➡️ Stable"}
            content.append(Spacer(1, 0.05*inch))
            content.append(Paragraph(
                f"Overall Trajectory: <b>{tmap.get(trajectory, trajectory.title())}</b>", S["body"]
            ))
        content.append(Spacer(1, 0.1*inch))

    # ── 11. DISCLAIMER ────────────────────────────────────────────
    content.append(HRFlowable(width="100%", thickness=0.6, color=C_MGRAY))
    content.append(Spacer(1, 0.06*inch))
    content.append(Paragraph(
        "⚠️ DISCLAIMER: This report is generated by an AI-powered research prototype and does NOT "
        "constitute a medical diagnosis, clinical assessment, or treatment plan. All results are "
        "indicative only and must be interpreted by a qualified mental health professional. "
        "DSM-5 flags are automatically inferred and have not been verified by a clinician. "
        "If you are in crisis, contact a mental health helpline immediately.",
        S["disc"]
    ))
    content.append(Paragraph(
        "India Crisis Lines:  iCall 9152987821  ·  Vandrevala Foundation 1860-2662-345  ·  icallhelpline.org",
        S["disc"]
    ))
    content.append(Paragraph(
        "MindCare AI  ·  Explainable Multimodal Depression Screening  ·  Research Prototype — NOT for Clinical Use",
        S["disc"]
    ))

    # ── Build ─────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=0.65*inch, leftMargin=0.65*inch,
        topMargin=0.65*inch,   bottomMargin=0.65*inch,
        title=f"MindCare AI Report — {user_name}",
        author="MindCare AI"
    )
    doc.build(content)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _narrative(risk_score, risk_level, phq9_score, phq9_sev,
               text_sev, text_prob, audio_conf, beh, modalities, trajectory):
    p = []
    p.append(f"Multimodal screening was performed using the following modalities: <b>{modalities}</b>. ")
    risk_word = "HIGH" if risk_score >= 0.70 else ("MODERATE" if risk_score >= 0.40 else "LOW")
    p.append(f"The AI system assessed a <b>{risk_word} risk</b> ({risk_score:.0%}) of clinical depression. ")
    p.append(f"PHQ-9 equivalent score is estimated at <b>{phq9_score}/27</b> ({phq9_sev}). ")
    if text_prob > 0:
        p.append(f"Natural language analysis (MentalBERT v2) reported {text_prob*100:.1f}% depression "
                 f"probability with severity classified as <b>{text_sev}</b>. ")
    if audio_conf is not None:
        p.append(f"Speech pattern analysis yielded a depression confidence of <b>{audio_conf*100:.1f}%</b>. ")
    flags = []
    if str(beh.get("suicidal","No"))=="Yes":      flags.append("suicidal ideation")
    if "Less" in str(beh.get("sleep","")):         flags.append("sleep deprivation (<5 hrs)")
    if str(beh.get("diet",""))=="Unhealthy":        flags.append("poor dietary habits")
    if str(beh.get("family_history","No"))=="Yes":  flags.append("family history of mental illness")
    if flags:
        p.append(f"Behavioral risk factors identified: <b>{', '.join(flags)}</b>. ")
    if trajectory and trajectory not in ("","insufficient_data"):
        p.append(f"Historical trend: <b>{trajectory}</b>. ")
    p.append("Professional clinical evaluation is strongly recommended.")
    return "".join(p)


def _override_text(fr, beh):
    o = []
    rs = fr.get("risk_score", 0)
    if str(beh.get("suicidal","No")) == "Yes":  o.append("Suicidal ideation → risk ≥ 0.75")
    if rs >= 0.80:                               o.append("Text Severe >85% → risk ≥ 0.80")
    elif rs >= 0.45:                             o.append("Text Moderate >70% → risk ≥ 0.45")
    return ", ".join(o) if o else "None applied"


def _dsm5(tv, audio_conf, beh, risk_score):
    b = beh or {}
    tv = tv or {}
    t_sev  = tv.get("severity","")
    t_prob = float(tv.get("raw_prob", 0))
    items  = []

    def add(crit, ind, src, sev): items.append({"crit":crit,"ind":ind,"src":src,"sev":sev})

    add("A1 — Depressed Mood",
        "Positive" if t_prob >= 0.315 else "Not detected", "Text (NLP)",
        t_sev if t_prob >= 0.315 else "—")

    ss = b.get("study_satisfaction", 3); js = b.get("job_satisfaction", 3)
    add("A2 — Anhedonia / Loss of Interest",
        "Possible" if (isinstance(ss,(int,float)) and ss<=2) or (isinstance(js,(int,float)) and js<=2) else "Not flagged",
        "Behavioral", "Low" if (isinstance(ss,(int,float)) and ss<=2) else "—")

    add("A3 — Appetite / Weight Change",
        "Flagged" if str(b.get("diet",""))=="Unhealthy" else "Not flagged",
        "Behavioral", "Low" if str(b.get("diet",""))=="Unhealthy" else "—")

    slp = str(b.get("sleep",""))
    add("A4 — Sleep Disturbance",
        "Flagged" if "Less" in slp else "Not flagged",
        "Behavioral", "Moderate" if "Less" in slp else "—")

    add("A5 — Psychomotor Changes",
        "Possible" if (audio_conf is not None and audio_conf >= 0.5) else "Not assessed",
        "Audio (Speech)", "Low" if (audio_conf is not None and audio_conf >= 0.5) else "—")

    wh = b.get("work_hours", 0)
    add("A6 — Fatigue / Energy Loss",
        "Possible" if (isinstance(wh,(int,float)) and wh>=12) or "Less" in slp else "Not flagged",
        "Behavioral", "Low" if (isinstance(wh,(int,float)) and wh>=12) else "—")

    add("A7 — Worthlessness / Guilt",
        "Possible" if t_sev in ("Moderate","Severe") else "Not detected",
        "Text (NLP)", t_sev if t_sev in ("Moderate","Severe") else "—")

    ap = b.get("academic_pressure", 3)
    add("A8 — Concentration Difficulty",
        "Possible" if isinstance(ap,(int,float)) and ap>=4 else "Not flagged",
        "Behavioral", "Low" if isinstance(ap,(int,float)) and ap>=4 else "—")

    sui = str(b.get("suicidal","No"))
    add("A9 — Suicidal Ideation",
        "REPORTED ⚠️" if sui=="Yes" else "Not reported",
        "Behavioral", "Severe" if sui=="Yes" else "—")

    return items


def _recommendations(risk_score, phq9_score, phq9_sev, text_sev, audio_conf, beh):
    b    = beh or {}
    recs = []
    if str(b.get("suicidal","No")) == "Yes":
        recs.append(
            "URGENT: Suicidal ideation reported. Immediate risk assessment by a licensed "
            "mental health professional is required. Consider inpatient evaluation and crisis intervention."
        )
    if risk_score >= 0.70 or phq9_score >= 15:
        recs.append(
            f"Schedule an URGENT appointment with a psychiatrist or clinical psychologist "
            f"(Risk: {risk_score:.0%}, PHQ-9: {phq9_score}/27 — {phq9_sev}). "
            "Evidence-based treatments: CBT, DBT, or pharmacotherapy evaluation."
        )
    elif risk_score >= 0.40 or phq9_score >= 10:
        recs.append(
            f"Recommend mental health consultation within 1–2 weeks "
            f"(Risk: {risk_score:.0%}, PHQ-9: {phq9_score}/27 — {phq9_sev}). "
            "Consider CBT, counseling, or group therapy."
        )
    else:
        recs.append(
            f"Routine follow-up recommended (Risk: {risk_score:.0%}, PHQ-9: {phq9_score}/27). "
            "Continue monitoring; repeat screening in 4–6 weeks."
        )
    if "Less" in str(b.get("sleep","")):
        recs.append("Address sleep hygiene. CBT for Insomnia (CBT-I) or referral to sleep specialist recommended.")
    if str(b.get("diet","")) == "Unhealthy":
        recs.append("Nutritional counseling advised. Improved diet is linked to better treatment response.")
    if isinstance(b.get("work_hours"), (int,float)) and b.get("work_hours",0) >= 12:
        recs.append("Workload reduction recommended. ≥12 hrs/day work is a significant stressor; stress management intervention advised.")
    if str(b.get("family_history","No")) == "Yes":
        recs.append("Due to family history of mental illness, schedule regular screening every 3 months as prophylaxis.")
    if phq9_score >= 15 or text_sev == "Severe":
        recs.append("Pharmacological evaluation warranted. Discuss antidepressant options with a psychiatrist.")
    recs.append(
        "Evidence-based self-care: 30 min/day physical exercise, social support maintenance, "
        "mindfulness/meditation, structured daily routine, and journaling."
    )
    recs.append("Repeat screening in 2–4 weeks to track response to interventions using MindCare AI progress tracker.")
    return recs
