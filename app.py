import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from datetime import date
import io
import os
import tempfile

# =========================================================
# PDF GENERATION  (reportlab)
# =========================================================
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, Image as RLImage, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MSSA-Net | Seizure Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
/* ---- global ---- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* ---- metric cards ---- */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    color: #e0e0e0;
}
.metric-card .label {
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7ecfff;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff;
}

/* ---- section header ---- */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #7ecfff;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-bottom: 1px solid #0f3460;
    padding-bottom: 6px;
    margin-bottom: 14px;
}

/* ---- seizure alert ---- */
.alert-seizure {
    background: linear-gradient(135deg, #3d0000, #5c1a00);
    border-left: 4px solid #ff4444;
    border-radius: 8px;
    padding: 16px 20px;
    color: #ffcccc;
    font-size: 1.1rem;
    font-weight: 600;
}
.alert-normal {
    background: linear-gradient(135deg, #003d00, #005c1a);
    border-left: 4px solid #44ff88;
    border-radius: 8px;
    padding: 16px 20px;
    color: #ccffdd;
    font-size: 1.1rem;
    font-weight: 600;
}

/* ---- recommendation box ---- */
.rec-box {
    background: #0d1b2a;
    border: 1px solid #1a3a5c;
    border-radius: 8px;
    padding: 14px 18px;
    color: #c8d8e8;
    font-size: 0.92rem;
    line-height: 1.7;
}

/* ---- sidebar ---- */
.sidebar-info {
    background: #0d1b2a;
    border: 1px solid #1a3a5c;
    border-radius: 8px;
    padding: 12px 16px;
    color: #c8d8e8;
    font-size: 0.86rem;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL PATHS  – update these to your actual paths
# =========================================================
MODEL_PATHS = {
    "CHB-MIT": r"D:\MSSA_Net_Project\outputs\models\mssa_net.keras",
    "Bonn":    r"D:\MSSA_Net_Project\outputs\models\mssa_net_bonn.keras",
}

@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.image("eeg.png", use_container_width=True, caption="EEG-based Seizure Detection")
    st.markdown("## 🧠 MSSA-Net")
    st.markdown("""
<div class="sidebar-info">
<b>Version</b> &nbsp; 1.0<br>
<b>Framework</b> &nbsp; TensorFlow / Keras<br>
<b>Input Size</b> &nbsp; 224 × 224 px<br>
<b>Architecture</b> &nbsp; Multi-Scale Spatial Attention Network<br><br>
<b>Supported Datasets</b><br>
✓ &nbsp; CHB-MIT Scalp EEG<br>
✓ &nbsp; Bonn University EEG<br><br>
<b>Classes</b><br>
🔴 &nbsp; Seizure<br>
🟢 &nbsp; Non-Seizure
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚕️ For research / demonstration purposes only. Not a clinical diagnostic tool.")

# =========================================================
# TITLE
# =========================================================
st.markdown("# 🧠 Epileptic Seizure Detection")
st.markdown("##### Clinical Decision Support · MSSA-Net · Wavelet Scalogram Analysis")
st.markdown("---")

# =========================================================
# PATIENT INFORMATION
# =========================================================
st.markdown('<div class="section-header">👤 Patient Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    patient_id   = st.text_input("Patient ID",   placeholder="e.g. PT-2024-001")
    patient_name = st.text_input("Patient Name", placeholder="Full name")
with col2:
    age    = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
with col3:
    exam_date = st.date_input("Examination Date", value=date.today())
    referring  = st.text_input("Referring Clinician", placeholder="Optional")

st.markdown("---")

# =========================================================
# DATASET / MODEL SELECTION
# =========================================================
st.markdown('<div class="section-header">📂 Dataset & Model Selection</div>', unsafe_allow_html=True)

dataset = st.radio(
    "Select the dataset the scalogram belongs to:",
    options=["CHB-MIT", "Bonn"],
    horizontal=True,
    help="CHB-MIT: paediatric scalp EEG · Bonn: surface/depth EEG"
)

model_path = MODEL_PATHS[dataset]
if os.path.exists(model_path):
    model = load_model(model_path)
    st.success(f"✅ Model loaded: **{dataset}** ({os.path.basename(model_path)})")
else:
    st.error(f"⚠️ Model file not found: `{model_path}`. Update MODEL_PATHS in app.py.")
    model = None

st.markdown("---")

# =========================================================
# IMAGE UPLOAD
# =========================================================
st.markdown('<div class="section-header">🖼️ Upload Wavelet Scalogram</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a wavelet scalogram image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).convert("RGB")

    col_img, col_meta = st.columns([1, 1])
    with col_img:
        st.image(uploaded_image, caption="Uploaded Scalogram", use_column_width=True)
    with col_meta:
        st.markdown("**Image details**")
        st.write(f"• Format : `{uploaded_file.type}`")
        st.write(f"• Size   : `{uploaded_image.size[0]} × {uploaded_image.size[1]} px`")
        st.write(f"• Mode   : `{uploaded_image.mode}`")
        st.markdown("> Image will be resized to **224 × 224** before inference.")

    st.markdown("---")

    # =========================================================
    # PREDICT
    # =========================================================
    predict_btn = st.button("🔍 Run Prediction", type="primary", disabled=(model is None))

    if predict_btn and model is not None:

        with st.spinner("Running inference …"):
            # Save temp → load via tf
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                uploaded_image.save(tmp.name)
                tmp_path = tmp.name

            img       = tf.keras.utils.load_img(tmp_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            prediction = model.predict(img_array)

        os.unlink(tmp_path)

        non_seizure_prob = float(prediction[0][0])
        seizure_prob     = float(prediction[0][1])

        if seizure_prob > non_seizure_prob:
            predicted_label = "Seizure"
            confidence      = seizure_prob * 100
        else:
            predicted_label = "Non-Seizure"
            confidence      = non_seizure_prob * 100

        st.markdown("---")

        # =========================================================
        # PREDICTION SUMMARY CARDS
        # =========================================================
        st.markdown('<div class="section-header">📊 Prediction Summary</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
<div class="metric-card">
  <div class="label">Dataset</div>
  <div class="value">{dataset}</div>
</div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
<div class="metric-card">
  <div class="label">Prediction</div>
  <div class="value">{'⚠️ ' if predicted_label == 'Seizure' else '✅ '}{predicted_label}</div>
</div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
<div class="metric-card">
  <div class="label">Confidence</div>
  <div class="value">{confidence:.2f}%</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Alert banner
        if predicted_label == "Seizure":
            st.markdown(f'<div class="alert-seizure">⚠️ &nbsp; Seizure activity detected with {confidence:.2f}% confidence.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-normal">✅ &nbsp; No seizure activity detected. Confidence: {confidence:.2f}%</div>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # =========================================================
        # PROBABILITY CHART
        # =========================================================
        st.markdown('<div class="section-header">📈 Prediction Probabilities</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 2.2))
        fig.patch.set_facecolor("#0d1b2a")
        ax.set_facecolor("#0d1b2a")

        classes = ["Non-Seizure", "Seizure"]
        probs   = [non_seizure_prob, seizure_prob]
        bar_colors = ["#44cc88", "#ff4444"]

        bars = ax.barh(classes, probs, color=bar_colors, height=0.45,
                       edgecolor="none", zorder=3)

        for bar, prob in zip(bars, probs):
            ax.text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.4f}", va="center", ha="left",
                    color="white", fontsize=11, fontweight="bold")

        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Probability", color="#7ecfff", fontsize=10)
        ax.tick_params(colors="white", labelsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.grid(True, color="#1a3a5c", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")

        # =========================================================
        # CLINICAL INTERPRETATION
        # =========================================================
        st.markdown('<div class="section-header">🏥 Clinical Interpretation</div>', unsafe_allow_html=True)

        if predicted_label == "Seizure":
            rec_html = """
<div class="rec-box">
<b>Prediction:</b> &nbsp; ⚠️ Seizure<br>
<b>Confidence:</b> &nbsp; {conf:.2f}%<br><br>
<b>Recommendations:</b><br>
• &nbsp; Immediate neurological assessment advised<br>
• &nbsp; Review full EEG recording for confirmation<br>
• &nbsp; Correlate with patient's clinical history<br>
• &nbsp; Neurologist consultation recommended<br>
• &nbsp; Consider anti-epileptic medication review<br><br>
<i style="color:#ff9999;">⚕️ This is an AI-assisted recommendation only and does not replace clinical judgement.</i>
</div>
""".format(conf=confidence)
        else:
            rec_html = """
<div class="rec-box">
<b>Prediction:</b> &nbsp; ✅ Non-Seizure<br>
<b>Confidence:</b> &nbsp; {conf:.2f}%<br><br>
<b>Recommendations:</b><br>
• &nbsp; No seizure activity detected in this scalogram<br>
• &nbsp; Continue routine monitoring as clinically indicated<br>
• &nbsp; Clinical correlation with patient symptoms advised<br>
• &nbsp; Repeat EEG if symptoms persist<br><br>
<i style="color:#99ffcc;">⚕️ This is an AI-assisted recommendation only and does not replace clinical judgement.</i>
</div>
""".format(conf=confidence)

        st.markdown(rec_html, unsafe_allow_html=True)

        st.markdown("---")

        # =========================================================
        # PDF REPORT
        # =========================================================
        st.markdown('<div class="section-header">📄 Generate Report</div>', unsafe_allow_html=True)

        if not REPORTLAB_AVAILABLE:
            st.warning("Install `reportlab` to enable PDF export:  `pip install reportlab`")
        else:
            def generate_pdf(patient_id, patient_name, age, gender, exam_date,
                             referring, dataset, predicted_label, confidence,
                             non_seizure_prob, seizure_prob, image_pil):

                buf = io.BytesIO()
                doc = SimpleDocTemplate(
                    buf, pagesize=A4,
                    leftMargin=20*mm, rightMargin=20*mm,
                    topMargin=20*mm, bottomMargin=20*mm
                )

                styles = getSampleStyleSheet()
                title_style = ParagraphStyle(
                    "title", parent=styles["Heading1"],
                    fontSize=16, textColor=colors.HexColor("#003366"),
                    alignment=TA_CENTER, spaceAfter=4
                )
                subtitle_style = ParagraphStyle(
                    "subtitle", parent=styles["Normal"],
                    fontSize=9, textColor=colors.grey,
                    alignment=TA_CENTER, spaceAfter=12
                )
                section_style = ParagraphStyle(
                    "section", parent=styles["Heading2"],
                    fontSize=11, textColor=colors.HexColor("#003366"),
                    spaceBefore=12, spaceAfter=4
                )
                body_style = ParagraphStyle(
                    "body", parent=styles["Normal"],
                    fontSize=9, leading=14
                )

                story = []

                # Header
                story.append(Paragraph("Epileptic Seizure Detection Report", title_style))
                story.append(Paragraph("MSSA-Net · Wavelet Scalogram Analysis", subtitle_style))
                story.append(HRFlowable(width="100%", thickness=1.5,
                                         color=colors.HexColor("#003366")))
                story.append(Spacer(1, 8*mm))

                # Patient info table
                story.append(Paragraph("Patient Information", section_style))
                patient_data = [
                    ["Patient ID",         str(patient_id)   or "—"],
                    ["Patient Name",        str(patient_name) or "—"],
                    ["Age",                 str(age)],
                    ["Gender",              str(gender)],
                    ["Examination Date",    str(exam_date)],
                    ["Referring Clinician", str(referring)    or "—"],
                ]
                pt = Table(patient_data, colWidths=[55*mm, 110*mm])
                pt.setStyle(TableStyle([
                    ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#e8f0fe")),
                    ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE",    (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 0), (-1, -1),
                     [colors.white, colors.HexColor("#f7f9ff")]),
                    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#ccddee")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING",  (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.append(pt)
                story.append(Spacer(1, 6*mm))

                # Prediction summary table
                story.append(Paragraph("Prediction Summary", section_style))
                result_color = colors.HexColor("#cc0000") if predicted_label == "Seizure" \
                               else colors.HexColor("#007700")
                pred_data = [
                    ["Dataset Used",      dataset],
                    ["Prediction",        predicted_label],
                    ["Confidence",        f"{confidence:.2f}%"],
                    ["Non-Seizure Prob.", f"{non_seizure_prob:.4f}"],
                    ["Seizure Prob.",     f"{seizure_prob:.4f}"],
                ]
                prt = Table(pred_data, colWidths=[55*mm, 110*mm])
                prt.setStyle(TableStyle([
                    ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#e8f0fe")),
                    ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE",    (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 0), (-1, -1),
                     [colors.white, colors.HexColor("#f7f9ff")]),
                    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#ccddee")),
                    ("TEXTCOLOR",   (1, 1), (1, 1), result_color),
                    ("FONTNAME",    (1, 1), (1, 1), "Helvetica-Bold"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING",  (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.append(prt)
                story.append(Spacer(1, 6*mm))

                # Scalogram image
                story.append(Paragraph("Uploaded Wavelet Scalogram", section_style))
                img_buf = io.BytesIO()
                image_pil.save(img_buf, format="PNG")
                img_buf.seek(0)
                rl_img = RLImage(img_buf, width=80*mm, height=80*mm)
                story.append(rl_img)
                story.append(Spacer(1, 6*mm))

                # Clinical interpretation
                story.append(Paragraph("Clinical Interpretation", section_style))
                if predicted_label == "Seizure":
                    interp = (
                        "The MSSA-Net model predicts <b>seizure activity</b> in the uploaded "
                        "wavelet scalogram with a confidence of <b>{:.2f}%</b>.<br/><br/>"
                        "<b>Recommendations:</b><br/>"
                        "• Immediate neurological assessment advised<br/>"
                        "• Review full EEG recording for confirmation<br/>"
                        "• Correlate with patient's clinical history<br/>"
                        "• Neurologist consultation recommended<br/>"
                        "• Consider anti-epileptic medication review"
                    ).format(confidence)
                else:
                    interp = (
                        "The MSSA-Net model predicts <b>no seizure activity</b> in the uploaded "
                        "wavelet scalogram with a confidence of <b>{:.2f}%</b>.<br/><br/>"
                        "<b>Recommendations:</b><br/>"
                        "• No seizure activity detected in this scalogram<br/>"
                        "• Continue routine monitoring as clinically indicated<br/>"
                        "• Clinical correlation with patient symptoms advised<br/>"
                        "• Repeat EEG if symptoms persist"
                    ).format(confidence)

                story.append(Paragraph(interp, body_style))
                story.append(Spacer(1, 8*mm))

                # Disclaimer
                story.append(HRFlowable(width="100%", thickness=0.8, color=colors.grey))
                disclaimer = ParagraphStyle(
                    "disclaimer", parent=styles["Normal"],
                    fontSize=7.5, textColor=colors.grey,
                    alignment=TA_CENTER, spaceBefore=4
                )
                story.append(Paragraph(
                    "This report is generated by an AI-assisted research prototype (MSSA-Net) "
                    "for academic demonstration purposes only. It does not constitute a clinical "
                    "diagnosis and must not replace professional medical evaluation.",
                    disclaimer
                ))

                doc.build(story)
                buf.seek(0)
                return buf.read()

            pdf_bytes = generate_pdf(
                patient_id, patient_name, age, gender, exam_date,
                referring, dataset, predicted_label, confidence,
                non_seizure_prob, seizure_prob, uploaded_image
            )

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"seizure_report_{patient_id or 'patient'}_{exam_date}.pdf",
                mime="application/pdf",
                type="primary"
            )
            st.caption("The PDF includes patient details, scalogram image, prediction, and clinical interpretation.")