import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os, pickle, json, tempfile, time

# Page config — must be first
st.set_page_config(
    page_title = "MindCare AI",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Imports ────────────────────────────────────────────────────────
from model_loader    import load_all_models, predict_text_v2
from explainability  import display_full_explanation, explain_fusion
from audio_processor import process_audio_for_model, transcribe_audio, extract_raw_features_for_shap
from conversation    import ConversationManager
from firebase_manager import (
    init_firebase, create_user, get_user, get_all_users,
    save_session, get_sessions, get_trend_data,
    compute_trajectory, detect_anomaly,
    map_to_phq9, get_weekly_summary
)
from pdf_report import generate_pdf_report

DEVICE = torch.device("cpu")

# ── CSS Styling ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #4CAF50 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #1E2329;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .chat-message-user {
        background: #1a237e;
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 12px 12px 0 12px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-message-ai {
        background: #1E2329;
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 12px 12px 12px 0;
        margin: 0.5rem 0;
        max-width: 80%;
        border-left: 3px solid #4CAF50;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "models_loaded"        : False,
        "models"               : None,
        "db"                   : None,
        "user_id"              : None,
        "user_name"            : None,
        "user_age"             : 0,
        "user_gender"          : "",
        "conversation"         : None,
        "chat_history"         : [],
        "analysis_done"        : False,
        "latest_result"        : None,
        "groq_key"             : "",
        "mode"                 : "chat",
        "page"                 : "home",
        "pdf_bytes"            : None,
        # Fusion results survive button reruns (Bug 6 fix)
        "fusion_ran"           : False,
        "fusion_text_input"    : None,
        "fusion_audio_raw"     : None,
        "fusion_beh_raw"       : None,
        "fusion_tv2"           : None,
        "fusion_audio_conf"    : None,
        "fusion_beh_answers"   : None,
        # Rich data for clinical PDF
        "last_audio_conf"      : None,
        "last_beh_answers"     : None,
        # Chat audio dedup (Bug 1 fix)
        "last_chat_audio_name" : None,
        "last_chat_audio_raw"  : None,
        # Inline SHAP cache for audio/behavioral tabs
        "audio_shap_values"    : None,
        "audio_feat_names"     : None,
        "audio_shap_err"       : None,
        "beh_shap_values"      : None,
        "beh_feat_names"       : None,
        "beh_shap_err"         : None,
        "beh_raw_values"       : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ── Load Models (cached) ───────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_all_models()


# ══════════════════════════════════════════════════════════════════
# EMBEDDING HELPERS
# All three helpers call get_embeddings() — NOT forward()
# forward() returns [1,2] logits — wrong for fusion
# get_embeddings() returns hidden state (128/768/32-dim) — correct
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_text_embedding(models, text: str):
    """Returns [1, 768] text embedding for fusion."""
    tokenizer  = models["tokenizer"]
    text_model = models["text_model"]
    enc = tokenizer(
        text, max_length=128,
        padding="max_length", truncation=True,
        return_tensors="pt"
    )
    emb = text_model.get_embeddings(
        enc["input_ids"], enc["attention_mask"]
    )
    return emb.numpy()   # [1, 768]


@torch.no_grad()
def get_beh_embedding(models, features: np.ndarray):
    """Returns [1, 32] behavioral embedding for fusion."""
    beh_model = models["beh_model"]
    sd_scaler  = models["sd_scaler"]
    scaled     = sd_scaler.transform(features.reshape(1, -1))
    tensor     = torch.tensor(scaled, dtype=torch.float32)
    emb        = beh_model.get_embeddings(tensor)
    return emb.numpy()   # [1, 32]


@torch.no_grad()
def get_audio_embedding(models, audio_path: str):
    """
    Returns ([1, 128] audio embedding for fusion, confidence float).

    FIX: process_audio_for_model() returns [1,2] logits — WRONG for fusion.
    This calls get_embeddings() which returns the 128-dim hidden state.
    Without this fix the fusion audio arm is silently zeroed out and
    the model ignores audio entirely.
    """
    raw = extract_raw_features_for_shap(audio_path, models["audio_config"])
    if raw is None:
        return None, 0.0

    seq_len    = models["audio_config"].get("max_frames", 512)
    input_dim  = models["audio_config"].get("input_dim", 79)
    # Scaler was trained on input_dim features — truncate/pad to match
    if raw.shape[1] > input_dim:
        raw = raw[:, :input_dim]
    elif raw.shape[1] < input_dim:
        raw = np.pad(raw, ((0, 0), (0, input_dim - raw.shape[1])))
    scaled = models["audio_scaler"].transform(raw)
    tensor  = torch.tensor(
        np.repeat(scaled[:, np.newaxis, :], seq_len, axis=1),
        dtype=torch.float32
    )   # [1, seq_len, input_dim]

    emb  = models["audio_model"].get_embeddings(tensor)          # [1, 128] ✅
    conf = torch.softmax(
        models["audio_model"](tensor), dim=1
    )[0][1].item()
    return emb.numpy(), conf


# ══════════════════════════════════════════════════════════════════
# FUSION HELPER
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_fusion(models, audio_emb=None, text_emb=None, beh_emb=None):
    """
    Multimodal fusion with clinical override rules.

    Expected embedding dimensions:
      audio_emb : [1, 128]  from get_audio_embedding()
      text_emb  : [1, 768]  from get_text_embedding()
      beh_emb   : [1,  32]  from get_beh_embedding()

    Clinical override rules (safety layer):
      Rule 1 — Text says Severe >85%  → risk ≥ 0.80
      Rule 2 — Text says Moderate >70% → risk ≥ 0.45
      Rule 3 — Suicidal thoughts = Yes → risk ≥ 0.75
    """
    AUDIO_DIM = 128
    TEXT_DIM  = 768
    BEH_DIM   = 32

    def to_tensor(emb, expected_dim):
        """Convert embedding to tensor, validate dimension."""
        if emb is None:
            return None
        if isinstance(emb, np.ndarray):
            emb = torch.tensor(emb, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        # Dimension mismatch means wrong method was called
        if emb.shape[-1] != expected_dim:
            return None
        return emb

    audio_t = to_tensor(audio_emb, AUDIO_DIM)
    text_t  = to_tensor(text_emb,  TEXT_DIM)
    beh_t   = to_tensor(beh_emb,   BEH_DIM)

    # Missing-modality mask
    mask = torch.tensor([[
        1.0 if audio_t is not None else 0.0,
        1.0 if text_t  is not None else 0.0,
        1.0 if beh_t   is not None else 0.0,
    ]], dtype=torch.float32)

    # Zero-fill missing modalities
    audio_in = audio_t if audio_t is not None else torch.zeros(1, AUDIO_DIM)
    text_in  = text_t  if text_t  is not None else torch.zeros(1, TEXT_DIM)
    beh_in   = beh_t   if beh_t   is not None else torch.zeros(1, BEH_DIM)

    fusion_model = models["fusion_model"]
    fusion_model.eval()

    logits, weights = fusion_model(audio_in, text_in, beh_in, mask)
    probs           = torch.softmax(logits, dim=1)
    risk_score      = probs[0][1].item()
    weights_np      = weights[0].numpy()

    # ── Clinical override rules ───────────────────────────────────
    # Rule 1 & 2: text model confidence overrides weak fusion
    _tv2 = (
        st.session_state.get("text_v2_result") or
        st.session_state.get("manual_text_v2_result")
    )
    if _tv2 and text_emb is not None:
        text_raw_prob = _tv2.get("raw_prob", 0)
        text_severity = _tv2.get("severity", "")
        if text_raw_prob > 0.85 and text_severity == "Severe":
            risk_score = max(risk_score, 0.80)
        elif text_raw_prob > 0.70 and text_severity in ("Moderate", "Severe"):
            risk_score = max(risk_score, 0.45)

    # Rule 3: suicidal thoughts → minimum High Risk
    beh_raw = st.session_state.get("manual_beh_raw")
    if beh_raw is not None:
        beh_arr = np.array(beh_raw).flatten()
        if len(beh_arr) > 9 and beh_arr[9] >= 1:  # index 9 = suicidal thoughts
            risk_score = max(risk_score, 0.75)

    # Risk level mapping
    if risk_score >= 0.70:
        risk_level = "High Risk"
        color      = "#F44336"
    elif risk_score >= 0.40:
        risk_level = "Moderate Risk"
        color      = "#FF9800"
    else:
        risk_level = "Low Risk"
        color      = "#4CAF50"

    modalities = []
    if audio_t is not None: modalities.append("Audio")
    if text_t  is not None: modalities.append("Text")
    if beh_t   is not None: modalities.append("Behavioral")

    return {
        "risk_score" : risk_score,
        "risk_level" : risk_level,
        "color"      : color,
        "confidence" : float(max(probs[0]).item()),
        "weights"    : weights_np,
        "modalities" : ", ".join(modalities) if modalities else "None",
    }


# ── Display Results ────────────────────────────────────────────────
def display_results(result, phq9_data=None, show_weights=True):
    has_behavioral = "Behavioral" in result.get("modalities", "")

    _cols = st.columns(3) if has_behavioral else st.columns(2)
    col1, col2 = _cols[0], _cols[1]
    col3 = _cols[2] if len(_cols) > 2 else None

    with col1:
        st.markdown(f"""
        <div class="risk-card" style="background:{result['color']}">
            Risk Level: {result['risk_level']}<br>
            <span style="font-size:2rem">{result['risk_score']:.0%}</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="risk-card" style="background:#1a237e">
            Confidence<br>
            <span style="font-size:2rem">{result['confidence']:.0%}</span><br>
            <span style="font-size:0.9rem">Modalities: {result['modalities']}</span>
        </div>
        """, unsafe_allow_html=True)

    if has_behavioral and phq9_data and col3 is not None:
        with col3:
            st.markdown(f"""
            <div class="risk-card" style="background:{phq9_data['color']}">
                PHQ-9 Equivalent<br>
                <span style="font-size:2rem">{phq9_data['phq9_score']}/27</span><br>
                <span style="font-size:0.9rem">{phq9_data['severity']}</span>
            </div>
            """, unsafe_allow_html=True)
    elif phq9_data:
        st.caption("Note: PHQ-9 equivalent is only shown when behavioral questionnaire is completed.")

    if show_weights:
        weights = result["weights"]
        fig = go.Figure(go.Bar(
            x            = ["Audio", "Text", "Behavioral"],
            y            = weights,
            marker_color = ["#2196F3", "#4CAF50", "#FF9800"],
            text         = [f"{w:.1%}" for w in weights],
            textposition = "outside"
        ))
        fig.update_layout(
            title         = "Modality Attention Weights",
            yaxis_range   = [0, 1.1],
            height        = 250,
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font_color    = "white"
        )
        st.plotly_chart(fig, use_container_width=True)

    if phq9_data:
        st.info(f"💡 **Clinical Advice:** {phq9_data['advice']}")


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 MindCare AI")
    st.markdown("---")

    with st.expander("⚙️ Configuration", expanded=not st.session_state.models_loaded):
        groq_key    = st.text_input("Groq API Key",
                                     type="password",
                                     value=st.session_state.groq_key)
        firebase_ok = os.path.exists("firebase-key.json")
        if firebase_ok:
            st.success("✅ Firebase key found")
        else:
            st.error("❌ firebase-key.json not found")

        if st.button("🚀 Load Models", use_container_width=True):
            with st.spinner("Loading AI models... (~2 mins first time)"):
                try:
                    st.session_state.models        = get_models()
                    st.session_state.models_loaded = True
                    st.session_state.groq_key      = groq_key
                    if firebase_ok:
                        st.session_state.db = init_firebase()
                    st.success("✅ Models loaded!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 📍 Navigation")
    pages = {
        "🏠 Home"            : "home",
        "💬 Chat Screen"     : "chat",
        "📊 Manual Analysis" : "manual",
        "📈 My Progress"     : "progress",
    }
    for label, page_id in pages.items():
        if st.button(label, use_container_width=True):
            st.session_state.page = page_id

    st.markdown("---")

    # ── User Profile — always visible, not gated on Firebase ──────
    # Bug fix: was inside `if st.session_state.db:` so hidden when
    # Firebase not configured — user had no way to set their identity
    with st.expander("👤 User Profile", expanded=not bool(st.session_state.user_id)):
        if not st.session_state.user_id:
            st.markdown("**Set up your profile to save progress**")
            name    = st.text_input("Your Name",              key="profile_name")
            age     = st.number_input("Age", 10, 100, 25,    key="profile_age")
            gender  = st.selectbox("Gender",
                                   ["Male", "Female", "Other"], key="profile_gender")
            user_id = st.text_input("User ID (email or username)", key="profile_uid")

            if st.button("💾 Save Profile", use_container_width=True):
                if name and user_id:
                    # Save to Firebase if available
                    if st.session_state.db:
                        try:
                            create_user(st.session_state.db, user_id, name, age, gender)
                        except Exception as e:
                            st.error(f"Firebase error: {e}")

                    # Always store in session state (Bug fix: age/gender were missing)
                    st.session_state.user_id     = user_id
                    st.session_state.user_name   = name
                    st.session_state.user_age    = int(age)
                    st.session_state.user_gender = gender
                    st.rerun()   # Bug fix: rerun so expander updates to show profile
                else:
                    st.warning("Please fill in both Name and User ID.")
        else:
            st.success(f"👤 **{st.session_state.user_name}**")
            st.caption(f"ID: {st.session_state.user_id}  ·  "
                       f"Age: {st.session_state.user_age}  ·  "
                       f"{st.session_state.user_gender}")
            if not st.session_state.db:
                st.warning("⚠️ Firebase not configured — progress won't be saved to cloud.")
            if st.button("🔄 Switch User", use_container_width=True):
                st.session_state.user_id     = None
                st.session_state.user_name   = None
                st.session_state.user_age    = 0
                st.session_state.user_gender = ""
                st.rerun()


# ══════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 MindCare AI</h1>
        <p style="font-size:1.2rem">Explainable Multimodal Depression Screening</p>
        <p>Audio • Text • Behavioral Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>💬 Chat Mode</h3>
            <p>Have a natural conversation with our AI.
            It will gently assess your mental health
            through empathetic dialogue.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Manual Mode</h3>
            <p>Upload audio, enter text, or fill the
            behavioral questionnaire for a detailed
            multimodal analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Progress Tracking</h3>
            <p>Monitor your mental health over time
            with trend analysis, anomaly alerts,
            and PDF reports for your doctor.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    perf_data = {
        "Model"    : ["Audio Only", "Text Only (v2)", "Behavioral",
                      "Basic Fusion", "MM Fusion (Ours)"],
        "F1 Score" : [0.76, 0.9917, 0.84, 0.76, 0.88],
        "AUC-ROC"  : [0.79, 1.00,   0.92, 0.81, 0.95],
    }
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="F1 Score", x=perf_data["Model"],
        y=perf_data["F1 Score"], marker_color="#4CAF50"
    ))
    fig.add_trace(go.Bar(
        name="AUC-ROC", x=perf_data["Model"],
        y=perf_data["AUC-ROC"], marker_color="#2196F3"
    ))
    fig.update_layout(
        barmode       = "group",
        height        = 300,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font_color    = "white",
        yaxis_range   = [0.5, 1.05]
    )
    st.plotly_chart(fig, use_container_width=True)

    if not st.session_state.models_loaded:
        st.warning("⚠️ Please configure API keys and load models from the sidebar to get started!")


# ══════════════════════════════════════════════════════════════════
# CHAT PAGE
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == "chat":
    st.markdown("## 💬 Conversational Screening")

    if not st.session_state.models_loaded:
        st.warning("Please load models from the sidebar first!")
        st.stop()

    if not st.session_state.groq_key:
        st.error("Please enter your Groq API key in the sidebar!")
        st.stop()

    # Always define models at top of page — fixes Firebase save bug
    # where models was only defined inside button block and undefined on re-render
    models = st.session_state.models

    if st.session_state.conversation is None:
        st.session_state.conversation = ConversationManager(st.session_state.groq_key)
        opening = st.session_state.conversation.chat("Hello, I'm here to chat.")
        st.session_state.chat_history.append({"role": "assistant", "content": opening})

    # ── 1. CHAT HISTORY (top) ──────────────────────────────────────
    st.markdown("#### 💬 Conversation")
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("Start a conversation below. Share how you're feeling.")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message-user">👤 {msg['content']}</div>
                """, unsafe_allow_html=True)
            else:
                clean = msg["content"].replace("[READY_FOR_ANALYSIS]", "").strip()
                st.markdown(f"""
                <div class="chat-message-ai">🧠 {clean}</div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. INPUT BOX (middle) ──────────────────────────────────────
    st.markdown("#### ✍️ Your Message")
    input_method = st.radio("Input method:", ["⌨️ Type", "🎤 Upload Voice"], horizontal=True)

    user_input = None

    if input_method == "⌨️ Type":
        col1, col2 = st.columns([5, 1])
        with col1:
            typed = st.text_input(
                "Message:",
                placeholder="How are you feeling today?",
                key="chat_input",
                label_visibility="collapsed"
            )
        with col2:
            send = st.button("Send 📤", use_container_width=True)
        if send and typed:
            user_input = typed

    else:
        audio_file = st.file_uploader(
            "Upload voice message (WAV/MP3)",
            type=["wav", "mp3", "ogg"]
        )
        if audio_file:
            # Bug 1 fix: only process audio if it hasn't been processed yet
            # (Streamlit reruns on every interaction — without this check the
            # same file gets re-analyzed on every click including Send)
            audio_already_processed = (
                st.session_state.get("last_chat_audio_name") == audio_file.name
            )
            if not audio_already_processed:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                with st.spinner("Transcribing..."):
                    transcribed = transcribe_audio(tmp_path)

                if transcribed:
                    st.info(f"📝 Transcribed: *{transcribed}*")
                    user_input = transcribed
                    with st.spinner("Analyzing audio..."):
                        audio_emb, audio_conf = get_audio_embedding(models, tmp_path)
                        if audio_emb is not None:
                            st.session_state["last_audio_emb"]      = audio_emb
                            st.session_state["last_chat_audio_name"] = audio_file.name
                            st.session_state["last_chat_audio_raw"]  = extract_raw_features_for_shap(
                                tmp_path, models["audio_config"]
                            )
                            st.session_state["last_audio_conf"]  = audio_conf
                else:
                    st.warning("Could not transcribe audio. Please try typing.")
            else:
                st.info(f"✅ Audio already analyzed: *{audio_file.name}*")

    # Process new message
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("MindCare AI is thinking..."):
            response = st.session_state.conversation.chat(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        if st.session_state.conversation.is_ready_for_analysis():
            st.session_state["ready_for_analysis"] = True
        # Bug 5 fix: clear text box so next message starts empty
        st.session_state["chat_input"] = ""
        st.rerun()

    st.markdown("---")

    # ── 3. ACTION BUTTONS (bottom of input) ───────────────────────
    analyze_ready = (
        len(st.session_state.chat_history) >= 6 or
        st.session_state.get("ready_for_analysis", False)
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Analyze Now", disabled=not analyze_ready, use_container_width=True):
            with st.spinner("Running multimodal analysis..."):
                conv = st.session_state.conversation

                full_text = " ".join([
                    m["content"]
                    for m in st.session_state.chat_history
                    if m["role"] == "user"
                ])

                text_emb = get_text_embedding(models, full_text)

                text_v2 = predict_text_v2(
                    full_text,
                    models["text_model"],
                    models["severity_model"],
                    models["tokenizer"]
                )
                st.session_state["text_v2_result"] = text_v2

                audio_emb = st.session_state.get("last_audio_emb")
                result    = run_fusion(models, audio_emb, text_emb)
                phq9      = map_to_phq9(
                    result["risk_score"],
                    beh_features_raw=st.session_state.get("manual_beh_raw")
                )

                explanation = conv.explain_results(
                    result["risk_level"],
                    result["confidence"],
                    result["modalities"].split(", "),
                    f"PHQ-9 equivalent: {phq9['severity']}"
                )

                st.session_state.latest_result  = result
                st.session_state.analysis_done  = True
                st.session_state["last_phq9"]   = phq9
                st.session_state["explanation"] = explanation
                st.rerun()

    with col2:
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.conversation  = None
            st.session_state.chat_history  = []
            st.session_state.analysis_done = False
            st.session_state.latest_result = None
            st.session_state.pop("ready_for_analysis",    None)
            st.session_state.pop("last_audio_emb",        None)
            st.session_state.pop("text_v2_result",        None)
            # Bug 1 fix: clear audio tracking so next upload is processed fresh
            st.session_state["last_chat_audio_name"] = None
            st.session_state["last_chat_audio_raw"]  = None
            st.session_state["last_audio_conf"]      = None
            st.rerun()

    # ── 4. ANALYSIS RESULTS (shown below after analysis) ──────────
    if st.session_state.analysis_done and st.session_state.latest_result:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        result = st.session_state.latest_result
        phq9   = st.session_state["last_phq9"]

        if st.session_state.get("text_v2_result"):
            tv2 = st.session_state["text_v2_result"]
            st.markdown("### 📝 Text Analysis (MentalBERT v2)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="risk-card" style="background:{tv2['color']}">
                    {tv2['emoji']} {tv2['severity']}<br>
                    <span style="font-size:0.9rem">{tv2['raw_prob']*100:.1f}% Depression Probability</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="risk-card" style="background:#1a237e">
                    Severity Model: {tv2['sev_model']}<br>
                    <span style="font-size:0.9rem">Confidence: {tv2['sev_conf']*100:.0f}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("### 🔀 Multimodal Fusion Result")

        display_results(result, phq9)

        st.markdown("### 🤖 AI Explanation")
        st.markdown(f"""
        <div class="chat-message-ai">🧠 {st.session_state['explanation']}</div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("🔍 Why did the AI say this?", expanded=False):
            chat_text = " ".join([
                m["content"] for m in st.session_state.chat_history
                if m["role"] == "user"
            ])
            _chat_tv2 = st.session_state.get("text_v2_result")
            display_full_explanation(
                text            = chat_text,
                text_model      = models["text_model"],
                tokenizer       = models["tokenizer"],
                # Bug 2 fix: pass audio features so audio XAI shows in chat mode
                audio_features  = st.session_state.get("last_chat_audio_raw"),
                audio_model     = models["audio_model"],
                audio_scaler    = models["audio_scaler"],
                audio_config    = models["audio_config"],
                fusion_weights  = result.get("weights"),
                modalities_used = result.get("modalities"),
                risk_score      = result.get("risk_score"),
                severity_label  = _chat_tv2["severity"] if _chat_tv2 else None,
                raw_prob        = _chat_tv2["raw_prob"]  if _chat_tv2 else None,
            )

        # Firebase save — models is defined at top of page so always available
        if st.session_state.db and st.session_state.user_id:
            if st.button("💾 Save to My Progress"):
                try:
                    save_session(
                        st.session_state.db,
                        st.session_state.user_id,
                        {
                            "risk_score" : result["risk_score"],
                            "risk_level" : result["risk_level"],
                            "confidence" : result["confidence"],
                            "modalities" : result["modalities"],
                            "phq9_score" : phq9["phq9_score"],
                            "severity"   : phq9["severity"],
                            "source"     : "chat"
                        }
                    )
                    st.success("✅ Saved to your progress tracker!")
                except Exception as e:
                    st.error(f"❌ Save failed: {e}")
        elif not st.session_state.db:
            st.warning("⚠️ Firebase not configured — add firebase-key.json to save progress.")
        elif not st.session_state.user_id:
            st.warning("👤 Set up your **User Profile** in the sidebar to save progress.")


# ══════════════════════════════════════════════════════════════════
# MANUAL ANALYSIS PAGE
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == "manual":
    st.markdown("## 📊 Manual Multimodal Analysis")

    if not st.session_state.models_loaded:
        st.warning("Please load models from the sidebar first!")
        st.stop()

    # Allow user to clear previous results and start a fresh analysis
    if st.session_state.get("fusion_ran"):
        if st.button("🔄 Start New Analysis", key="clear_fusion"):
            for k in ["fusion_ran","fusion_text_input","fusion_audio_raw",
                      "fusion_beh_raw","fusion_tv2","fusion_audio_conf",
                      "fusion_beh_answers","pdf_bytes","last_beh_answers",
                      "last_audio_conf","text_v2_result",
                      "audio_shap_values","audio_feat_names","audio_shap_err",
                      "beh_shap_values","beh_feat_names","beh_shap_err","beh_raw_values"]:
                st.session_state.pop(k, None)
            st.session_state["fusion_ran"] = False
            st.rerun()

    tab1, tab2, tab3 = st.tabs([
        "🎤 Audio Analysis",
        "📝 Text Analysis",
        "📋 Behavioral Questionnaire"
    ])

    audio_emb_manual = None
    text_emb_manual  = None
    beh_emb_manual   = None

    # ── Audio Tab ─────────────────────────────────────────────────
    with tab1:
        st.markdown("### 🎤 Upload Audio Recording")
        st.info("Upload a voice recording (WAV/MP3). The AI will analyze speech patterns for depression indicators.")

        audio_file = st.file_uploader(
            "Choose audio file",
            type=["wav", "mp3", "ogg"],
            key="manual_audio"
        )

        if audio_file:
            st.audio(audio_file)
            if st.button("Analyze Audio 🔍"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                models = st.session_state.models

                with st.spinner("Extracting audio features..."):
                    emb, conf = get_audio_embedding(models, tmp_path)

                if emb is not None:
                    audio_emb_manual = emb
                    st.session_state["manual_audio_emb"] = emb
                    st.session_state["last_audio_conf"]  = conf   # ← FIX: was missing

                    raw_feats = extract_raw_features_for_shap(
                        tmp_path, models["audio_config"]
                    )
                    st.session_state["manual_audio_raw"] = raw_feats

                    # Compute SHAP values now and cache for display
                    if raw_feats is not None:
                        from explainability import explain_audio_shap
                        shap_vals, feat_names, shap_err = explain_audio_shap(
                            raw_feats, models["audio_model"],
                            models["audio_scaler"], models["audio_config"]
                        )
                        st.session_state["audio_shap_values"] = shap_vals
                        st.session_state["audio_feat_names"]  = feat_names
                        st.session_state["audio_shap_err"]    = shap_err

                    st.success(f"✅ Audio analyzed! Depression confidence: {conf:.1%}")

                    with st.spinner("Transcribing..."):
                        text = transcribe_audio(tmp_path)
                    if text:
                        st.markdown(f"**📝 Transcript:** {text}")
                else:
                    st.error("Could not process audio file.")

        # Show SHAP results if already computed (persists across reruns)
        if st.session_state.get("audio_shap_values") is not None:
            st.markdown("---")
            st.markdown("#### 🎤 Audio Feature Importance (SHAP)")
            from explainability import display_audio_explanation
            display_audio_explanation(
                st.session_state["audio_shap_values"],
                st.session_state["audio_feat_names"]
            )
        elif st.session_state.get("audio_shap_err"):
            st.warning(f"SHAP unavailable: {st.session_state['audio_shap_err']}")

    # ── Text Tab ──────────────────────────────────────────────────
    with tab2:
        st.markdown("### 📝 Text Analysis")
        st.info("Enter any text — journal entry, social media post, or description of how you're feeling.")

        text_input = st.text_area(
            "Enter text:",
            height=200,
            placeholder="Write about how you've been feeling lately..."
        )

        if text_input and st.button("Analyze Text 🔍"):
            with st.spinner("Analyzing text with MentalBERT v2..."):
                models    = st.session_state.models
                text_emb  = get_text_embedding(models, text_input)
                st.session_state["manual_text_emb"]  = text_emb
                st.session_state["manual_last_text"] = text_input

                result_v2 = predict_text_v2(
                    text_input,
                    models["text_model"],
                    models["severity_model"],
                    models["tokenizer"]
                )
                # Store for fusion clinical override
                st.session_state["text_v2_result"] = result_v2

            st.success("✅ Text analyzed!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="risk-card" style="background:{result_v2['color']}">
                    {result_v2['emoji']} {result_v2['severity']}<br>
                    <span style="font-size:0.9rem">Depression Severity</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="risk-card" style="background:#1a237e">
                    {result_v2['raw_prob']*100:.1f}%<br>
                    <span style="font-size:0.9rem">Depression Probability</span>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                sev_colors = {
                    "Low": "#FFC107", "Moderate": "#FF9800", "Severe": "#F44336"
                }
                sev_col = sev_colors.get(result_v2['sev_model'], "#9E9E9E")
                st.markdown(f"""
                <div class="risk-card" style="background:{sev_col}">
                    {result_v2['sev_model']}<br>
                    <span style="font-size:0.9rem">
                        Severity Model ({result_v2['sev_conf']*100:.0f}% conf)
                    </span>
                </div>
                """, unsafe_allow_html=True)

            advice_map = {
                "Not Depressed" : "✅ No significant depression indicators detected. Keep maintaining healthy habits!",
                "Low"           : "🟡 Mild indicators present. Consider mindfulness, exercise, and talking to someone you trust.",
                "Moderate"      : "🟠 Moderate depression indicators. Consider speaking with a mental health professional.",
                "Severe"        : "🔴 Severe depression indicators detected. Please reach out to a mental health professional immediately.",
            }
            st.info(f"💡 **Clinical Guidance:** {advice_map[result_v2['severity']]}")

            st.markdown("---")
            with st.expander("🔍 Explain this prediction", expanded=True):
                display_full_explanation(
                    text           = text_input,
                    text_model     = models["text_model"],
                    tokenizer      = models["tokenizer"],
                    severity_label = result_v2["severity"],
                    raw_prob       = result_v2["raw_prob"],
                )

    # ── Behavioral Tab ────────────────────────────────────────────
    with tab3:
        st.markdown("### 📋 Behavioral Questionnaire")
        st.info("Answer these questions about your lifestyle and behavioral patterns.")

        col1, col2 = st.columns(2)
        with col1:
            gender  = st.selectbox("Gender", ["Male", "Female"])
            age     = st.slider("Age", 15, 60, 25)
            acad_p  = st.slider("Academic Pressure (1-5)", 1, 5, 3)
            work_p  = st.slider("Work Pressure (1-5)", 1, 5, 2)
            cgpa    = st.slider("CGPA / GPA", 0.0, 10.0, 7.0)
            study_s = st.slider("Study Satisfaction (1-5)", 1, 5, 3)

        with col2:
            job_s    = st.slider("Job Satisfaction (1-5)", 1, 5, 3)
            sleep    = st.selectbox(
                "Sleep Duration",
                ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
            )
            diet     = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"])
            suicidal = st.selectbox("Suicidal Thoughts?", ["No", "Yes"])
            work_hrs = st.slider("Work/Study Hours/day", 0, 16, 8)
            fin_str  = st.slider("Financial Stress (1-5)", 1, 5, 3)
            fam_hist = st.selectbox("Family History of Mental Illness?", ["No", "Yes"])

        if st.button("Analyze Behavioral Data 🔍"):
            sleep_map = {
                "Less than 5 hours" : 4,
                "5-6 hours"         : 5.5,
                "7-8 hours"         : 7.5,
                "More than 8 hours" : 9
            }
            features = np.array([[
                1 if gender == "Female" else 0,
                age, acad_p, work_p, cgpa, study_s, job_s,
                sleep_map[sleep],
                {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}[diet],
                1 if suicidal == "Yes" else 0,
                work_hrs, fin_str,
                1 if fam_hist == "Yes" else 0
            ]], dtype=np.float32)

            with st.spinner("Analyzing behavioral data..."):
                models         = st.session_state.models
                beh_emb        = get_beh_embedding(models, features)
                beh_emb_manual = beh_emb
                st.session_state["manual_beh_emb"] = beh_emb
                st.session_state["manual_beh_raw"] = features

                # FIX: store human-readable answers for PDF (was never stored before)
                st.session_state["last_beh_answers"] = {
                    "gender"            : gender,
                    "age"               : age,
                    "academic_pressure" : acad_p,
                    "work_pressure"     : work_p,
                    "cgpa"              : cgpa,
                    "study_satisfaction": study_s,
                    "job_satisfaction"  : job_s,
                    "sleep"             : sleep,
                    "diet"              : diet,
                    "suicidal"          : suicidal,
                    "work_hours"        : work_hrs,
                    "financial_stress"  : fin_str,
                    "family_history"    : fam_hist,
                }

                # Compute SHAP and cache for inline display
                from explainability import explain_behavioral_shap
                shap_vals, feat_names, shap_err = explain_behavioral_shap(
                    features, models["beh_model"], models["sd_scaler"]
                )
                st.session_state["beh_shap_values"] = shap_vals
                st.session_state["beh_feat_names"]  = feat_names
                st.session_state["beh_shap_err"]    = shap_err
                st.session_state["beh_raw_values"]  = features
                st.success("✅ Behavioral data analyzed!")

        # Show SHAP inline (persists across reruns)
        if st.session_state.get("beh_shap_values") is not None:
            st.markdown("---")
            from explainability import display_behavioral_explanation
            display_behavioral_explanation(
                st.session_state["beh_shap_values"],
                st.session_state["beh_feat_names"],
                st.session_state["beh_raw_values"]
            )
        elif st.session_state.get("beh_shap_err"):
            st.warning(f"SHAP unavailable: {st.session_state['beh_shap_err']}")

    # ── Multimodal Fusion ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔀 Multimodal Fusion")

    available = []
    if st.session_state.get("manual_audio_emb") is not None: available.append("🎤 Audio")
    if st.session_state.get("manual_text_emb")  is not None: available.append("📝 Text")
    if st.session_state.get("manual_beh_emb")   is not None: available.append("📋 Behavioral")

    if available:
        st.success(f"Ready to fuse: {', '.join(available)}")
    else:
        st.info("Analyze at least one modality above to enable fusion.")

    if available and st.button("🚀 Run Fusion Analysis", use_container_width=True):
        with st.spinner("Running fusion..."):
            models = st.session_state.models
            result = run_fusion(
                models,
                st.session_state.get("manual_audio_emb"),
                st.session_state.get("manual_text_emb"),
                st.session_state.get("manual_beh_emb")
            )
            phq9 = map_to_phq9(
                result["risk_score"],
                beh_features_raw=st.session_state.get("manual_beh_raw")
            )
            # Store everything needed in session state BEFORE any rerun
            st.session_state.latest_result             = result
            st.session_state["last_phq9"]              = phq9
            st.session_state["fusion_ran"]             = True
            # Snapshot the text/audio/beh data for PDF (keys get cleared below)
            st.session_state["fusion_text_input"]      = st.session_state.get("manual_last_text")
            st.session_state["fusion_audio_raw"]       = st.session_state.get("manual_audio_raw")
            st.session_state["fusion_beh_raw"]         = st.session_state.get("manual_beh_raw")
            st.session_state["fusion_tv2"]             = st.session_state.get("text_v2_result")
            st.session_state["fusion_audio_conf"]      = st.session_state.get("last_audio_conf")
            st.session_state["fusion_beh_answers"]     = st.session_state.get("last_beh_answers")

        # Clear embeddings immediately so next analysis starts fresh
        # (Bug 12 fix: stale audio_emb caused audio to always dominate weights)
        for key in ["manual_audio_emb", "manual_text_emb", "manual_beh_emb",
                     "manual_audio_raw", "manual_beh_raw", "manual_last_text"]:
            st.session_state.pop(key, None)

    # ── Results — OUTSIDE the button block so they survive reruns ──
    # (Bug 6 fix: previously inside if-block, so PDF/Save buttons wiped everything)
    if st.session_state.get("fusion_ran") and st.session_state.latest_result:
        models = st.session_state.models
        result = st.session_state.latest_result
        phq9   = st.session_state["last_phq9"]
        _tv2   = st.session_state.get("fusion_tv2")

        st.markdown("## 📊 Results")
        display_results(result, phq9)

        st.markdown("---")
        with st.expander("🔍 Explainability Report", expanded=True):
            display_full_explanation(
                text            = st.session_state.get("fusion_text_input"),
                text_model      = models["text_model"],
                tokenizer       = models["tokenizer"],
                audio_features  = st.session_state.get("fusion_audio_raw"),
                audio_model     = models["audio_model"],
                audio_scaler    = models["audio_scaler"],
                audio_config    = models["audio_config"],
                beh_features    = st.session_state.get("fusion_beh_raw"),
                beh_model       = models["beh_model"],
                sd_scaler       = models["sd_scaler"],
                fusion_weights  = result["weights"],
                modalities_used = result["modalities"],
                risk_score      = result["risk_score"],
                severity_label  = _tv2["severity"] if _tv2 else None,
                raw_prob        = _tv2["raw_prob"]  if _tv2 else result["risk_score"],
            )

        # ── Clinical PDF ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📄 Clinical Report")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("📥 Generate Clinical PDF", use_container_width=True, key="manual_pdf_gen"):
                with st.spinner("Generating clinical report..."):
                    try:
                        pdf_bytes = generate_pdf_report(
                            user_name      = st.session_state.get("user_name", "Anonymous"),
                            user_age       = st.session_state.get("user_age",  0),
                            user_gender    = st.session_state.get("user_gender",""),
                            fusion_result  = result,
                            phq9_data      = phq9,
                            text_input     = st.session_state.get("fusion_text_input"),
                            text_v2_result = _tv2,
                            audio_conf     = st.session_state.get("fusion_audio_conf"),
                            beh_answers    = st.session_state.get("fusion_beh_answers"),
                            report_source  = "manual",
                        )
                        st.session_state["pdf_bytes"] = pdf_bytes
                        st.success("✅ Clinical report ready! Click Download.")
                    except Exception as e:
                        st.error(f"❌ PDF failed: {e}")

        if st.session_state.get("pdf_bytes"):
            with col_r2:
                st.download_button(
                    label               = "⬇️ Download Clinical Report",
                    data                = st.session_state["pdf_bytes"],
                    file_name           = f"mindcare_clinical_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime                = "application/pdf",
                    use_container_width = True,
                    key                 = "manual_pdf_dl"
                )

        # ── Save to Firebase ──────────────────────────────────────
        if st.session_state.db and st.session_state.user_id:
            if st.button("💾 Save to My Progress"):
                try:
                    save_session(
                        st.session_state.db,
                        st.session_state.user_id,
                        {
                            "risk_score"       : result["risk_score"],
                            "risk_level"       : result["risk_level"],
                            "confidence"       : result["confidence"],
                            "modalities"       : result["modalities"],
                            "phq9_score"       : phq9["phq9_score"],
                            "severity"         : phq9["severity"],
                            "source"           : "manual",
                            "text_probability" : float(_tv2["raw_prob"]) if _tv2 else None,
                            "text_severity"    : _tv2["severity"]        if _tv2 else None,
                            "audio_confidence" : st.session_state.get("fusion_audio_conf"),
                            "fusion_weights"   : [float(w) for w in result["weights"]],
                            "suicidal_flag"    : phq9.get("suicidal_flag", False),
                        }
                    )
                    st.success("✅ Saved to progress tracker!")
                except Exception as e:
                    st.error(f"❌ Save failed: {e}")


# ══════════════════════════════════════════════════════════════════
# PROGRESS PAGE
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == "progress":
    st.markdown("## 📈 Mental Health Progress Tracker")

    if not st.session_state.db:
        st.warning("⚠️ Firebase not configured — progress tracking requires firebase-key.json.")
        st.info("You can still analyze using **Manual Analysis** and **Chat** — "
                "results won't be saved to cloud without Firebase.")
        st.stop()

    if not st.session_state.user_id:
        st.warning("👤 Please set up your **User Profile** in the sidebar first!")
        st.info("Open the sidebar → scroll down → expand **👤 User Profile** → fill in your details and click Save.")
        st.stop()

    db      = st.session_state.db
    user_id = st.session_state.user_id

    dates, scores, risk_labels = get_trend_data(db, user_id)
    sessions                   = get_sessions(db, user_id)
    weekly                     = get_weekly_summary(db, user_id)

    if not scores:
        st.info("No screening sessions yet. Complete a screening to see your progress!")
        st.stop()

    trajectory = compute_trajectory(scores)
    anomaly    = detect_anomaly(scores)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions",  len(sessions))
    with col2:
        st.metric("Latest Risk",     f"{scores[-1]:.0%}" if scores else "N/A")
    with col3:
        traj_emoji = {"improving": "📈", "worsening": "📉", "stable": "➡️"}
        st.metric("Trajectory", f"{traj_emoji.get(trajectory, '?')} {trajectory.title()}")
    with col4:
        st.metric("Weekly Sessions", weekly.get("sessions", 0))

    if anomaly:
        st.warning(
            "⚠️ **Anomaly Detected!** Your latest score is significantly "
            "different from your baseline. Consider speaking with a professional."
        )

    st.markdown("---")
    view = st.radio("View:", ["All Time", "Last 30 Days", "Last 7 Days"], horizontal=True)

    if view == "Last 7 Days":
        dates_f, scores_f = dates[-7:],  scores[-7:]
    elif view == "Last 30 Days":
        dates_f, scores_f = dates[-30:], scores[-30:]
    else:
        dates_f, scores_f = dates, scores

    st.markdown("### 📊 Risk Score Over Time")
    colors_map = [
        "#F44336" if s >= 0.7 else "#FF9800" if s >= 0.4 else "#4CAF50"
        for s in scores_f
    ]

    fig = go.Figure()
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor="#F44336", opacity=0.08,
                  annotation_text="High Risk",  annotation_position="right")
    fig.add_hrect(y0=0.4, y1=0.7, fillcolor="#FF9800", opacity=0.08,
                  annotation_text="Moderate",   annotation_position="right")
    fig.add_hrect(y0=0.0, y1=0.4, fillcolor="#4CAF50", opacity=0.08,
                  annotation_text="Low Risk",   annotation_position="right")
    fig.add_trace(go.Scatter(
        x=dates_f, y=scores_f, mode="lines+markers", name="Risk Score",
        line=dict(color="#4CAF50", width=2),
        marker=dict(size=10, color=colors_map, line=dict(width=2, color="white")),
        hovertemplate="Date: %{x}<br>Risk: %{y:.1%}<extra></extra>"
    ))
    if len(scores_f) >= 3:
        ma = np.convolve(scores_f, np.ones(3)/3, mode="valid")
        fig.add_trace(go.Scatter(
            x=dates_f[2:], y=ma, mode="lines", name="3-session avg",
            line=dict(color="#2196F3", width=2, dash="dash")
        ))
    fig.update_layout(
        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", yaxis_range=[0, 1.05],
        yaxis_tickformat=".0%", legend_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 PHQ-9 Equivalent Score Over Time")
    phq9_scores = [map_to_phq9(s)["phq9_score"] for s in scores_f]
    fig2 = go.Figure(go.Bar(
        x=dates_f, y=phq9_scores,
        marker_color=[map_to_phq9(s)["color"] for s in scores_f],
        hovertemplate="Date: %{x}<br>PHQ-9: %{y}/27<extra></extra>"
    ))
    fig2.add_hline(y=10, line_dash="dash", line_color="#FF9800",
                   annotation_text="Moderate threshold (10)")
    fig2.update_layout(
        height=280, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="white", yaxis_range=[0, 28]
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📅 Session History")
    if sessions:
        import pandas as pd
        df = pd.DataFrame(sessions)[
            ["date", "time", "risk_score", "risk_level",
             "phq9_score", "severity", "modalities", "source"]
        ]
        df["risk_score"] = df["risk_score"].apply(lambda x: f"{x:.1%}")
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📄 Export Clinical Report for Doctor")

    col_pdf1, col_pdf2 = st.columns([1, 1])
    with col_pdf1:
        if st.button("📥 Generate Clinical PDF", use_container_width=True, key="progress_pdf_gen"):
            with st.spinner("Generating clinical report..."):
                try:
                    user_data      = get_user(db, user_id)
                    latest_session = sessions[0] if sessions else {}
                    mods_str       = latest_session.get("modalities", "")

                    # Bug 14 fix: reconstruct weights from session or infer from modalities
                    stored_weights = latest_session.get("fusion_weights", None)
                    if stored_weights and any(w > 0 for w in stored_weights):
                        weights = stored_weights
                    else:
                        has_audio = "Audio"      in mods_str
                        has_text  = "Text"       in mods_str
                        has_beh   = "Behavioral" in mods_str
                        n_active  = sum([has_audio, has_text, has_beh])
                        w = 1.0 / n_active if n_active > 0 else 0.0
                        weights = [
                            w if has_audio else 0.0,
                            w if has_text  else 0.0,
                            w if has_beh   else 0.0,
                        ]

                    latest_fr = {
                        "risk_score" : latest_session.get("risk_score",  0),
                        "risk_level" : latest_session.get("risk_level",  "—"),
                        "confidence" : latest_session.get("confidence",  0),
                        "modalities" : mods_str,
                        "weights"    : weights,
                    }
                    phq9_latest = map_to_phq9(latest_session.get("risk_score", 0))

                    # Bug 4 / Bug 13 fix: infer text_v2_result from session record
                    # Previously failed when text_probability key not present in old records
                    has_text_data = (
                        latest_session.get("text_probability") is not None
                        or "Text" in mods_str
                    )
                    text_v2 = {
                        "raw_prob" : float(latest_session.get("text_probability", 0) or 0),
                        "severity" : latest_session.get("text_severity", "—") or "—",
                        "sev_model": latest_session.get("text_severity", "—") or "—",
                        "sev_conf" : 0.0,
                    } if has_text_data else None

                    audio_conf = (
                        latest_session.get("audio_confidence")
                        if "Audio" in mods_str else None
                    )

                    pdf_bytes = generate_pdf_report(
                        user_name      = user_data.get("name",   "User")  if user_data else "User",
                        user_age       = user_data.get("age",    0)       if user_data else 0,
                        user_gender    = user_data.get("gender", "")      if user_data else "",
                        fusion_result  = latest_fr,
                        phq9_data      = phq9_latest,
                        text_v2_result = text_v2,
                        audio_conf     = audio_conf,
                        beh_answers    = st.session_state.get("last_beh_answers"),
                        sessions       = sessions,
                        trajectory     = trajectory,
                        report_source  = "progress_history",
                    )
                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.success("✅ Clinical report ready! Click Download.")
                except Exception as e:
                    st.error(f"❌ PDF generation failed: {e}")

    if st.session_state.get("pdf_bytes"):
        with col_pdf2:
            st.download_button(
                label               = "⬇️ Download Clinical Report",
                data                = st.session_state["pdf_bytes"],
                file_name           = f"mindcare_clinical_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime                = "application/pdf",
                use_container_width = True,
                key                 = "progress_pdf_dl"
            )

    st.markdown("---")
    st.markdown("### 🆘 Crisis Resources")
    st.error("""
    **If you are in crisis or having thoughts of self-harm, please reach out immediately:**
    - 🇮🇳 **iCall (India):** 9152987821
    - 🌍 **Vandrevala Foundation:** 1860-2662-345
    - 💬 **iCall Chat:** icallhelpline.org
    """)