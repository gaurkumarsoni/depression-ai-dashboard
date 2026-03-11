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
from model_loader   import load_all_models, predict_text_v2
from explainability import display_full_explanation, explain_fusion
from audio_processor import process_audio_for_model, transcribe_audio, extract_raw_features_for_shap
from conversation   import ConversationManager
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
        "models_loaded"     : False,
        "models"            : None,
        "db"                : None,
        "user_id"           : None,
        "user_name"         : None,
        "conversation"      : None,
        "chat_history"      : [],
        "analysis_done"     : False,
        "latest_result"     : None,
        "groq_key"          : "",
        "mode"              : "chat",
        "page"              : "home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ── Load Models (cached) ───────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_all_models()


# ── Helper: Run Fusion ─────────────────────────────────────────────
@torch.no_grad()
def run_fusion(models, audio_emb=None, text_emb=None, beh_emb=None):
    AUDIO_DIM = 128
    TEXT_DIM  = 768
    BEH_DIM   = 32

    audio = torch.tensor(
        audio_emb if audio_emb is not None
        else np.zeros((1, AUDIO_DIM)),
        dtype=torch.float32
    )
    text  = torch.tensor(
        text_emb if text_emb is not None
        else np.zeros((1, TEXT_DIM)),
        dtype=torch.float32
    )
    beh   = torch.tensor(
        beh_emb if beh_emb is not None
        else np.zeros((1, BEH_DIM)),
        dtype=torch.float32
    )
    mask  = torch.tensor([[
        1.0 if audio_emb is not None else 0.0,
        1.0 if text_emb  is not None else 0.0,
        1.0 if beh_emb   is not None else 0.0,
    ]], dtype=torch.float32)

    fusion_model = models["fusion_model"]
    fusion_model.eval()

    logits, weights = fusion_model(audio, text, beh, mask)
    probs           = torch.softmax(logits, dim=1)
    risk_score      = probs[0][1].item()
    weights_np      = weights[0].numpy()

    if risk_score >= 0.7:
        risk_level = "High Risk"
        color      = "#F44336"
    elif risk_score >= 0.4:
        risk_level = "Moderate Risk"
        color      = "#FF9800"
    else:
        risk_level = "Low Risk"
        color      = "#4CAF50"

    modalities = []
    if audio_emb is not None: modalities.append("Audio")
    if text_emb  is not None: modalities.append("Text")
    if beh_emb   is not None: modalities.append("Behavioral")

    return {
        "risk_score"  : risk_score,
        "risk_level"  : risk_level,
        "color"       : color,
        "confidence"  : max(probs[0]).item(),
        "weights"     : weights_np,
        "modalities"  : ", ".join(modalities) if modalities else "None"
    }


# ── Helper: Get Text Embedding ─────────────────────────────────────
@torch.no_grad()
def get_text_embedding(models, text: str):
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
    return emb.numpy()


# ── Helper: Get Behavioral Embedding ──────────────────────────────
@torch.no_grad()
def get_beh_embedding(models, features: np.ndarray):
    beh_model = models["beh_model"]
    sd_scaler  = models["sd_scaler"]
    scaled     = sd_scaler.transform(features.reshape(1, -1))
    tensor     = torch.tensor(scaled, dtype=torch.float32)
    emb        = beh_model.get_embeddings(tensor)
    return emb.numpy()


# ── Display Results ────────────────────────────────────────────────
def display_results(result, phq9_data=None, show_weights=True):
    has_behavioral = "Behavioral" in result.get("modalities", "")

    col1, col2 = st.columns(2) if not has_behavioral else st.columns(3)

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

    if has_behavioral and phq9_data:
        cols = st.columns(3)
        with cols[2]:
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
            x     = ["Audio", "Text", "Behavioral"],
            y     = weights,
            marker_color = ["#2196F3", "#4CAF50", "#FF9800"],
            text  = [f"{w:.1%}" for w in weights],
            textposition = "outside"
        ))
        fig.update_layout(
            title      = "Modality Attention Weights",
            yaxis_range= [0, 1.1],
            height     = 250,
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font_color    = "white"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"💡 **Clinical Advice:** {phq9_data['advice']}")


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 MindCare AI")
    st.markdown("---")

    # API Keys & Firebase
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
                    st.session_state.models      = get_models()
                    st.session_state.models_loaded = True
                    st.session_state.groq_key    = groq_key
                    if firebase_ok:
                        st.session_state.db = init_firebase()
                    st.success("✅ Models loaded!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    # Navigation
    st.markdown("### 📍 Navigation")
    pages = {
        "🏠 Home"         : "home",
        "💬 Chat Screen"  : "chat",
        "📊 Manual Analysis": "manual",
        "📈 My Progress"  : "progress",
    }
    for label, page_id in pages.items():
        if st.button(label, use_container_width=True):
            st.session_state.page = page_id

    st.markdown("---")

    # User Profile
    if st.session_state.db:
        with st.expander("👤 User Profile"):
            if not st.session_state.user_id:
                name   = st.text_input("Your Name")
                age    = st.number_input("Age", 10, 100, 25)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                user_id = st.text_input("User ID (email or username)")
                if st.button("Save Profile"):
                    if name and user_id:
                        create_user(
                            st.session_state.db,
                            user_id, name, age, gender
                        )
                        st.session_state.user_id   = user_id
                        st.session_state.user_name = name
                        st.success(f"✅ Welcome, {name}!")
            else:
                st.success(f"👤 {st.session_state.user_name}")
                if st.button("Switch User"):
                    st.session_state.user_id   = None
                    st.session_state.user_name = None


# ══════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 MindCare AI</h1>
        <p style="font-size:1.2rem">
            Explainable Multimodal Depression Screening
        </p>
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

    # Init conversation
    if st.session_state.conversation is None:
        st.session_state.conversation = ConversationManager(
            st.session_state.groq_key
        )
        # Start conversation
        opening = st.session_state.conversation.chat(
            "Hello, I'm here to chat."
        )
        st.session_state.chat_history.append({
            "role": "assistant", "content": opening
        })

    # Voice input option
    st.markdown("#### 🎤 Voice or Text Input")
    input_method = st.radio(
        "Input method:",
        ["⌨️ Type", "🎤 Upload Voice"],
        horizontal=True
    )

    user_input = None

    if input_method == "⌨️ Type":
        col1, col2 = st.columns([4, 1])
        with col1:
            typed = st.text_input(
                "Your message:",
                placeholder="How are you feeling today?",
                key="chat_input"
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
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            with st.spinner("Transcribing..."):
                transcribed = transcribe_audio(tmp_path)

            if transcribed:
                st.info(f"📝 Transcribed: *{transcribed}*")
                user_input = transcribed

                # Also analyze audio for depression
                with st.spinner("Analyzing audio..."):
                    models     = st.session_state.models
                    audio_emb, audio_conf = process_audio_for_model(
                        tmp_path,
                        models["audio_scaler"],
                        models["audio_model"],
                        models["audio_config"]
                    )
                    if audio_emb is not None:
                        st.session_state["last_audio_emb"] = audio_emb
            else:
                st.warning("Could not transcribe audio. Please try typing.")

    # Process user input
    if user_input:
        st.session_state.chat_history.append({
            "role": "user", "content": user_input
        })

        with st.spinner("MindCare AI is thinking..."):
            response = st.session_state.conversation.chat(user_input)

        st.session_state.chat_history.append({
            "role": "assistant", "content": response
        })

        # Check if ready for analysis
        if st.session_state.conversation.is_ready_for_analysis():
            st.session_state["ready_for_analysis"] = True

    # Display chat history
    st.markdown("---")
    st.markdown("#### 💬 Conversation")
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message-user">
                    👤 {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                clean = msg["content"].replace(
                    "[READY_FOR_ANALYSIS]", ""
                ).strip()
                st.markdown(f"""
                <div class="chat-message-ai">
                    🧠 {clean}
                </div>
                """, unsafe_allow_html=True)

    # Analysis button
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        analyze_ready = (
            len(st.session_state.chat_history) >= 6 or
            st.session_state.get("ready_for_analysis", False)
        )
        if st.button(
            "🔍 Analyze Now",
            disabled=not analyze_ready,
            use_container_width=True
        ):
            with st.spinner("Running multimodal analysis..."):
                models = st.session_state.models
                conv   = st.session_state.conversation

                # Extract features from conversation
                features = conv.extract_features()

                # Get text embedding from conversation
                full_text = " ".join([
                    m["content"]
                    for m in st.session_state.chat_history
                    if m["role"] == "user"
                ])
                text_emb  = get_text_embedding(models, full_text)

                # v2 text severity for display
                text_v2 = predict_text_v2(
                    full_text,
                    models["text_model"],
                    models["severity_model"],
                    models["tokenizer"]
                )
                st.session_state["text_v2_result"] = text_v2

                # Get audio embedding if available
                audio_emb = st.session_state.get("last_audio_emb")

                # Run fusion
                result  = run_fusion(models, audio_emb, text_emb)
                phq9    = map_to_phq9(result["risk_score"])

                # Get AI explanation
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

    with col2:
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.conversation  = None
            st.session_state.chat_history  = []
            st.session_state.analysis_done = False
            st.session_state.latest_result = None
            st.session_state.pop("ready_for_analysis", None)
            st.session_state.pop("last_audio_emb", None)
            st.rerun()

    # Show results
    if st.session_state.analysis_done and st.session_state.latest_result:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        result = st.session_state.latest_result
        phq9   = st.session_state["last_phq9"]

        # Show v2 text severity if available
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

        # AI Explanation
        st.markdown("### 🤖 AI Explanation")
        st.markdown(f"""
        <div class="chat-message-ai">
            🧠 {st.session_state['explanation']}
        </div>
        """, unsafe_allow_html=True)

        # ── Text Explainability for Chat ───────────────────────────
        st.markdown("---")
        with st.expander("🔍 Why did the AI say this?", expanded=False):
            chat_text = " ".join([
                m["content"] for m in st.session_state.chat_history
                if m["role"] == "user"
            ])
            display_full_explanation(
                text       = chat_text,
                text_model = models["text_model"],
                tokenizer  = models["tokenizer"],
                # Fusion weights if available
                fusion_weights  = result.get("weights"),
                modalities_used = result.get("modalities"),
                risk_score      = result.get("risk_score"),
            )

        # Save to Firebase
        if st.session_state.db and st.session_state.user_id:
            if st.button("💾 Save to My Progress"):
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


# ══════════════════════════════════════════════════════════════════
# MANUAL ANALYSIS PAGE
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == "manual":
    st.markdown("## 📊 Manual Multimodal Analysis")

    if not st.session_state.models_loaded:
        st.warning("Please load models from the sidebar first!")
        st.stop()

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
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                with st.spinner("Extracting audio features..."):
                    models    = st.session_state.models
                    emb, conf = process_audio_for_model(
                        tmp_path,
                        models["audio_scaler"],
                        models["audio_model"],
                        models["audio_config"]
                    )

                if emb is not None:
                    audio_emb_manual = emb
                    st.session_state["manual_audio_emb"]  = emb
                    # Store raw aggregated features for SHAP — uses exact same
                    # feature extraction as the model (extract_covarep_like_features)
                    raw_feats = extract_raw_features_for_shap(
                        tmp_path, models["audio_config"]
                    )
                    st.session_state["manual_audio_raw"] = raw_feats
                    st.success(
                        f"✅ Audio analyzed! "
                        f"Depression confidence: {conf:.1%}"
                    )

                    # Transcribe
                    with st.spinner("Transcribing..."):
                        text = transcribe_audio(tmp_path)
                    if text:
                        st.markdown(f"**📝 Transcript:** {text}")
                else:
                    st.error("Could not process audio file.")

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
                models   = st.session_state.models
                text_emb = get_text_embedding(models, text_input)
                st.session_state["manual_text_emb"]  = text_emb
                st.session_state["manual_last_text"] = text_input

                # v2 prediction with severity
                result_v2 = predict_text_v2(
                    text_input,
                    models["text_model"],
                    models["severity_model"],
                    models["tokenizer"]
                )

            st.success("✅ Text analyzed!")

            # ── Show v2 results ────────────────────────────────────
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

            # Clinical advice
            advice_map = {
                "Not Depressed": "✅ No significant depression indicators detected. Keep maintaining healthy habits!",
                "Low"          : "🟡 Mild indicators present. Consider mindfulness, exercise, and talking to someone you trust.",
                "Moderate"     : "🟠 Moderate depression indicators. Consider speaking with a mental health professional.",
                "Severe"       : "🔴 Severe depression indicators detected. Please reach out to a mental health professional immediately.",
            }
            st.info(f"💡 **Clinical Guidance:** {advice_map[result_v2['severity']]}")

            # ── Text Explainability ────────────────────────────────
            st.markdown("---")
            with st.expander("🔍 Explain this prediction", expanded=True):
                display_full_explanation(
                    text        = text_input,
                    text_model  = models["text_model"],
                    tokenizer   = models["tokenizer"],
                )

    # ── Behavioral Tab ────────────────────────────────────────────
    with tab3:
        st.markdown("### 📋 Behavioral Questionnaire")
        st.info("Answer these questions about your lifestyle and behavioral patterns.")

        col1, col2 = st.columns(2)
        with col1:
            gender   = st.selectbox("Gender", ["Male", "Female"])
            age      = st.slider("Age", 15, 60, 25)
            acad_p   = st.slider("Academic Pressure (1-5)", 1, 5, 3)
            work_p   = st.slider("Work Pressure (1-5)", 1, 5, 2)
            cgpa     = st.slider("CGPA / GPA", 0.0, 10.0, 7.0)
            study_s  = st.slider("Study Satisfaction (1-5)", 1, 5, 3)

        with col2:
            job_s    = st.slider("Job Satisfaction (1-5)", 1, 5, 3)
            sleep    = st.selectbox(
                "Sleep Duration",
                ["Less than 5 hours", "5-6 hours",
                 "7-8 hours", "More than 8 hours"]
            )
            diet     = st.selectbox(
                "Dietary Habits",
                ["Unhealthy", "Moderate", "Healthy"]
            )
            suicidal = st.selectbox(
                "Suicidal Thoughts?",
                ["No", "Yes"]
            )
            work_hrs = st.slider("Work/Study Hours/day", 0, 16, 8)
            fin_str  = st.slider("Financial Stress (1-5)", 1, 5, 3)
            fam_hist = st.selectbox(
                "Family History of Mental Illness?",
                ["No", "Yes"]
            )

        if st.button("Analyze Behavioral Data 🔍"):
            sleep_map = {
                "Less than 5 hours": 4,
                "5-6 hours"        : 5.5,
                "7-8 hours"        : 7.5,
                "More than 8 hours": 9
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
                models  = st.session_state.models
                beh_emb = get_beh_embedding(models, features)
                beh_emb_manual = beh_emb
                st.session_state["manual_beh_emb"]  = beh_emb
                st.session_state["manual_beh_raw"]  = features  # raw for SHAP
                st.success("✅ Behavioral data analyzed!")

    # ── Run Fusion ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔀 Multimodal Fusion")

    available = []
    if st.session_state.get("manual_audio_emb") is not None:
        available.append("🎤 Audio")
    if st.session_state.get("manual_text_emb")  is not None:
        available.append("📝 Text")
    if st.session_state.get("manual_beh_emb")   is not None:
        available.append("📋 Behavioral")

    if available:
        st.success(f"Ready to fuse: {', '.join(available)}")
    else:
        st.info("Analyze at least one modality above to enable fusion.")

    if available and st.button("🚀 Run Fusion Analysis", use_container_width=True):
        with st.spinner("Running fusion..."):
            models  = st.session_state.models
            result  = run_fusion(
                models,
                st.session_state.get("manual_audio_emb"),
                st.session_state.get("manual_text_emb"),
                st.session_state.get("manual_beh_emb")
            )
            phq9    = map_to_phq9(result["risk_score"])
            st.session_state.latest_result = result
            st.session_state["last_phq9"]  = phq9

        st.markdown("## 📊 Results")
        display_results(result, phq9)

        # ── Full Explainability Report ─────────────────────────────
        st.markdown("---")
        with st.expander("🔍 Explainability Report", expanded=True):
            display_full_explanation(
                # Text
                text        = st.session_state.get("manual_last_text"),
                text_model  = models["text_model"],
                tokenizer   = models["tokenizer"],
                # Audio
                audio_features = st.session_state.get("manual_audio_raw"),
                audio_model    = models["audio_model"],
                audio_scaler   = models["audio_scaler"],
                audio_config   = models["audio_config"],
                # Behavioral
                beh_features = st.session_state.get("manual_beh_raw"),
                beh_model    = models["beh_model"],
                sd_scaler    = models["sd_scaler"],
                # Fusion
                fusion_weights  = result["weights"],
                modalities_used = result["modalities"],
                risk_score      = result["risk_score"],
            )

        # Save option
        if st.session_state.db and st.session_state.user_id:
            if st.button("💾 Save to My Progress"):
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
                        "source"     : "manual"
                    }
                )
                st.success("✅ Saved!")

        # Clear manual embeddings
        for key in ["manual_audio_emb", "manual_text_emb", "manual_beh_emb",
                     "manual_audio_raw", "manual_beh_raw", "manual_last_text"]:
            st.session_state.pop(key, None)


# ══════════════════════════════════════════════════════════════════
# PROGRESS PAGE
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == "progress":
    st.markdown("## 📈 Mental Health Progress Tracker")

    if not st.session_state.db:
        st.warning("Firebase not configured. Please add firebase-key.json to enable progress tracking.")
        st.stop()

    if not st.session_state.user_id:
        st.warning("Please set up your user profile in the sidebar first!")
        st.stop()

    db      = st.session_state.db
    user_id = st.session_state.user_id

    # Get data
    dates, scores, risk_labels = get_trend_data(db, user_id)
    sessions                   = get_sessions(db, user_id)
    weekly                     = get_weekly_summary(db, user_id)

    if not scores:
        st.info("No screening sessions yet. Complete a screening to see your progress!")
        st.stop()

    trajectory   = compute_trajectory(scores)
    anomaly      = detect_anomaly(scores)

    # ── Summary Cards ──────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions",  len(sessions))
    with col2:
        st.metric("Latest Risk",     f"{scores[-1]:.0%}" if scores else "N/A")
    with col3:
        traj_emoji = {"improving": "📈", "worsening": "📉", "stable": "➡️"}
        st.metric("Trajectory",
                  f"{traj_emoji.get(trajectory, '?')} {trajectory.title()}")
    with col4:
        st.metric("Weekly Sessions", weekly.get("sessions", 0))

    # Anomaly alert
    if anomaly:
        st.warning(
            "⚠️ **Anomaly Detected!** Your latest score is significantly "
            "different from your baseline. Consider speaking with a professional."
        )

    # ── Time Range ─────────────────────────────────────────────────
    st.markdown("---")
    view = st.radio(
        "View:",
        ["All Time", "Last 30 Days", "Last 7 Days"],
        horizontal=True
    )

    if view == "Last 7 Days":
        dates_f, scores_f = dates[-7:],  scores[-7:]
    elif view == "Last 30 Days":
        dates_f, scores_f = dates[-30:], scores[-30:]
    else:
        dates_f, scores_f = dates, scores

    # ── Trend Chart ────────────────────────────────────────────────
    st.markdown("### 📊 Risk Score Over Time")

    colors_map = [
        "#F44336" if s >= 0.7 else
        "#FF9800" if s >= 0.4 else
        "#4CAF50"
        for s in scores_f
    ]

    fig = go.Figure()

    # Risk zones
    fig.add_hrect(y0=0.7, y1=1.0,
                  fillcolor="#F44336", opacity=0.08,
                  annotation_text="High Risk", annotation_position="right")
    fig.add_hrect(y0=0.4, y1=0.7,
                  fillcolor="#FF9800", opacity=0.08,
                  annotation_text="Moderate", annotation_position="right")
    fig.add_hrect(y0=0.0, y1=0.4,
                  fillcolor="#4CAF50", opacity=0.08,
                  annotation_text="Low Risk", annotation_position="right")

    # Score line
    fig.add_trace(go.Scatter(
        x    = dates_f,
        y    = scores_f,
        mode = "lines+markers",
        name = "Risk Score",
        line = dict(color="#4CAF50", width=2),
        marker=dict(
            size  = 10,
            color = colors_map,
            line  = dict(width=2, color="white")
        ),
        hovertemplate="Date: %{x}<br>Risk: %{y:.1%}<extra></extra>"
    ))

    # Moving average
    if len(scores_f) >= 3:
        ma = np.convolve(scores_f, np.ones(3)/3, mode="valid")
        fig.add_trace(go.Scatter(
            x    = dates_f[2:],
            y    = ma,
            mode = "lines",
            name = "3-session avg",
            line = dict(color="#2196F3", width=2, dash="dash"),
        ))

    fig.update_layout(
        height        = 350,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font_color    = "white",
        yaxis_range   = [0, 1.05],
        yaxis_tickformat = ".0%",
        legend_bgcolor   = "rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── PHQ-9 Chart ────────────────────────────────────────────────
    st.markdown("### 📋 PHQ-9 Equivalent Score Over Time")
    phq9_scores = [map_to_phq9(s)["phq9_score"] for s in scores_f]

    fig2 = go.Figure(go.Bar(
        x              = dates_f,
        y              = phq9_scores,
        marker_color   = [map_to_phq9(s)["color"] for s in scores_f],
        hovertemplate  = "Date: %{x}<br>PHQ-9: %{y}/27<extra></extra>"
    ))
    fig2.add_hline(y=10, line_dash="dash", line_color="#FF9800",
                   annotation_text="Moderate threshold (10)")
    fig2.update_layout(
        height        = 280,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font_color    = "white",
        yaxis_range   = [0, 28]
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Session Table ──────────────────────────────────────────────
    st.markdown("### 📅 Session History")
    if sessions:
        import pandas as pd
        df = pd.DataFrame(sessions)[
            ["date", "time", "risk_score", "risk_level",
             "phq9_score", "severity", "modalities", "source"]
        ]
        df["risk_score"] = df["risk_score"].apply(lambda x: f"{x:.1%}")
        st.dataframe(df, use_container_width=True)

    # ── PDF Export ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Export Report for Doctor")

    if st.button("📥 Generate PDF Report", use_container_width=True):
        with st.spinner("Generating PDF..."):
            user_data    = get_user(db, user_id)
            latest_result = sessions[0] if sessions else {}
            phq9_latest   = map_to_phq9(
                latest_result.get("risk_score", 0)
            )

            pdf_bytes = generate_pdf_report(
                user_name     = user_data.get("name", "User"),
                user_age      = user_data.get("age", 0),
                sessions      = sessions,
                latest_result = latest_result,
                phq9_data     = phq9_latest,
                trajectory    = trajectory
            )

        st.download_button(
            label    = "⬇️ Download PDF Report",
            data     = pdf_bytes,
            file_name= f"mindcare_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime     = "application/pdf"
        )
        st.success("✅ Report generated!")

    # Crisis resources
    st.markdown("---")
    st.markdown("### 🆘 Crisis Resources")
    st.error("""
    **If you are in crisis or having thoughts of self-harm, please reach out immediately:**
    - 🇮🇳 **iCall (India):** 9152987821
    - 🌍 **Vandrevala Foundation:** 1860-2662-345
    - 💬 **iCall Chat:** icallhelpline.org
    """)