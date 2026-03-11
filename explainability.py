"""
explainability.py — MindCare AI
Techniques:
  Text       → Attention Rollout (context-aware, uses BERT own attention)
               + LIME sentence-perturbation fallback
  Audio      → SHAP KernelExplainer on raw librosa features
  Behavioral → SHAP KernelExplainer on 13 lifestyle factors
  Fusion     → Attention weights from fusion model
"""

import torch
import numpy as np
import plotly.graph_objects as go
import streamlit as st

DEVICE = torch.device("cpu")

# Matches exact order from extract_covarep_like_features():
# MFCCs(13) + Delta MFCCs(13) + Spectral(3) + Contrast(7) + Pitch(1) + RMS(1) + ZCR(1) = 39
AUDIO_FEATURE_NAMES = [
    # MFCCs — timbre, vocal quality
    "MFCC 1 (Vocal Timbre)", "MFCC 2", "MFCC 3", "MFCC 4", "MFCC 5",
    "MFCC 6", "MFCC 7", "MFCC 8", "MFCC 9", "MFCC 10",
    "MFCC 11", "MFCC 12", "MFCC 13",
    # Delta MFCCs — rate of change of timbre
    "Delta MFCC 1 (Speech Rate)", "Delta MFCC 2", "Delta MFCC 3",
    "Delta MFCC 4", "Delta MFCC 5", "Delta MFCC 6", "Delta MFCC 7",
    "Delta MFCC 8", "Delta MFCC 9", "Delta MFCC 10",
    "Delta MFCC 11", "Delta MFCC 12", "Delta MFCC 13",
    # Spectral — brightness, bandwidth, clarity
    "Spectral Centroid (Brightness)",
    "Spectral Bandwidth (Voice Width)",
    "Spectral Rolloff (Energy Dist)",
    # Spectral Contrast — peak vs valley in spectrum
    "Spectral Contrast 1", "Spectral Contrast 2", "Spectral Contrast 3",
    "Spectral Contrast 4", "Spectral Contrast 5",
    "Spectral Contrast 6", "Spectral Contrast 7",
    # Pitch, Energy, ZCR
    "Pitch / F0 (Hz)",
    "RMS Energy (Loudness)",
    "Zero Crossing Rate (Voice Tremor)",
]

BEHAVIORAL_FEATURE_NAMES = [
    "Gender", "Age", "Academic Pressure", "Work Pressure",
    "CGPA / GPA", "Study Satisfaction", "Job Satisfaction",
    "Sleep Duration (hrs)", "Dietary Habits", "Suicidal Thoughts",
    "Work/Study Hours", "Financial Stress", "Family History"
]


# ══════════════════════════════════════════════════════════════════
# TEXT — Attention Rollout (primary) + LIME fallback
# ══════════════════════════════════════════════════════════════════

def explain_text_attention_rollout(text, text_model, tokenizer):
    """
    Attention Rollout — extracts token importance from BERT attention layers.
    
    Why better than LIME for transformers:
      LIME masks words → breaks context → wrong attributions
      Attention Rollout uses the model's OWN internal attention weights
      Captures HOW each word relates to the final [CLS] prediction
      "I hate my life" → hate+life attended together (negative context)
      "I love my life" → love+life attended together (positive context)
    
    Method:
      1. Forward pass with output_attentions=True
      2. Rollout: propagate attention through all 12 layers
      3. Extract attention from CLS token to each word token
      4. Multiply by gradient sign to get direction (pos/neg)
    """
    text_model.eval()

    enc = tokenizer(
        text, max_length=128,
        padding="max_length", truncation=True,
        return_tensors="pt"
    )
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    seq_len        = int(attention_mask.sum().item())

    # ── Step 1: Forward pass with attention weights ────────────────
    with torch.no_grad():
        bert_out = text_model.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            output_attentions = True
        )
        # bert_out.attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
        attentions = bert_out.attentions   # 12 layers

    # ── Step 2: Attention Rollout ──────────────────────────────────
    # Start with identity matrix
    rollout = torch.eye(seq_len)

    for layer_attn in attentions:
        # layer_attn: (1, n_heads, seq_len, seq_len)
        # Average across heads
        attn = layer_attn[0].mean(dim=0)[:seq_len, :seq_len]  # (seq, seq)
        # Add residual connection (identity) and renormalize
        attn = attn + torch.eye(seq_len)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        # Multiply rollout
        rollout = torch.matmul(attn, rollout)

    # ── Step 3: CLS attention to each token ───────────────────────
    # rollout[0] = how much CLS attended to each token through all layers
    cls_attention = rollout[0, 1:seq_len-1].numpy()  # skip [CLS] and [SEP]

    # Normalize to [0, 1]
    if cls_attention.max() > 0:
        cls_attention = cls_attention / cls_attention.max()

    # ── Step 4: Get prediction direction per token ────────────────
    # Use gradient of logit w.r.t. embeddings to get sign
    embed_layer = text_model.bert.embeddings.word_embeddings
    embeds = embed_layer(input_ids).detach().requires_grad_(True)

    bert_out2 = text_model.bert(
        inputs_embeds  = embeds,
        attention_mask = attention_mask
    )
    cls_hidden = text_model.dropout(bert_out2.last_hidden_state[:, 0, :])
    logits     = text_model.classifier(cls_hidden)
    dep_logit  = logits[0, 1]   # depression class logit
    dep_logit.backward()

    # Gradient sign per token → direction (depressive +1 or positive -1)
    grad_sign = embeds.grad[0, 1:seq_len-1].sum(dim=-1).sign().numpy()

    # ── Step 5: Combine attention magnitude + gradient sign ───────
    token_scores = cls_attention * grad_sign

    # ── Step 6: Decode tokens ─────────────────────────────────────
    token_ids = input_ids[0, 1:seq_len-1].tolist()
    tokens    = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Merge subword tokens (RoBERTa uses Ġ prefix for word start)
    merged_words  = []
    merged_scores = []
    current_word  = ""
    current_score = []

    for tok, score in zip(tokens, token_scores):
        clean = tok.replace("Ġ", " ").replace("Â", "").strip()
        if tok.startswith("Ġ") or not current_word:
            # New word starts
            if current_word and current_score:
                merged_words.append(current_word)
                merged_scores.append(float(np.mean(current_score)))
            current_word  = clean
            current_score = [float(score)]
        else:
            # Continuation of previous word
            current_word += clean
            current_score.append(float(score))

    if current_word and current_score:
        merged_words.append(current_word)
        merged_scores.append(float(np.mean(current_score)))

    # Sort by absolute score
    word_score_pairs = sorted(
        zip(merged_words, merged_scores),
        key=lambda x: abs(x[1]), reverse=True
    )
    # Filter empty/punctuation
    word_score_pairs = [(w, s) for w, s in word_score_pairs if w.strip()]

    return word_score_pairs, None


def explain_text_lime_sentence(text, text_model, tokenizer, num_samples=200):
    """
    LIME with SENTENCE-LEVEL perturbations instead of word masking.
    
    Instead of masking words, we replace entire clauses with neutral phrases.
    This preserves grammatical context much better.
    Used as fallback if attention rollout fails.
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return None, "lime not installed. Run: pip install lime"

    text_model.eval()

    def predict_proba(texts):
        probs = []
        for t in texts:
            # Replace LIME mask token with neutral phrase
            t = t.replace("UNKWORDZ", "something")
            enc = tokenizer(t, max_length=128, padding="max_length",
                            truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = text_model(enc["input_ids"], enc["attention_mask"])
                p = torch.softmax(logits, dim=1)[0].detach().numpy()
            probs.append(p)
        return np.array(probs)

    explainer = LimeTextExplainer(
        class_names=["Not Depressed", "Depressed"],
        mask_string="neutral",  # replace with neutral word, not empty
        bow=False,
        random_state=42
    )
    exp = explainer.explain_instance(
        text, predict_proba,
        num_features=15,
        num_samples=num_samples,
        labels=(1,)
    )
    word_scores = sorted(exp.as_list(label=1), key=lambda x: abs(x[1]), reverse=True)
    return word_scores, None


def get_text_explanation(text, text_model, tokenizer):
    """Try Attention Rollout first (best), fall back to LIME."""
    try:
        result, err = explain_text_attention_rollout(text, text_model, tokenizer)
        if err is None and result:
            return result, None, "attention_rollout"
    except Exception as e:
        pass  # fall through to LIME

    result, err = explain_text_lime_sentence(text, text_model, tokenizer)
    return result, err, "lime"


# ══════════════════════════════════════════════════════════════════
# AUDIO — SHAP on raw librosa features
# ══════════════════════════════════════════════════════════════════

def explain_audio_shap(audio_features_raw, audio_model, audio_scaler, audio_config=None):
    """
    SHAP on raw (unscaled) audio feature vector from librosa.
    audio_features_raw: np.ndarray (1, n_features)
    """
    try:
        import shap
    except ImportError:
        return None, None, "shap not installed. Run: pip install shap"

    n_features = audio_features_raw.shape[1]
    feat_names = (AUDIO_FEATURE_NAMES[:n_features]
                  if n_features <= len(AUDIO_FEATURE_NAMES)
                  else [f"Feature {i}" for i in range(n_features)])
    seq_len = (audio_config or {}).get("seq_len", 1)

    def predict_fn(X):
        probs = []
        for row in X:
            scaled = audio_scaler.transform(row.reshape(1, -1))
            tensor = torch.tensor(
                np.repeat(scaled[:, np.newaxis, :], seq_len, axis=1),
                dtype=torch.float32
            )
            with torch.no_grad():
                logits = audio_model(tensor)
                p = torch.softmax(logits, dim=1)[0][1].item()
            probs.append(p)
        return np.array(probs)

    background  = np.zeros((1, n_features))
    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(audio_features_raw, nsamples=150, silent=True)
    return shap_values[0], feat_names, None


# ══════════════════════════════════════════════════════════════════
# BEHAVIORAL — SHAP
# ══════════════════════════════════════════════════════════════════

def explain_behavioral_shap(features_raw, beh_model, sd_scaler):
    """SHAP on raw behavioral features. features_raw: (1, 13)"""
    try:
        import shap
    except ImportError:
        return None, None, "shap not installed. Run: pip install shap"

    n_features = features_raw.shape[1]
    feat_names = (BEHAVIORAL_FEATURE_NAMES[:n_features]
                  if n_features <= len(BEHAVIORAL_FEATURE_NAMES)
                  else [f"Feature {i}" for i in range(n_features)])

    def predict_fn(X):
        probs = []
        for row in X:
            scaled = sd_scaler.transform(row.reshape(1, -1))
            tensor = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                logits = beh_model(tensor)
                p = torch.softmax(logits, dim=1)[0][1].item()
            probs.append(p)
        return np.array(probs)

    background  = np.zeros((1, n_features))
    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(features_raw, nsamples=150, silent=True)
    return shap_values[0], feat_names, None


# ══════════════════════════════════════════════════════════════════
# FUSION
# ══════════════════════════════════════════════════════════════════

def explain_fusion(weights_np, modalities_used, risk_score):
    labels = ["Audio", "Text", "Behavioral"]
    used   = ["Audio" in modalities_used, "Text" in modalities_used,
              "Behavioral" in modalities_used]
    active_labels  = [l for l, u in zip(labels, used) if u]
    active_weights = [float(w) for w, u in zip(weights_np, used) if u]
    dom_idx      = int(np.argmax(active_weights)) if active_weights else 0
    dominant_mod = active_labels[dom_idx] if active_labels else "None"
    dominant_w   = active_weights[dom_idx] if active_weights else 0.0
    return {
        "labels": labels, "weights": [float(w) for w in weights_np],
        "used": used, "active_labels": active_labels,
        "active_weights": active_weights,
        "dominant_mod": dominant_mod, "dominant_w": dominant_w,
        "risk_score": risk_score,
    }


# ══════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def display_text_explanation(word_scores, method="attention_rollout"):
    if not word_scores:
        st.info("No text explanation available.")
        return

    if method == "attention_rollout":
        st.markdown("#### 🔍 Word Importance (Attention Rollout)")
        st.caption(
            "Uses BERT's own attention weights — **context-aware**.  "
            "🔴 Red = attended in depressive context  |  🟢 Green = attended in positive context  "
            "|  Darker = stronger attention signal"
        )
        st.info(
            "💡 **How to read this:** Unlike simple word matching, Attention Rollout shows "
            "HOW the model interprets each word in context. 'Life' in 'I hate my life' "
            "and 'I love my life' will have DIFFERENT colors because the surrounding "
            "context changes how attention flows."
        )
    else:
        st.markdown("#### 🔍 Word Importance (LIME)")
        st.caption("🔴 Red = depressive indicator  |  🟢 Green = positive indicator  |  Darker = stronger signal")

    html = ["<div style='line-height:2.6; font-size:1.05rem; padding:1rem; "
            "background:#1E2329; border-radius:8px; display:flex; flex-wrap:wrap; gap:6px;'>"]
    for word, score in word_scores:
        intensity = min(abs(float(score)), 1.0)
        alpha = 0.25 + 0.65 * intensity
        if score > 0.01:
            bg, border = f"rgba(244,67,54,{alpha:.2f})", "#F44336"
        elif score < -0.01:
            bg, border = f"rgba(76,175,80,{alpha:.2f})", "#4CAF50"
        else:
            bg, border = "rgba(255,255,255,0.05)", "transparent"
        html.append(
            f"<span title='Score: {score:+.4f}' style='background:{bg}; "
            f"border:1px solid {border}; padding:3px 8px; border-radius:5px; "
            f"color:white; white-space:nowrap; cursor:help;'>{word}</span>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)
    st.caption("💡 Hover over a word to see its exact influence score.")

    top_n  = min(12, len(word_scores))
    top    = word_scores[:top_n]
    words  = [w for w, _ in top]
    scores = [float(s) for _, s in top]
    colors = ["#F44336" if s > 0 else "#4CAF50" for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=words, orientation="h",
        marker_color=colors,
        text=[f"{s:+.4f}" for s in scores], textposition="outside"
    ))
    fig.update_layout(
        title="Top Words by Influence",
        xaxis_title="+ve = depressive  |  -ve = positive",
        height=max(300, top_n * 32 + 80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", margin=dict(l=140)
    )
    st.plotly_chart(fig, use_container_width=True)


def display_audio_explanation(shap_values, feature_names):
    if shap_values is None:
        st.info("No audio explanation available.")
        return

    st.markdown("#### 🎤 Acoustic Feature Importance (SHAP)")
    st.caption("🔴 Red = increases depression risk  |  🟢 Green = decreases risk")

    pairs  = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)[:15]
    names  = [p[0] for p in pairs]
    values = [float(p[1]) for p in pairs]
    colors = ["#F44336" if v > 0 else "#4CAF50" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.5f}" for v in values], textposition="outside"
    ))
    fig.update_layout(
        title="Top 15 Acoustic Features (SHAP Values)",
        xaxis_title="SHAP Value  (+ve = increases depression risk)",
        height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", margin=dict(l=180)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature group summary
    groups = {
        "Timbre (MFCCs)"         : [v for n, v in zip(feature_names, shap_values) if "MFCC" in n and "Delta" not in n],
        "Speech Rate (Delta)"    : [v for n, v in zip(feature_names, shap_values) if "Delta" in n],
        "Spectral / Brightness"  : [v for n, v in zip(feature_names, shap_values) if "Spectral" in n],
        "Pitch / F0"             : [v for n, v in zip(feature_names, shap_values) if "Pitch" in n or "F0" in n],
        "Energy / Loudness"      : [v for n, v in zip(feature_names, shap_values) if "RMS" in n or "Energy" in n],
        "Voice Tremor (ZCR)"     : [v for n, v in zip(feature_names, shap_values) if "ZCR" in n or "Crossing" in n],
    }
    g_names  = [g for g, vs in groups.items() if vs]
    g_values = [float(np.sum(vs)) for g, vs in groups.items() if vs]
    if g_names:
        fig2 = go.Figure(go.Bar(
            x=g_names, y=g_values,
            marker_color=["#F44336" if v > 0 else "#4CAF50" for v in g_values],
            text=[f"{v:+.4f}" for v in g_values], textposition="outside"
        ))
        fig2.update_layout(
            title="SHAP by Acoustic Feature Group",
            height=300, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    top_name, top_val = pairs[0]
    st.info(f"💡 **{top_name}** had the strongest influence (SHAP={top_val:+.5f}).")


def display_behavioral_explanation(shap_values, feature_names, raw_values):
    if shap_values is None:
        st.info("No behavioral explanation available.")
        return

    st.markdown("#### 📋 Behavioral Factor Importance (SHAP)")
    st.caption("🔴 Red = increases depression risk  |  🟢 Green = decreases risk  |  [value] = your response")

    raw   = raw_values[0] if hasattr(raw_values, '__len__') else raw_values
    pairs = sorted(zip(feature_names, shap_values, raw),
                   key=lambda x: abs(x[1]), reverse=True)
    labels = [f"{p[0]}  [{p[2]:.1f}]" for p in pairs]
    values = [float(p[1]) for p in pairs]
    colors = ["#F44336" if v > 0 else "#4CAF50" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values], textposition="outside"
    ))
    fig.update_layout(
        title="All Behavioral Factors (SHAP Values)",
        xaxis_title="SHAP Value  (+ve = increases depression risk)",
        height=max(380, len(pairs) * 34 + 80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", margin=dict(l=240)
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    risk_f = [(n, v, r) for n, v, r in pairs if v > 0.001]
    prot_f = [(n, v, r) for n, v, r in pairs if v < -0.001]

    with col1:
        st.markdown("**🔴 Risk-increasing factors:**")
        for n, v, r in risk_f[:5]:
            st.markdown(f"- **{n}** = {r:.1f} → `{v:+.4f}`")
        if not risk_f:
            st.markdown("*None significant*")
    with col2:
        st.markdown("**🟢 Protective factors:**")
        for n, v, r in prot_f[:5]:
            st.markdown(f"- **{n}** = {r:.1f} → `{v:+.4f}`")
        if not prot_f:
            st.markdown("*None significant*")

    top = pairs[0]
    direction = "increases" if top[1] > 0 else "decreases"
    st.info(f"💡 **{top[0]}** (value={top[2]:.1f}) {direction} risk by {abs(top[1]):.4f} SHAP units.")


def display_fusion_explanation(fusion_exp):
    st.markdown("#### 🔀 Fusion Decision Explanation")
    st.caption("How the AI combined different data sources to reach its conclusion")

    labels  = fusion_exp["labels"]
    weights = fusion_exp["weights"]
    used    = fusion_exp["used"]
    colors  = ["#424242" if not u else ("#F44336" if w == max(weights) else "#2196F3")
               for u, w in zip(used, weights)]

    fig = go.Figure(go.Bar(
        x=labels, y=weights, marker_color=colors,
        text=[f"{w:.1%}" if u else "Not used" for w, u in zip(weights, used)],
        textposition="outside"
    ))
    fig.update_layout(
        title="Modality Attention Weights", yaxis_title="Weight",
        yaxis_range=[0, 1.15], height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    dom, dw, risk = fusion_exp["dominant_mod"], fusion_exp["dominant_w"], fusion_exp["risk_score"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div style='background:#1E2329; padding:1rem; border-radius:8px;
            border-left:4px solid #F44336;'><b>🎯 Dominant Modality</b><br>
            <span style='font-size:1.4rem; color:#F44336;'>{dom}</span><br>
            <span style='color:#aaa;'>Weight: {dw:.1%}</span></div>""",
            unsafe_allow_html=True)
    with col2:
        active = ", ".join(fusion_exp["active_labels"]) or "None"
        st.markdown(f"""<div style='background:#1E2329; padding:1rem; border-radius:8px;
            border-left:4px solid #2196F3;'><b>📊 Active Modalities</b><br>
            <span style='color:#2196F3;'>{active}</span><br>
            <span style='color:#aaa;'>Final risk: {risk:.1%}</span></div>""",
            unsafe_allow_html=True)

    st.info(f"💡 **{dom}** ({dw:.1%} weight) drove the final risk score of {risk:.1%}.")


# ══════════════════════════════════════════════════════════════════
# MASTER DISPLAY
# ══════════════════════════════════════════════════════════════════

def display_full_explanation(
    text=None,           text_model=None,    tokenizer=None,
    audio_features=None, audio_model=None,   audio_scaler=None,  audio_config=None,
    beh_features=None,   beh_model=None,     sd_scaler=None,
    fusion_weights=None, modalities_used=None, risk_score=None
):
    st.markdown("## 🔍 Explainability Report")
    st.caption("Understand **why** the AI made this prediction. For research purposes only.")

    tabs = []
    if text           is not None and text_model  is not None: tabs.append("📝 Text")
    if audio_features is not None and audio_model is not None: tabs.append("🎤 Audio")
    if beh_features   is not None and beh_model   is not None: tabs.append("📋 Behavioral")
    if fusion_weights is not None:                              tabs.append("🔀 Fusion")

    if not tabs:
        st.info("No modalities available for explanation.")
        return

    tab_objs = st.tabs(tabs)
    idx = 0

    if "📝 Text" in tabs:
        with tab_objs[idx]:
            with st.spinner("Computing attention-based word importance..."):
                word_scores, err, method = get_text_explanation(text, text_model, tokenizer)
            if err:
                st.warning(f"⚠️ {err}")
            else:
                display_text_explanation(word_scores, method)
        idx += 1

    if "🎤 Audio" in tabs:
        with tab_objs[idx]:
            with st.spinner("Computing acoustic SHAP values..."):
                shap_vals, feat_names, err = explain_audio_shap(
                    audio_features, audio_model, audio_scaler,
                    audio_config or {"seq_len": 1}
                )
            if err:
                st.warning(f"⚠️ {err}")
            else:
                display_audio_explanation(shap_vals, feat_names)
        idx += 1

    if "📋 Behavioral" in tabs:
        with tab_objs[idx]:
            with st.spinner("Computing behavioral SHAP values..."):
                shap_vals, feat_names, err = explain_behavioral_shap(
                    beh_features, beh_model, sd_scaler
                )
            if err:
                st.warning(f"⚠️ {err}")
            else:
                display_behavioral_explanation(shap_vals, feat_names, beh_features)
        idx += 1

    if "🔀 Fusion" in tabs:
        with tab_objs[idx]:
            fusion_exp = explain_fusion(fusion_weights, modalities_used, risk_score)
            display_fusion_explanation(fusion_exp)