import os, pickle, torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

HF_REPO            = "gaurkumarsoni/depression-ai-models"
DEVICE             = torch.device("cpu")
BINARY_THRESHOLD   = 0.315  # 31.5% — covers birthday/party edge cases
LOW_THRESHOLD      = 0.70
MODERATE_THRESHOLD = 0.90


# ── Model Definitions ──────────────────────────────────────────────

class AudioTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4,
                 num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        self.pos_enc     = nn.Parameter(torch.zeros(1, 512, d_model))
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool   = nn.Linear(d_model, 1)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x      = self.input_proj(x)
        x      = x + self.pos_enc[:, :x.size(1), :]
        x      = self.transformer(x)
        w      = torch.softmax(self.attn_pool(x), dim=1)
        pooled = (x * w).sum(dim=1)
        return self.classifier(pooled)

    def get_embeddings(self, x):
        x = self.input_proj(x)
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.transformer(x)
        w = torch.softmax(self.attn_pool(x), dim=1)
        return (x * w).sum(dim=1)


class MentalBERTClassifier(nn.Module):
    """Binary depression classifier — MentalBERT v2 (F1=0.9917)"""
    def __init__(self, bert_model, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert       = bert_model          # shared backbone passed in
        hidden_size     = self.bert.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls)

    def get_embeddings(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]


class SeverityClassifier3(nn.Module):
    """3-class severity: Low / Moderate / Severe (F1=0.7268)"""
    def __init__(self, bert_model, num_classes=3, dropout=0.3):
        super().__init__()
        self.bert       = bert_model          # shared backbone passed in
        hidden_size     = self.bert.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls)


class StudentDepMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

    def get_embeddings(self, x):
        for layer in list(self.network.children())[:-1]:
            x = layer(x)
        return x


class MissingModalityFusion(nn.Module):
    def __init__(self, audio_dim=128, text_dim=768,
                 beh_dim=32, hidden=128,
                 num_classes=2, dropout=0.4):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden),
            nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.text_proj  = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.beh_proj   = nn.Sequential(
            nn.Linear(beh_dim, hidden),
            nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.mask_attention = nn.Sequential(
            nn.Linear(hidden * 3 + 3, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio, text, beh, mask):
        a = self.audio_proj(audio)
        t = self.text_proj(text)
        b = self.beh_proj(beh)
        if mask.shape[0] != audio.shape[0]:
            mask = mask.expand(audio.shape[0], -1)
        a = a * mask[:, 0:1]
        t = t * mask[:, 1:2]
        b = b * mask[:, 2:3]
        concat      = torch.cat([a, t, b, mask], dim=1)
        raw_weights = self.mask_attention(concat)
        weights     = raw_weights * mask
        weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights     = weights / weights_sum
        fused = (
            weights[:, 0:1] * a +
            weights[:, 1:2] * t +
            weights[:, 2:3] * b
        )
        return self.classifier(fused), weights


# ── Text Prediction v2 ─────────────────────────────────────────────
@torch.no_grad()
def predict_text_v2(text, text_model, severity_model, tokenizer):
    """
    Combined binary + severity prediction.
    Returns dict with probability, severity label, color, emoji.
    """
    enc  = tokenizer(
        text, max_length=128,
        padding="max_length", truncation=True,
        return_tensors="pt"
    )
    ids  = enc["input_ids"]
    mask = enc["attention_mask"]

    # Binary probability
    logits   = text_model(ids, mask)
    raw_prob = torch.softmax(logits, dim=1)[0][1].item()

    # Severity
    sev_logits = severity_model(ids, mask)
    sev_probs  = torch.softmax(sev_logits, dim=1)[0]
    sev_label  = torch.argmax(sev_logits, dim=1).item()
    sev_names  = {0: "Low", 1: "Moderate", 2: "Severe"}
    sev_name   = sev_names[sev_label]
    sev_conf   = sev_probs[sev_label].item()

    # Decision logic
    if raw_prob < BINARY_THRESHOLD:
        severity = "Not Depressed"
        emoji    = "✅"
        color    = "#4CAF50"
    elif raw_prob < LOW_THRESHOLD:
        severity = "Low"
        emoji    = "🟡"
        color    = "#FFC107"
    elif raw_prob < MODERATE_THRESHOLD:
        severity = "Moderate"
        emoji    = "🟠"
        color    = "#FF9800"
    else:
        if sev_label == 0:
            severity = "Low"
            emoji    = "🟡"
            color    = "#FFC107"
        elif sev_label == 1:
            severity = "Moderate"
            emoji    = "🟠"
            color    = "#FF9800"
        else:
            severity = "Severe"
            emoji    = "🔴"
            color    = "#F44336"

    return {
        "raw_prob"  : raw_prob,
        "severity"  : severity,
        "emoji"     : emoji,
        "color"     : color,
        "sev_model" : sev_name,
        "sev_conf"  : sev_conf,
    }


# ── Load All Models ────────────────────────────────────────────────
@torch.no_grad()
def load_all_models():
    print("Loading models from HuggingFace...")

    # Config files
    audio_config_path = hf_hub_download(HF_REPO, "audio_config.pkl")
    audio_scaler_path = hf_hub_download(HF_REPO, "audio_scaler.pkl")
    sd_scaler_path    = hf_hub_download(HF_REPO, "sd_scaler.pkl")
    sd_features_path  = hf_hub_download(HF_REPO, "sd_features.pkl")

    with open(audio_config_path, "rb") as f: audio_config = pickle.load(f)
    with open(audio_scaler_path, "rb") as f: audio_scaler = pickle.load(f)
    with open(sd_scaler_path,    "rb") as f: sd_scaler    = pickle.load(f)
    with open(sd_features_path,  "rb") as f: sd_features  = pickle.load(f)

    # Audio model
    audio_model   = AudioTransformer(input_dim=audio_config["input_dim"]).to(DEVICE)
    audio_weights = hf_hub_download(HF_REPO, "audio_transformer_best.pt")
    audio_model.load_state_dict(torch.load(audio_weights, map_location=DEVICE))
    audio_model.eval()
    print("✅ Audio model loaded!")

    # Load tokenizer exactly like original working code — direct HF model name
    # SamLowe/roberta-base-go_emotions is roberta-based, identical tokenizer
    tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    print("Tokenizer loaded!")

    # Load BERT backbone ONCE — shared between binary + severity models
    print("Loading shared BERT backbone (roberta-base)...")
    shared_bert = AutoModel.from_pretrained("SamLowe/roberta-base-go_emotions").to(DEVICE)
    print("✅ Shared BERT backbone loaded!")

    # Binary text model v2 — uses shared backbone
    text_model        = MentalBERTClassifier(shared_bert).to(DEVICE)
    text_weights_path = hf_hub_download(HF_REPO, "mentalbert_v2/binary_model_weights.pt")
    text_model.load_state_dict(torch.load(text_weights_path, map_location=DEVICE))
    text_model.eval()
    print("✅ Binary text model v2 loaded! (F1=0.9917)")

    # Severity model v2 — reuses same backbone, no extra download
    severity_model        = SeverityClassifier3(shared_bert).to(DEVICE)
    severity_weights_path = hf_hub_download(HF_REPO, "mentalbert_v2/severity_model_weights.pt")
    severity_model.load_state_dict(torch.load(severity_weights_path, map_location=DEVICE))
    severity_model.eval()
    print("✅ Severity model v2 loaded! (F1=0.7268)")

    # Behavioral model
    beh_model   = StudentDepMLP(input_dim=len(sd_features)).to(DEVICE)
    beh_weights = hf_hub_download(HF_REPO, "student_dep_mlp_best.pt")
    beh_model.load_state_dict(torch.load(beh_weights, map_location=DEVICE))
    beh_model.eval()
    print("✅ Behavioral model loaded!")

    # Fusion model
    fusion_model   = MissingModalityFusion().to(DEVICE)
    fusion_weights = hf_hub_download(HF_REPO, "mm_fusion_best.pt")
    fusion_model.load_state_dict(torch.load(fusion_weights, map_location=DEVICE))
    fusion_model.eval()
    print("✅ Fusion model loaded!")

    print("\n🚀 All models ready!")

    return {
        "audio_model"   : audio_model,
        "text_model"    : text_model,
        "severity_model": severity_model,
        "beh_model"     : beh_model,
        "fusion_model"  : fusion_model,
        "tokenizer"     : tokenizer,
        "audio_scaler"  : audio_scaler,
        "sd_scaler"     : sd_scaler,
        "sd_features"   : sd_features,
        "audio_config"  : audio_config,
    }