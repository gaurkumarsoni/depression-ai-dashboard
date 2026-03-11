# debug_test.py
import torch
import pickle
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import numpy as np

HF_REPO = "gaurkumarsoni/depression-ai-models"
DEVICE  = torch.device("cpu")

class TextClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert       = AutoModel.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(cls)

    def get_embeddings(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

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

# ── Load models ────────────────────────────────────────────────────
print("Loading models...")
tokenizer  = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
text_model = TextClassifier()
text_weights = hf_hub_download(HF_REPO, "text_roberta_reddit.pt")
text_model.load_state_dict(torch.load(text_weights, map_location=DEVICE))
text_model.eval()

fusion_model   = MissingModalityFusion()
fusion_weights = hf_hub_download(HF_REPO, "mm_fusion_best.pt")
fusion_model.load_state_dict(torch.load(fusion_weights, map_location=DEVICE))
fusion_model.eval()
print("Models loaded!\n")

# ── Test ───────────────────────────────────────────────────────────
tests = [
    "I cleared my UGC NET exam, I am so happy!",
    "I am very happy today, life is great!",
    "I feel hopeless, can't get out of bed",
    "I hate myself, everything is pointless",
    "I am blessed and thankful to God for everything",
]

print("=" * 60)
print(f"{'Text':<45} {'TextModel':>10} {'Fusion':>10}")
print("=" * 60)

for text in tests:
    enc = tokenizer(
        text, max_length=256,
        padding="max_length", truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        # Direct text model
        logits = text_model(enc["input_ids"], enc["attention_mask"])
        probs  = torch.softmax(logits, dim=1)
        text_pred  = torch.argmax(logits, dim=1).item()
        text_conf  = probs[0][1].item()  # depression probability

        # Text embedding
        text_emb = text_model.get_embeddings(
            enc["input_ids"], enc["attention_mask"]
        )

        # Fusion with text only
        audio = torch.zeros(1, 128)
        beh   = torch.zeros(1, 32)
        mask  = torch.tensor([[0.0, 1.0, 0.0]])

        fusion_logits, weights = fusion_model(audio, text_emb, beh, mask)
        fusion_probs = torch.softmax(fusion_logits, dim=1)
        fusion_risk  = fusion_probs[0][1].item()

    text_label   = "DEP" if text_pred == 1 else "OK"
    fusion_label = "DEP" if fusion_risk > 0.5 else "OK"

    print(f"{text[:45]:<45} "
          f"{text_label}({text_conf:.0%})"
          f" | Fusion: {fusion_label}({fusion_risk:.0%})")

print("=" * 60)
print("\nKey insight:")
print("If TextModel=OK but Fusion=DEP → fusion model is biased")
print("If TextModel=DEP → text model has issue")