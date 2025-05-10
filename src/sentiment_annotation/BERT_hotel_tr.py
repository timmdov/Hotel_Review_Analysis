"""
BERT (Turkish hotel reviews) sentiment predictor
Model: anilguven/bert_tr_turkish_hotel_reviews
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── TEMP SSL workaround ────────────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")  # allow download
# ────────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "anilguven/bert_tr_turkish_hotel_reviews"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, trust_remote_code=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def predict_sentiment(text) -> str:
    """
    Return 'negative' | 'neutral' | 'positive' for one review string.
    Non-string input → str(), empty → neutral by default.
    """
    if not isinstance(text, str):
        text = str(text or "")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred   = int(logits.argmax(dim=-1).cpu())

    return LABEL_MAP.get(pred, "neutral")
