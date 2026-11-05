"""Load the fine-tuned model and predict sentiment.   python sentiment.py "Apple beats earnings" """
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


def load_model(path=config.MODEL_DIR):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tok


def predict(model, tok, texts):
    device = next(model.parameters()).device
    enc = tok(texts, padding=True, truncation=True, max_length=config.MAX_LEN, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().tolist()
    results = []
    for p in probs:
        best = max(range(len(p)), key=lambda i: p[i])
        results.append({
            "label": config.LABELS[best],
            "confidence": p[best],
            "probs": {config.LABELS[i]: p[i] for i in range(len(p))},
        })
    return results


if __name__ == "__main__":
    texts = sys.argv[1:] or ["Tesla shares surge after record deliveries",
                             "Fed signals more rate hikes, stocks tumble",
                             "Company to hold annual meeting on Thursday"]
    model, tok = load_model()
    for t, r in zip(texts, predict(model, tok, texts)):
        print(f"{r['label']:8s} {r['confidence']:.0%}  |  {t}")
