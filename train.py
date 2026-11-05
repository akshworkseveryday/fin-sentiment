"""Fine-tune DistilBERT for financial sentiment, compare with baselines, and record results."""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

import config
from baseline import run_baselines, score
from data import load_data


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tok):
        enc = tok(texts, truncation=True, max_length=config.MAX_LEN)
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    d = load_data()

    # ---- Baselines (a few seconds) ----
    baselines = run_baselines(d["train_texts"], d["train_labels"], d["eval_texts"], d["eval_labels"])
    print("Baselines:", json.dumps(baselines, indent=2))

    # ---- Fine-tune DistilBERT ----
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.LABELS),
        id2label=dict(enumerate(config.LABELS)),
        label2id={l: i for i, l in enumerate(config.LABELS)},
    )
    train_ds = SentimentDataset(d["train_texts"], d["train_labels"], tok)
    eval_ds = SentimentDataset(d["eval_texts"], d["eval_labels"], tok)

    args = TrainingArguments(
        output_dir=os.path.join(config.ROOT, "checkpoints"),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=config.LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=config.SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)

    t0 = time.time()
    trainer.train()
    seconds = round(time.time() - t0, 1)

    preds = trainer.predict(eval_ds).predictions.argmax(-1)
    y = d["eval_labels"]
    finetuned = score(y, preds)

    # ---- Save model ----
    trainer.save_model(config.MODEL_DIR)
    tok.save_pretrained(config.MODEL_DIR)

    # ---- Record results ----
    results = {
        "data_source": d["source"],
        "train_examples": len(d["train_texts"]),
        "eval_examples": len(d["eval_texts"]),
        "model": config.MODEL_NAME,
        "epochs": config.EPOCHS,
        "training_seconds": seconds,
        "majority_class": baselines["majority_class"],
        "tfidf_logreg": baselines["tfidf_logreg"],
        "finetuned_distilbert": finetuned,
    }
    with open(os.path.join(config.RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(config.RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(classification_report(y, preds, labels=list(range(len(config.LABELS))),
                                      target_names=config.LABELS, digits=3, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y, preds, labels=list(range(len(config.LABELS))),
                                            display_labels=config.LABELS, cmap="Blues")
    plt.title("Fine-tuned DistilBERT: confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    log = trainer.state.log_history
    steps = [e["step"] for e in log if "loss" in e]
    losses = [e["loss"] for e in log if "loss" in e]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses)
    plt.xlabel("step"); plt.ylabel("training loss"); plt.title("Fine-tuning loss")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "loss_curve.png"), dpi=120)
    plt.close()

    print(json.dumps(results, indent=2))
    print("Saved model to", config.MODEL_DIR, "and results to", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
