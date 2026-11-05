"""Load the financial sentiment data (Hugging Face dataset, or the tiny local CSV as a fallback)."""
import csv
import random

import config


def _from_csv():
    by_label = {i: [] for i in range(len(config.LABELS))}
    with open(config.SAMPLE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_label[config.LABELS.index(row["label"])].append(row["text"])
    tr_x, tr_y, ev_x, ev_y = [], [], [], []
    for label, texts in by_label.items():
        cut = int(len(texts) * 2 / 3)          # first 2/3 train, last 1/3 eval, per class
        tr_x += texts[:cut]; tr_y += [label] * cut
        ev_x += texts[cut:]; ev_y += [label] * (len(texts) - cut)
    return tr_x, tr_y, ev_x, ev_y


def _from_hub():
    from datasets import load_dataset
    ds = load_dataset(config.DATASET)
    tr, ev = ds["train"], ds["validation"]
    names = getattr(tr.features["label"], "names", None)
    mapping = {n.lower(): i for i, n in enumerate(config.LABELS)}
    if names:   # make sure label ids line up with config.LABELS
        remap = [mapping[n.lower()] for n in names]
    else:
        remap = list(range(len(config.LABELS)))
    tr_y = [remap[int(v)] for v in tr["label"]]
    ev_y = [remap[int(v)] for v in ev["label"]]
    return list(tr["text"]), tr_y, list(ev["text"]), ev_y


def _subsample(x, y, n, seed):
    if n is None or n >= len(x):
        return x, y
    idx = random.Random(seed).sample(range(len(x)), n)
    return [x[i] for i in idx], [y[i] for i in idx]


def load_data():
    try:
        tr_x, tr_y, ev_x, ev_y = _from_hub()
        source = config.DATASET
    except Exception as e:
        print(f"[data] Could not load '{config.DATASET}' ({type(e).__name__}: {e})")
        print("[data] Falling back to the tiny local sample file. Results will NOT be representative.")
        tr_x, tr_y, ev_x, ev_y = _from_csv()
        source = "local data/sample_data.csv (fallback)"
    tr_x, tr_y = _subsample(tr_x, tr_y, config.TRAIN_SAMPLES, config.SEED)
    ev_x, ev_y = _subsample(ev_x, ev_y, config.EVAL_SAMPLES, config.SEED)
    print(f"[data] source: {source} | train: {len(tr_x)} | eval: {len(ev_x)}")
    return {"train_texts": tr_x, "train_labels": tr_y, "eval_texts": ev_x, "eval_labels": ev_y, "source": source}
