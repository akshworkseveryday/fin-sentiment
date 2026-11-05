# 📈 FinSentiment: Financial News Sentiment Analysis

Fine-tunes **DistilBERT** to classify financial headlines as **Bearish / Neutral / Bullish**, compares it with
simple baselines, and serves it in a small **Streamlit** app.

Dataset: [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
(finance-related tweets, 3 labels). If it cannot be downloaded, `data/sample_data.csv` (45 hand-written headlines)
is used as a fallback so the pipeline still runs.

## Files
```
fin-sentiment/
├── app.py            # Streamlit UI
├── train.py          # baselines + fine-tuning + records results
├── data.py           # dataset loading (+ offline fallback)
├── baseline.py       # majority-class and TF-IDF + logistic regression baselines
├── sentiment.py      # load model + predict
├── config.py         # settings
├── data/sample_data.csv
├── results/          # created by train.py
└── requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
python train.py                      # ~2-5 min on a GPU
python sentiment.py "Apple beats earnings estimates"
streamlit run app.py                 # http://localhost:8501
```

## Results
Fill in from `results/results.json` after training.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class (always Neutral) | | |
| TF-IDF + Logistic Regression | | |
| **DistilBERT (fine-tuned)** | | |

Also see `results/confusion_matrix.png`, `results/classification_report.txt`, `results/loss_curve.png`.

## Notes
- The dataset is imbalanced (about 65% Neutral), so **Macro-F1** is more informative than accuracy: it treats all three classes equally.
- Trained on a 3,000-example subset for speed. Set `TRAIN_SAMPLES = None` in `config.py` to use everything.
- Trained on tweets, so it works best on short, headline-style text.
- Educational project. Not financial advice.
