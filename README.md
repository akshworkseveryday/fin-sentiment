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
