"""Streamlit UI:  streamlit run app.py"""
import json
import os

import pandas as pd
import streamlit as st

import config

# Draw the page first, load the model afterwards (avoids a blank screen while loading)
st.set_page_config(page_title="FinSentiment", page_icon="📈")
st.title("📈 FinSentiment")
st.caption("Financial news sentiment (Bearish / Neutral / Bullish) with a fine-tuned DistilBERT.")

EMOJI = {"Bullish": "🟢 Bullish", "Bearish": "🔴 Bearish", "Neutral": "⚪ Neutral"}
DEFAULT = (
    "Tesla shares surge after record quarterly deliveries\n"
    "Fed signals more rate hikes, stocks tumble\n"
    "Company X to hold annual general meeting on Thursday\n"
    "Oil prices slump as demand fears grow"
)


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    from sentiment import load_model
    return load_model()


try:
    # sidebar: recorded results
    res_path = os.path.join(config.RESULTS_DIR, "results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            r = json.load(f)
        st.sidebar.header("Model results")
        st.sidebar.table(pd.DataFrame({
            "Accuracy": {"Majority class": r["majority_class"]["accuracy"],
                         "TF-IDF + LogReg": r["tfidf_logreg"]["accuracy"],
                         "DistilBERT (ours)": r["finetuned_distilbert"]["accuracy"]},
            "Macro-F1": {"Majority class": r["majority_class"]["macro_f1"],
                         "TF-IDF + LogReg": r["tfidf_logreg"]["macro_f1"],
                         "DistilBERT (ours)": r["finetuned_distilbert"]["macro_f1"]},
        }))
        img = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
        if os.path.exists(img):
            st.sidebar.image(img)

    if not os.path.isdir(config.MODEL_DIR):
        st.error("No trained model found. Run `python train.py` first.")
        st.stop()

    text = st.text_area("Paste financial headlines (one per line)", value=DEFAULT, height=160)

    if st.button("Analyze"):
        headlines = [l.strip() for l in text.splitlines() if l.strip()]
        if not headlines:
            st.info("Please enter at least one headline.")
        else:
            from sentiment import predict
            model, tok = get_model()
            preds = predict(model, tok, headlines)
            df = pd.DataFrame({
                "Headline": headlines,
                "Sentiment": [EMOJI[p["label"]] for p in preds],
                "Confidence": [f"{p['confidence']:.0%}" for p in preds],
            })
            st.subheader("Results")
            st.table(df)

            counts = {lab: sum(p["label"] == lab for p in preds) for lab in config.LABELS}
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Bullish", counts["Bullish"])
            c2.metric("⚪ Neutral", counts["Neutral"])
            c3.metric("🔴 Bearish", counts["Bearish"])
            if counts["Bullish"] > counts["Bearish"]:
                st.success("Overall mood: Bullish")
            elif counts["Bearish"] > counts["Bullish"]:
                st.error("Overall mood: Bearish")
            else:
                st.info("Overall mood: Neutral / mixed")
            st.caption("Educational demo only. Not financial advice.")

except Exception as e:  # show errors on the page instead of a blank screen
    st.error("Something went wrong:")
    st.exception(e)
