"""Simple baselines to compare the fine-tuned model against (no deep learning)."""
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline


def score(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def run_baselines(train_texts, train_labels, eval_texts, eval_labels):
    # 1) always predict the most common class
    majority = Counter(train_labels).most_common(1)[0][0]
    majority_preds = [majority] * len(eval_labels)

    # 2) TF-IDF + logistic regression
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )
    clf.fit(train_texts, train_labels)
    tfidf_preds = clf.predict(eval_texts)

    return {
        "majority_class": score(eval_labels, majority_preds),
        "tfidf_logreg": score(eval_labels, tfidf_preds),
    }
