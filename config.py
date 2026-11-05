import os

ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "distilbert-base-uncased"                  # small, fast BERT
DATASET = "zeroshot/twitter-financial-news-sentiment"   # Hugging Face dataset (train / validation)
SAMPLE_CSV = os.path.join(ROOT, "data", "sample_data.csv")   # tiny offline fallback
MODEL_DIR = os.path.join(ROOT, "models", "fin-sentiment")    # fine-tuned model is saved here
RESULTS_DIR = os.path.join(ROOT, "results")

# Same order as the dataset's label ids: 0 = Bearish, 1 = Bullish, 2 = Neutral
LABELS = ["Bearish", "Bullish", "Neutral"]

TRAIN_SAMPLES = 3000      # None = use the full training set (slower)
EVAL_SAMPLES = None       # None = use the full validation set
EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LEN = 64
SEED = 42
