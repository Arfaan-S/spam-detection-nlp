"""Load both datasets into one common format: columns 'text' and 'label' (0 = legitimate/ham, 1 = spam)."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
LABEL_MAP = {"ham": 0, "spam": 1}


def load_sms() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "sms_spam.csv", encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    df["label"] = df["label"].map(LABEL_MAP)
    return df.dropna().reset_index(drop=True)


def load_enron() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DIR / "enron_spam_data.csv")
    # Subject and body are joined, because both contain spam signals
    text = raw["Subject"].fillna("") + " . " + raw["Message"].fillna("")
    df = pd.DataFrame({"text": text.str.strip(), "label": raw["Spam/Ham"].map(LABEL_MAP)})
    df = df[df["text"].str.len() > 3]
    return df.dropna().reset_index(drop=True)
