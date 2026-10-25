"""Download the two public datasets used in the project into data/raw/.

SMS Spam Collection  -> small/medium scenario (text messages)
Enron-Spam           -> large scenario (e-mails)

Run once:  python src/download_data.py
"""
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

SOURCES = {
    # A copy of the UCI "SMS Spam Collection" (Almeida & Hidalgo) as a CSV file
    "sms_spam.csv": "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv",
    # Enron-Spam e-mails with ham/spam labels, packaged as one CSV inside a zip file
    "enron_spam_data.zip": "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip",
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = RAW_DIR / name
        if target.exists():
            print(f"[skip] {name} already downloaded")
            continue
        print(f"[download] {name}")
        urllib.request.urlretrieve(url, target)
    zip_path = RAW_DIR / "enron_spam_data.zip"
    if not (RAW_DIR / "enron_spam_data.csv").exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
    print("Done. Files in", RAW_DIR)


if __name__ == "__main__":
    main()
