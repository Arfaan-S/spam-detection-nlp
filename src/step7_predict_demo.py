"""Step 7 - Prediction demo: classify NEW messages with the saved models.

Usage:
    python3 src/step7_predict_demo.py                      # built-in example messages
    python3 src/step7_predict_demo.py "your message here"  # classify your own text (SMS and e-mail model)

The saved models were trained on text pre-processed with stop-word removal + stemming,
so every new message must go through exactly the same function (mode="stem").
"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import preprocess

MOD = Path(__file__).resolve().parents[1] / "models"
MODELS = {"SMS model (Logistic Regression)": joblib.load(MOD / "spam_model_sms.joblib")["pipeline"],
          "E-mail model (Linear SVM)": joblib.load(MOD / "spam_model_enron.joblib")["pipeline"]}

EXAMPLES = {
    "SMS model (Logistic Regression)": [
        "Hey, are you coming to dinner tonight? Let me know.",
        "WINNER!! You have won a free prize. Call 09061701461 now to claim your reward!",
        "Meeting moved to 3pm, see you in the conference room.",
        "Congratulations! Text WIN to 80082 to get your free ringtone, 1.50 per week."],
    "E-mail model (Linear SVM)": [
        "Subject: Q3 report. Please find the attached quarterly report. Let me know if you have questions.",
        "Subject: Cheap software online. Buy medications and software at low price, click http://example.com to order now.",
        "Subject: Schedule for Monday. The meeting is confirmed for 10 am, thanks.",
        "Subject: Investment opportunity. Remove your name from this list; earn money fast, click here."],
}


def classify(pipe, message):
    processed = preprocess(message, mode="stem")
    score = float(pipe.decision_function([processed])[0])        # > 0 means spam; larger = more confident
    return ("SPAM" if score > 0 else "HAM"), score


def show(name, messages):
    print(f"\n{name}\n" + "-" * 70)
    for msg in messages:
        label, score = classify(MODELS[name], msg)
        print(f"{label:4s} (score {score:+.2f})  {msg[:75]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for name in MODELS:
            show(name, [" ".join(sys.argv[1:])])
    else:
        for name, msgs in EXAMPLES.items():
            show(name, msgs)
