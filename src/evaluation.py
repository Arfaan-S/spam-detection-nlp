"""Evaluation helpers: metrics, bootstrap confidence interval, confusion-matrix figure."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

HAM_C, SPAM_C = "#2a78d6", "#eb6834"


def metrics(y_true, y_pred) -> dict:
    """Spam is the positive class. FP = legitimate message wrongly flagged as spam, FN = spam that got through."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def bootstrap_f1_ci(y_true, y_pred, n_boot: int = 2000, seed: int = 42):
    """95 % confidence interval of the F1 score by resampling the test set."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], pos_label=1, zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
