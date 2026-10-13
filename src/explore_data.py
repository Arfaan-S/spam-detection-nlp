"""Step 3 - Explore both datasets: sizes, class balance, duplicates, text length, frequent words.

Outputs: results/tables/dataset_overview.csv, results/tables/top_words_<dataset>.csv,
         results/figures/fig_class_distribution.png, results/figures/fig_text_length.png
"""
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_enron, load_sms

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "results" / "tables", ROOT / "results" / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

HAM_C, SPAM_C = "#2a78d6", "#eb6834"   # blue = legitimate, orange = spam (validated categorical palette)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})

DATASETS = {"SMS Spam Collection": load_sms(), "Enron-Spam": load_enron()}
STOP = set(stopwords.words("english"))


def words(text):
    return re.findall(r"[a-z']+", text.lower())


rows = []
for name, df in DATASETS.items():
    df["n_words"] = df["text"].map(lambda t: len(words(t)))
    n_dup = int(df.duplicated("text").sum())
    # conflicting duplicates: identical text that appears with both labels
    conflict = int((df.groupby("text")["label"].nunique() > 1).sum())
    rows.append({
        "dataset": name, "messages": len(df), "legitimate": int((df.label == 0).sum()),
        "spam": int((df.label == 1).sum()), "spam_share_%": round(100 * df.label.mean(), 1),
        "exact_duplicates": n_dup, "duplicates_%": round(100 * n_dup / len(df), 1),
        "texts_with_both_labels": conflict,
        "median_words_ham": float(df[df.label == 0].n_words.median()),
        "median_words_spam": float(df[df.label == 1].n_words.median()),
        "mean_words_ham": round(df[df.label == 0].n_words.mean(), 1),
        "mean_words_spam": round(df[df.label == 1].n_words.mean(), 1),
    })
    # most frequent non-stop-words per class
    out = {}
    for lab, tag in [(0, "legitimate"), (1, "spam")]:
        c = Counter(w for t in df[df.label == lab].text for w in words(t) if w not in STOP and len(w) > 2)
        out[tag] = [f"{w} ({n})" for w, n in c.most_common(15)]
    pd.DataFrame(out).to_csv(TAB / f"top_words_{name.split()[0].lower()}.csv", index_label="rank")

overview = pd.DataFrame(rows)
overview.to_csv(TAB / "dataset_overview.csv", index=False)
print(overview.T.to_string())

# ---- Figure 1: class distribution -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
for ax, (name, df) in zip(axes, DATASETS.items()):
    counts = df.label.value_counts().sort_index()
    bars = ax.bar(["Legitimate", "Spam"], counts.values, color=[HAM_C, SPAM_C], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({100 * v / len(df):.1f} %)",
                ha="center", va="bottom", color=INK, fontsize=8)
    ax.set_title(name, fontsize=9.5, color=INK, loc="left")
    ax.set_ylim(0, counts.max() * 1.28)
    ax.set_ylabel("Number of messages")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / "fig_class_distribution.png", dpi=220); plt.close(fig)

# ---- Figure 2: text length -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, (name, df) in zip(axes, DATASETS.items()):
    bins = np.logspace(0, np.log10(max(df.n_words.max(), 10)), 36)
    for lab, col, tag in [(0, HAM_C, "Legitimate"), (1, SPAM_C, "Spam")]:
        ax.hist(df[df.label == lab].n_words.clip(lower=1), bins=bins, histtype="step",
                color=col, linewidth=1.6, label=tag, density=True)
    ax.set_xscale("log"); ax.set_title(name, fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("Words per message (log scale)"); ax.set_ylabel("Share of messages (density)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fig_text_length.png", dpi=220); plt.close(fig)
print("figures and tables saved")
