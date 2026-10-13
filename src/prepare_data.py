"""Step 4 - Pre-process both datasets, remove duplicates, split into train/test, save results.

Outputs:
  data/processed/<name>_train.csv, <name>_test.csv   (columns: orig_idx, label, text_lemma, text_stem)
  results/tables/preprocessing_examples.csv           (before/after examples)
  results/tables/vocabulary_by_stage.csv              (vocabulary size after each pre-processing step)
  results/tables/split_overview.csv                   (sizes after de-duplication and split)
  results/tables/tfidf_encoding_summary.csv           (size and sparsity of the TF-IDF matrices)
"""
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_enron, load_sms
from features import build_tfidf
from preprocessing import STOP_WORDS, clean, lemmatize, preprocess, stem, tokenize

SEED = 42
TEST_SIZE = 0.20
ROOT = Path(__file__).resolve().parents[1]
PROC, TAB = ROOT / "data" / "processed", ROOT / "results" / "tables"
PROC.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

DATASETS = {"sms": ("SMS Spam Collection", load_sms), "enron": ("Enron-Spam", load_enron)}
vocab_rows, split_rows, tfidf_rows, example_rows = [], [], [], []

for key, (name, loader) in DATASETS.items():
    t0 = time.time()
    df = loader()
    df["orig_idx"] = df.index
    n_all = len(df)
    df = df.drop_duplicates("text", keep="first").reset_index(drop=True)   # remove exact duplicates
    n_unique = len(df)

    # --- vocabulary size after each pre-processing step (descriptive statistics) ---
    raw_tokens = [w for t in df.text for w in t.lower().split()]
    cleaned = [tokenize(t) for t in df.text]
    no_stop = [[w for w in toks if w not in STOP_WORDS] for toks in cleaned]
    lemmas = [[lemmatize(w) for w in toks] for toks in no_stop]
    stems = [[stem(w) for w in toks] for toks in no_stop]
    for stage, toks in [("0 raw (lowercase, split on spaces)", raw_tokens),
                        ("1 cleaned (placeholders, letters only)", [w for d in cleaned for w in d]),
                        ("2 + stop words removed", [w for d in no_stop for w in d]),
                        ("3a + lemmatized", [w for d in lemmas for w in d]),
                        ("3b + stemmed (alternative to 3a)", [w for d in stems for w in d])]:
        vocab_rows.append({"dataset": name, "stage": stage, "tokens": len(toks), "vocabulary": len(set(toks))})

    df["text_lemma"] = [" ".join(d) for d in lemmas]
    df["text_stem"] = [" ".join(d) for d in stems]
    n_before = len(df)
    df = df[df.text_lemma.str.len() > 0]                                  # drop messages that became empty
    n_empty = n_before - len(df)
    # near-duplicates: different raw texts that become identical after cleaning (e.g. only a phone number differs)
    n_before = len(df)
    df = df.drop_duplicates("text_lemma", keep="first").reset_index(drop=True)
    n_near = n_before - len(df)

    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df.label, random_state=SEED)
    for part, d in [("train", train), ("test", test)]:
        d[["orig_idx", "label", "text_lemma", "text_stem"]].to_csv(PROC / f"{key}_{part}.csv", index=False)
        split_rows.append({"dataset": name, "part": part, "messages": len(d), "spam": int(d.label.sum()),
                           "spam_share_%": round(100 * d.label.mean(), 1)})
    for label, n in [("removed: exact duplicates", n_all - n_unique), ("removed: empty after cleaning", n_empty),
                     ("removed: near-duplicates after cleaning", n_near)]:
        split_rows.append({"dataset": name, "part": label, "messages": n, "spam": "", "spam_share_%": ""})

    # --- encoding: TF-IDF fitted on the TRAINING part only ---
    for ngr in [(1, 1), (1, 2)]:
        vec = build_tfidf(ngram_range=ngr, min_df=2)
        Xtr = vec.fit_transform(train.text_lemma); Xte = vec.transform(test.text_lemma)
        tfidf_rows.append({"dataset": name, "n_gram_range": str(ngr), "train_shape": str(Xtr.shape),
                           "test_shape": str(Xte.shape), "features": Xtr.shape[1],
                           "sparsity_%": round(100 * (1 - Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1])), 2)})
    print(f"{name}: done in {time.time() - t0:.0f}s")


# examples: use SMS only (short and public), computed from the raw loader output
sms = load_sms(); sms = sms.drop_duplicates("text").reset_index(drop=True)
short = sms[sms.text.str.len().between(40, 110)]
for lab, tag in [(0, "legitimate"), (1, "spam")]:
    for _, r in short[short.label == lab].sample(2, random_state=SEED).iterrows():
        example_rows.append({"class": tag, "original": r.text, "cleaned": " ".join(tokenize(r.text)),
                             "lemmatized": preprocess(r.text, "lemma"), "stemmed": preprocess(r.text, "stem")})

pd.DataFrame(vocab_rows).to_csv(TAB / "vocabulary_by_stage.csv", index=False)
pd.DataFrame(split_rows).to_csv(TAB / "split_overview.csv", index=False)
pd.DataFrame(tfidf_rows).to_csv(TAB / "tfidf_encoding_summary.csv", index=False)
pd.DataFrame(example_rows).to_csv(TAB / "preprocessing_examples.csv", index=False)
for f in ["vocabulary_by_stage", "split_overview", "tfidf_encoding_summary"]:
    print("\n" + f); print(pd.read_csv(TAB / f"{f}.csv").to_string(index=False))
print("\nexamples"); print(pd.read_csv(TAB / "preprocessing_examples.csv").to_string(index=False))
