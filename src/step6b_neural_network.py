"""Step 6b - A small deep-learning model (CNN + BiLSTM) compared with the classical models.

Architecture (a reduced version of the CNN+BiLSTM idea of Huan et al., 2022 and Elakkiya et al., 2021):
    word ids -> Embedding(64) -> Conv1d(k=3, 64 filters) -> BiLSTM(32 per direction) -> global max-pool -> Dense -> spam score
Everything is learned from scratch on our own training data (no pretrained vectors, no internet needed).

Method (same honesty rules as before):
 * vocabulary built on the TRAINING part only (words seen >= 2 times), unknown words -> <unk>
 * 10 % of the training part is held out as a validation set; the epoch with the best validation F1 is kept (early stopping)
 * the test set is used exactly once at the end
 * 3 random seeds per dataset -> mean +- standard deviation (neural nets are random!)
"""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import metrics

ROOT = Path(__file__).resolve().parents[1]
TAB, MOD, PROC = ROOT / "results" / "tables", ROOT / "models", ROOT / "data" / "processed"
torch.set_num_threads(4)
D = {k: {p: pd.read_csv(PROC / f"{k}_{p}.csv").fillna("") for p in ("train", "test")} for k in ("sms", "enron")}
MAXLEN = {"sms": 40, "enron": 200}   # tokens kept per message (SMS are short, e-mails long)
EPOCHS, BATCH, LR = 12, 64, 2e-3


def build_vocab(texts, min_count=2):
    c = Counter(w for t in texts for w in t.split())
    itos = ["<pad>", "<unk>"] + sorted(w for w, n in c.items() if n >= min_count)
    return {w: i for i, w in enumerate(itos)}


def encode(texts, vocab, maxlen):
    X = np.zeros((len(texts), maxlen), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in t.split()][:maxlen]
        X[i, :len(ids)] = ids
    return torch.from_numpy(X)


class CnnBiLstm(nn.Module):
    def __init__(self, vocab_size, emb=64, filters=64, hidden=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.conv = nn.Conv1d(emb, filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(filters, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.4)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        h = self.emb(x)                                  # (batch, len, emb)
        h = torch.relu(self.conv(h.transpose(1, 2)))     # (batch, filters, len)
        h, _ = self.lstm(h.transpose(1, 2))              # (batch, len, 2*hidden)
        h = h.masked_fill((x == 0).unsqueeze(-1), -1e9).max(dim=1).values   # max-pool over real tokens only
        return self.out(self.drop(h)).squeeze(-1)


def predict_scores(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 512]) for i in range(0, len(X), 512)]).numpy()


def run(key, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, te = D[key]["train"], D[key]["test"]
    t_fit, t_val = train_test_split(tr, test_size=0.1, stratify=tr["label"], random_state=seed)
    vocab = build_vocab(t_fit["text_lemma"])
    L = MAXLEN[key]
    Xf, Xv, Xt = (encode(d["text_lemma"], vocab, L) for d in (t_fit, t_val, te))
    yf = torch.tensor(t_fit["label"].values, dtype=torch.float32)
    yv, yt = t_val["label"].values, te["label"].values
    model = CnnBiLstm(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, best_ep = -1, None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xf))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(); lossf(model(Xf[b]), yf[b]).backward(); opt.step()
        f1v = metrics(yv, (predict_scores(model, Xv) > 0).astype(int))["f1"]
        if f1v > best:
            best, best_ep = f1v, ep + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    s = predict_scores(model, Xt)
    m = metrics(yt, (s > 0).astype(int)); m["roc_auc"] = roc_auc_score(yt, s)
    m.update(dataset=key, seed=seed, best_epoch=best_ep, val_f1=best, train_seconds=train_time,
             n_params=sum(p.numel() for p in model.parameters()), vocab=len(vocab))
    return m, model, vocab


if __name__ == "__main__":
    rows, best_by_key = [], {}
    for key in ("sms", "enron"):
        for seed in (1, 2, 3):
            m, model, vocab = run(key, seed)
            rows.append(m)
            print(f"[{key}] seed {seed}: test F1 {m['f1']:.4f} P {m['precision']:.4f} R {m['recall']:.4f} "
                  f"AUC {m['roc_auc']:.4f} | best epoch {m['best_epoch']} | {m['train_seconds']:.0f}s", flush=True)
            if key not in best_by_key or m["val_f1"] > best_by_key[key][0]:   # pick by VALIDATION score, not test
                best_by_key[key] = (m["val_f1"], model, vocab)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "neural_network_runs.csv", index=False)
    summ = df.groupby("dataset")[["accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds", "best_epoch"]].agg(["mean", "std"]).round(4)
    summ.to_csv(TAB / "neural_network_summary.csv")
    print(summ.T.to_string())
    for key, (_, model, vocab) in best_by_key.items():
        torch.save({"state_dict": model.state_dict(), "vocab": vocab, "maxlen": MAXLEN[key]}, MOD / f"spam_nn_{key}.pt")
    print("done")
