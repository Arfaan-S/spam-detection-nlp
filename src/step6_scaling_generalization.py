"""Step 6 - Scaling up and generalization.

 1. Tune NB / SVM / LR on the large Enron training set (5-fold CV) and evaluate once on the Enron test set.
 2. Learning curves: growing training sizes (SMS 100 -> 4,040; Enron 250 -> 23,179), 3 random subsamples each.
 3. Cross-dataset tests: train on SMS -> test on Enron, train on Enron -> test on SMS, train on both -> test on each.
 4. Bias check: remove company-specific tokens (enron, ect, hou, corp, vince) from the Enron data.
Pre-processing variant: stop words removed + stemming (selected by cross-validation in step 5).
"""
import ast
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import bootstrap_f1_ci, metrics
from features import build_tfidf
from preprocessing import stem

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
TAB, FIG, MOD, PROC = ROOT / "results" / "tables", ROOT / "results" / "figures", ROOT / "models", ROOT / "data" / "processed"
CACHE = joblib.Memory(str(PROC / "_cache"), verbose=0)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
MODEL_COLORS = {"Naive Bayes": "#2a78d6", "Linear SVM": "#eb6834", "Logistic Regression": "#1baf7a"}
MARKERS = {"Naive Bayes": "o", "Linear SVM": "s", "Logistic Regression": "^"}

D = {k: {p: pd.read_csv(PROC / f"{k}_{p}.csv").fillna("") for p in ("train", "test")} for k in ("sms", "enron")}
NAMES = {"sms": "SMS", "enron": "Enron"}


def base_models():
    return {"Naive Bayes": MultinomialNB(), "Linear SVM": LinearSVC(random_state=SEED),
            "Logistic Regression": LogisticRegression(max_iter=3000, solver="liblinear", random_state=SEED)}


def score(pipe, X, y):
    """Metrics + ROC-AUC (threshold-free ranking quality)."""
    pred = pipe.predict(X)
    s = pipe.predict_proba(X)[:, 1] if hasattr(pipe[-1], "predict_proba") and isinstance(pipe[-1], MultinomialNB) else pipe.decision_function(X)
    m = metrics(y, pred); m["roc_auc"] = roc_auc_score(y, s)
    return m, pred


def make_pipe(name, params, memory=None):
    p = Pipeline([("tfidf", build_tfidf()), ("clf", base_models()[name])], memory=memory)
    return p.set_params(**params)


# ------------------------------------------------------------------ 1. tune on Enron, evaluate once
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
GRIDS = {"Naive Bayes": {"clf__alpha": [0.1, 0.5, 1.0]},
         "Linear SVM": {"clf__C": [0.3, 1, 3]},
         "Logistic Regression": {"clf__C": [3, 10, 30]}}
best = {"sms": {}, "enron": {}}
sms_cmp = pd.read_csv(TAB / "model_comparison_sms.csv")
for _, r in sms_cmp.iterrows():
    best["sms"][r.model] = ast.literal_eval(r.best_params)          # tuned in step 5
rows, enron_models, t_all = [], {}, time.time()
Xtr, ytr, Xte, yte = D["enron"]["train"].text_stem, D["enron"]["train"].label.values, D["enron"]["test"].text_stem, D["enron"]["test"].label.values
for name in base_models():
    pipe = Pipeline([("tfidf", build_tfidf()), ("clf", base_models()[name])], memory=CACHE)
    gs = GridSearchCV(pipe, {"tfidf__ngram_range": [(1, 1), (1, 2)], **GRIDS[name]}, scoring="f1", cv=cv, n_jobs=2).fit(Xtr, ytr)
    best["enron"][name] = gs.best_params_
    m, pred = score(gs.best_estimator_, Xte, yte); lo, hi = bootstrap_f1_ci(yte, pred)
    rows.append({"model": name, "best_params": str(gs.best_params_), "cv_f1": gs.best_score_, **m, "f1_ci95_low": lo,
                 "f1_ci95_high": hi, "train_seconds": gs.refit_time_})
    enron_models[name] = gs.best_estimator_
    print(f"[Enron] {name}: CV F1 {gs.best_score_:.4f} | test F1 {m['f1']:.4f} acc {m['accuracy']:.4f} {gs.best_params_}  ({time.time() - t_all:.0f}s)", flush=True)
pd.DataFrame(rows).to_csv(TAB / "model_comparison_enron.csv", index=False)
final_enron = pd.DataFrame(rows).sort_values("cv_f1").iloc[-1].model
joblib.dump({"pipeline": enron_models[final_enron], "variant": "E + stemmed", "model": final_enron}, MOD / "spam_model_enron.joblib")

# ------------------------------------------------------------------ 2. learning curves
SIZES = {"sms": [100, 250, 500, 1000, 2000, 4040], "enron": [250, 500, 1000, 2500, 5000, 10000, 23179]}
lc_rows = []
for key in ("sms", "enron"):
    tr, te = D[key]["train"], D[key]["test"]
    for n in SIZES[key]:
        for rep in range(1 if n >= len(tr) else 3):
            idx = tr.index if n >= len(tr) else train_test_split(tr.index, train_size=n, stratify=tr.label, random_state=SEED + rep)[0]
            sub = tr.loc[idx]
            for name in base_models():
                pipe = make_pipe(name, best[key][name]); t0 = time.perf_counter()
                pipe.fit(sub.text_stem, sub.label); fit_s = time.perf_counter() - t0
                m, _ = score(pipe, te.text_stem, te.label.values)
                lc_rows.append({"dataset": NAMES[key], "train_size": n, "repeat": rep, "model": name, "f1": m["f1"],
                                "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                                "fit_seconds": fit_s})
    print(f"learning curve {key} done ({time.time() - t_all:.0f}s)", flush=True)
lc = pd.DataFrame(lc_rows); lc.to_csv(TAB / "learning_curve.csv", index=False)
agg = lc.groupby(["dataset", "model", "train_size"]).agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                                                        acc_mean=("accuracy", "mean"), fit_seconds=("fit_seconds", "mean")).reset_index()
agg.to_csv(TAB / "learning_curve_summary.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), sharey=False)
for ax, ds in zip(axes, ("SMS", "Enron")):
    for name in base_models():
        a = agg[(agg.dataset == ds) & (agg.model == name)].sort_values("train_size")
        ax.plot(a.train_size, a.f1_mean, color=MODEL_COLORS[name], marker=MARKERS[name], ms=4.5, lw=1.6, label=name)
        ax.fill_between(a.train_size, a.f1_mean - a.f1_std.fillna(0), a.f1_mean + a.f1_std.fillna(0), color=MODEL_COLORS[name], alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_title(f"{ds} test set", fontsize=9.5, loc="left")
    ax.set_xlabel("Training messages (log scale)"); ax.set_ylabel("F1 score (spam)")
    ax.yaxis.grid(True, color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "fig_learning_curves.png", dpi=220); plt.close(fig)

# ------------------------------------------------------------------ 3. cross-dataset generalization
cross = []
train_sets = {"SMS": [D["sms"]["train"]], "Enron": [D["enron"]["train"]], "SMS + Enron": [D["sms"]["train"], D["enron"]["train"]]}
for tname, parts in train_sets.items():
    tr = pd.concat(parts, ignore_index=True)
    for name in base_models():
        params = best["sms"][name] if tname == "SMS" else best["enron"][name]
        pipe = make_pipe(name, params).fit(tr.text_stem, tr.label)
        for ename in ("SMS", "Enron"):
            te = D[ename.lower()]["test"]
            m, _ = score(pipe, te.text_stem, te.label.values)
            cross.append({"trained_on": tname, "tested_on": ename, "model": name, "in_domain": tname == ename,
                          **{k: m[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "FP", "FN"]}})
cross = pd.DataFrame(cross); cross.to_csv(TAB / "cross_dataset.csv", index=False)
print("\nCross-dataset (F1 / AUC)\n", cross.pivot_table(index=["trained_on", "model"], columns="tested_on", values=["f1", "roc_auc"]).round(3).to_string(), flush=True)

# ------------------------------------------------------------------ 4. bias check (Enron company-specific tokens)
BIAS = {stem(w) for w in ["enron", "ect", "hou", "corp", "vince"]}
strip = lambda s: " ".join(t for t in s.split() if t not in BIAS)
bias_rows = []
for name in base_models():
    for label, f in [("original", lambda s: s), ("company tokens removed", strip)]:
        pipe = make_pipe(name, best["enron"][name]).fit(D["enron"]["train"].text_stem.map(f), D["enron"]["train"].label)
        m, _ = score(pipe, D["enron"]["test"].text_stem.map(f), D["enron"]["test"].label.values)
        # transfer: the same Enron model applied to SMS
        m2, _ = score(pipe, D["sms"]["test"].text_stem.map(f), D["sms"]["test"].label.values)
        bias_rows.append({"model": name, "enron_text": label, "enron_f1": m["f1"], "enron_acc": m["accuracy"],
                          "sms_f1": m2["f1"], "sms_auc": m2["roc_auc"]})
bias = pd.DataFrame(bias_rows); bias.to_csv(TAB / "bias_check_enron.csv", index=False)
print("\nBias check\n", bias.round(4).to_string(index=False))

lr = enron_models["Logistic Regression"]
names = np.array(lr.named_steps["tfidf"].get_feature_names_out()); coef = lr.named_steps["clf"].coef_[0]; o = np.argsort(coef)
pd.DataFrame({"spam_feature": names[o[::-1][:20]], "spam_weight": coef[o[::-1][:20]],
              "ham_feature": names[o[:20]], "ham_weight": coef[o[:20]]}).to_csv(TAB / "top_features_enron.csv", index=False)
print("\nTop Enron features (spam):", list(names[o[::-1][:12]]), "\n(ham):", list(names[o[:12]]))
print(f"\ntotal {time.time() - t_all:.0f}s")
