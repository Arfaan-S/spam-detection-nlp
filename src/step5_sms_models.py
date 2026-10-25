"""Step 5 - Train and compare models on the SMS data.

Design (to keep the test set honest):
  A. Pre-processing ablation  : 5-fold cross-validation on the TRAINING set only (default model settings).
  B. Model selection          : grid search with 5-fold cross-validation on the TRAINING set only.
  C. Final evaluation         : each tuned model is evaluated ONCE on the untouched test set.
Outputs: results/tables/{ablation_preprocessing_sms,model_comparison_sms,errors_sms,top_features_sms}.csv,
         results/figures/{fig_confusion_matrices_sms,fig_top_features_sms}.png, models/spam_model_sms.joblib
"""
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_sms
from evaluation import HAM_C, SPAM_C, bootstrap_f1_ci, metrics
from features import build_tfidf
from preprocessing import preprocess

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
TAB, FIG, MOD = ROOT / "results" / "tables", ROOT / "results" / "figures", ROOT / "models"
for d in (TAB, FIG, MOD):
    d.mkdir(parents=True, exist_ok=True)
CACHE = joblib.Memory(str(ROOT / "data" / "processed" / "_cache"), verbose=0)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- data
raw = load_sms()
train = pd.read_csv(ROOT / "data" / "processed" / "sms_train.csv")
test = pd.read_csv(ROOT / "data" / "processed" / "sms_test.csv")
raw_train, raw_test = raw.loc[train.orig_idx, "text"].tolist(), raw.loc[test.orig_idx, "text"].tolist()
y_train, y_test = train.label.values, test.label.values
print(f"train {len(train)} (spam {y_train.sum()}), test {len(test)} (spam {y_test.sum()})")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
MODELS = {
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(random_state=SEED),
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED),
}

# ---------------------------------------------------------------- A. pre-processing ablation
VARIANTS = {
    "A lowercase only": lambda t: preprocess(t, "none", remove_stop=False, placeholders=False),
    "B + placeholders (url, money, number)": lambda t: preprocess(t, "none", remove_stop=False),
    "C + stop words removed": lambda t: preprocess(t, "none", remove_stop=True),
    "D + lemmatized": lambda t: preprocess(t, "lemma"),
    "E + stemmed (instead of D)": lambda t: preprocess(t, "stem"),
}
rows = []
for vname, fn in VARIANTS.items():
    texts = [fn(t) for t in raw_train]
    for mname, clf in MODELS.items():
        pipe = Pipeline([("tfidf", build_tfidf()), ("clf", clf)])
        sc = cross_val_score(pipe, texts, y_train, cv=cv, scoring="f1", n_jobs=2)
        rows.append({"variant": vname, "model": mname, "cv_f1_mean": sc.mean(), "cv_f1_std": sc.std()})
abl = pd.DataFrame(rows)
abl.to_csv(TAB / "ablation_preprocessing_sms.csv", index=False)
wide = abl.pivot(index="variant", columns="model", values="cv_f1_mean")
wide["mean"] = wide.mean(axis=1)
print("\nA. Ablation (CV F1, training data only)\n", wide.round(4).to_string())
best_variant = wide["mean"].idxmax()
print("-> variant with best mean CV F1:", best_variant)

# ---------------------------------------------------------------- B. grid search per model
texts_tr = [VARIANTS[best_variant](t) for t in raw_train]
texts_te = [VARIANTS[best_variant](t) for t in raw_test]
GRIDS = {
    "Naive Bayes": {"clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]},
    "Linear SVM": {"clf__C": [0.1, 0.3, 1, 3, 10], "clf__class_weight": [None, "balanced"]},
    "Logistic Regression": {"clf__C": [1, 3, 10, 30, 100, 300, 1000], "clf__class_weight": [None, "balanced"]},
}
results, fitted, preds = [], {}, {}
for mname, clf in MODELS.items():
    pipe = Pipeline([("tfidf", build_tfidf()), ("clf", clf)], memory=CACHE)
    grid = {"tfidf__ngram_range": [(1, 1), (1, 2)], **GRIDS[mname]}
    gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=2, refit=True).fit(texts_tr, y_train)
    t0 = time.perf_counter(); pred = gs.predict(texts_te); t_pred = (time.perf_counter() - t0) / len(texts_te)
    m = metrics(y_test, pred); lo, hi = bootstrap_f1_ci(y_test, pred)
    results.append({"model": mname, "best_params": str(gs.best_params_), "cv_f1": gs.best_score_,
                    **{k: m[k] for k in ["accuracy", "precision", "recall", "f1"]},
                    "f1_ci95_low": lo, "f1_ci95_high": hi, **{k: m[k] for k in ["TP", "FP", "FN", "TN"]},
                    "train_seconds": gs.refit_time_, "predict_ms_per_message": 1000 * t_pred})
    fitted[mname], preds[mname] = gs.best_estimator_, pred
    print(f"{mname}: CV F1 {gs.best_score_:.4f} | test F1 {m['f1']:.4f} acc {m['accuracy']:.4f} | {gs.best_params_}")
res = pd.DataFrame(results)
res.to_csv(TAB / "model_comparison_sms.csv", index=False)

# paired comparison of the models on the same test messages: exact McNemar test on the discordant pairs
from itertools import combinations
from scipy.stats import binomtest
pair_rows = []
for a, b in combinations(MODELS, 2):
    ok_a, ok_b = preds[a] == y_test, preds[b] == y_test
    only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p_val = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    pair_rows.append({"model_A": a, "model_B": b, "only_A_correct": only_a, "only_B_correct": only_b, "p_value": p_val})
pd.DataFrame(pair_rows).to_csv(TAB / "pairwise_mcnemar_sms.csv", index=False)
print("\nPairwise McNemar tests\n", pd.DataFrame(pair_rows).round(4).to_string(index=False))

# ---------------------------------------------------------------- C. figures / analysis
# confusion matrices
cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", HAM_C])
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5))
for ax, r in zip(axes, results):
    cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n({100 * share:.1f} %)", ha="center", va="center", fontsize=8,
                    color="white" if share > 0.55 else "#0b0b0b")
    ax.set_xticks([0, 1], ["Legit.", "Spam"]); ax.set_yticks([0, 1], ["Legit.", "Spam"])
    ax.set_xlabel("Predicted"); ax.set_title(r["model"], fontsize=9, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel("Actual")
fig.tight_layout(); fig.savefig(FIG / "fig_confusion_matrices_sms.png", dpi=220); plt.close(fig)

# top features of the logistic regression (interpretability)
lr = fitted["Logistic Regression"]
names = np.array(lr.named_steps["tfidf"].get_feature_names_out()); coef = lr.named_steps["clf"].coef_[0]
order = np.argsort(coef)
top_spam, top_ham = order[::-1][:15], order[:15]
pd.DataFrame({"spam_feature": names[top_spam], "spam_weight": coef[top_spam],
              "ham_feature": names[top_ham], "ham_weight": coef[top_ham]}).to_csv(TAB / "top_features_sms.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
for ax, idx, col, title in [(axes[0], top_ham[::-1], HAM_C, "Indicates legitimate"), (axes[1], top_spam[::-1], SPAM_C, "Indicates spam")]:
    ax.barh(names[idx], np.abs(coef[idx]), color=col, height=0.65)
    ax.set_title(title, fontsize=9, loc="left"); ax.set_xlabel("Weight (absolute value)")
fig.tight_layout(); fig.savefig(FIG / "fig_top_features_sms.png", dpi=220); plt.close(fig)

# error analysis for the model with the best CROSS-VALIDATION score (not the best test score)
final_name = res.loc[res.cv_f1.idxmax(), "model"]
fin = fitted[final_name]; pred = preds[final_name]
err = pd.DataFrame({"actual": y_test, "predicted": pred, "text": raw_test})
err = err[err.actual != err.predicted].assign(type=lambda d: np.where(d.actual == 0, "false positive (legit flagged)", "false negative (spam missed)"))
err.to_csv(TAB / "errors_sms.csv", index=False)
joblib.dump({"pipeline": fin, "variant": best_variant, "model": final_name}, MOD / "spam_model_sms.joblib")
print("\nFinal model (best CV):", final_name, "| errors on test:", len(err))
print(res.drop(columns=["best_params"]).round(4).to_string(index=False))
print(err.to_string(max_colwidth=110))
