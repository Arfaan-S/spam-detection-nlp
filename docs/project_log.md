# Project log (facts for the report)

## 2026-10-12 – Step 3: data collection and exploration
- Datasets: SMS Spam Collection (5,572 messages; 747 spam = 13.4 %), Enron-Spam (33,665 usable e-mails after removing empty texts; 17,120 spam = 50.9 %).
- Exact duplicate texts: SMS 403 (7.2 %), Enron 3,203 (9.5 %). No text carries both labels.
- Median words per message: SMS ham 10 / spam 24; Enron ham 120 / spam 97.
- Frequent SMS spam words: call, free, txt, mobile, claim, reply, prize, cash, win.
- Frequent Enron legitimate words: enron, ect, hou, vince, corp -> company-specific vocabulary (dataset bias; relevant for cross-dataset generalization).
- Decision: duplicates are removed BEFORE the train/test split to avoid leakage.
- Tooling issue: NLTK downloader blocked in the cloud sandbox; resources fetched directly. (Normal download works on a personal computer.)

## 2026-10-13 – Step 4: pre-processing and encoding
- Pipeline (src/preprocessing.py): lowercase -> strip HTML -> placeholders (urltoken, mailtoken, moneytoken, numtoken) -> alphabetic tokens (>= 2 letters) -> NLTK English stop words (198) -> WordNet lemmatizer OR Porter stemmer.
- Lemmatizer used without POS tags (treats words as nouns) – known limitation.
- Vocabulary (exact-duplicate-free data): SMS 13,496 (raw) -> 7,633 (cleaned) -> 7,497 (stop words) -> 6,989 (lemma) / 6,174 (stem). Enron 159,201 -> 143,308 -> 143,163 -> 136,490 / 118,246.
- Duplicates: exact duplicates removed (SMS 403, Enron 3,203). Check found 28 (SMS) / 204 (Enron) processed texts shared by train and test -> near-duplicates (identical after cleaning) also removed: SMS 110, Enron 1,487; emptied by cleaning: SMS 9 (stop-word-only messages such as "Same to u..."), Enron 1.
- Final: SMS 5,050 messages (train 4,040 / test 1,010, spam 11.6 %); Enron 28,974 emails (train 23,179 / test 5,795, spam 50.0 %). Stratified 80/20 split, random seed 42. Train/test overlap of processed texts: 0.
- Encoding: TF-IDF (sublinear tf, min_df = 2) fitted on training data only. Unigrams: SMS 2,724 features, Enron 56,051; uni+bigrams: SMS 5,593, Enron 395,138; sparsity 99.7-99.96 %.
- Example of stemming being crude: "customer" -> "custom", "early" -> "earli".

## 2026-10-13 – Step 5: models on the SMS data (src/step5_sms_models.py)
Design: all choices on the TRAINING set via 5-fold stratified CV (seed 42); test set evaluated once per tuned model.
- Pre-processing ablation (CV F1, default model settings; mean over NB / SVM / LR): A lowercase only 0.855; B + placeholders 0.886; C + stop words removed 0.886; D + lemmatized 0.890; E stemmed 0.895. Fold standard deviations 0.02-0.05, so D vs E (0.005) is within noise. Rule "highest mean CV F1" selected E (stemming) for all later experiments.
- Observations: placeholders help clearly (LR 0.815 -> 0.889; SVM 0.922 -> 0.940). Stop-word removal lowers SVM (0.940 -> 0.931) and LR (0.889 -> 0.864) but raises NB (0.830 -> 0.862).
- Grid search (TF-IDF unigram vs uni+bigram; NB alpha; SVM/LR C and class_weight). First LR grid ended at the upper edge (C = 100) -> grid extended to 1000; optimum stayed at C = 100 (interior).
- Enron stage uses a SMALLER grid than SMS (NB alpha 0.1/0.5/1; SVM C 0.3/1/3; LR C 3/10/30; no class_weight option) because of the larger data. Best Enron values for Linear SVM (C = 3) and Naive Bayes (alpha = 0.1) lie at the edge of this grid, so slightly better settings may exist (limitation to mention in the report).
- Test set (1,010 messages, 117 spam), spam = positive class:
  NB (alpha 0.1, 1-2-grams): acc 0.9812, P 0.9712, R 0.8632, F1 0.9140 (95 % CI 0.874-0.950), FP 3, FN 16
  Linear SVM (C 3, unigrams): acc 0.9832, P 0.9717, R 0.8803, F1 0.9238 (0.884-0.957), FP 3, FN 14
  Logistic Regression (C 100, balanced, 1-2-grams): acc 0.9871, P 0.9727, R 0.9145, F1 0.9427 (0.910-0.971), FP 3, FN 10
  CV F1: NB 0.9341, SVM 0.9350, LR 0.9382 -> LR selected (best CV, not best test).
- Paired exact McNemar tests: all p > 0.05 (NB-SVM 0.77, NB-LR 0.15, SVM-LR 0.22) -> differences between the models are not statistically significant.
- Speed: training 0.02-0.03 s, prediction about 0.01 ms per message (CPU).
- Errors (13 of 1,010): 3 false positives (very short chat messages), 10 false negatives (e.g. "Missed call alert ...", messages with unusual wording or truncated text).
- Top LR features: spam -> "call numtoken", urltoken, numtoken, txt, mobil, servic, ringtone; legitimate -> ok, hey, lol, gonna. Dataset artifacts visible: "arsen", "divorc" (single spam campaigns), "lt gt" (HTML entity remnants in legitimate messages).
- Model saved: models/spam_model_sms.joblib (pipeline + variant name).

## Step 6: Scaling and Generalization

**Enron Models (tuned on 5-fold CV, tested once)**
- Naive Bayes: test F1 0.9905
- Linear SVM: test F1 0.9938
- Logistic Regression: test F1 0.9907

**Learning Curves**
- SMS: F1 rises from 0.77–0.81 (100 messages) to 0.91–0.94 (4,040 messages). Curve flattens after ~1,000 messages.
- Enron: F1 rises from 0.95–0.96 (250 emails) to 0.99 (23,179 emails). Still improving at full data size.
- All three classical models show similar learning patterns. More data helps with diminishing returns.

**Cross-Dataset Generalization (Key Finding)**
- Train on SMS, test on Enron: F1 0.26–0.50 (fails badly; AUC ≈ 0.59–0.61).
- Train on Enron, test on SMS: F1 0.23–0.26 (fails badly; AUC ≈ 0.68–0.73).
- Train on both: Enron F1 0.98–0.99, SMS F1 0.57 (NB) – 0.81 (SVM); i.e. the Enron score is kept but SMS drops from ≈0.92–0.94 (SMS-only model).
- AUC values show the same gap: 0.59–0.73 for cross-domain vs 0.99 within-domain.
- Interpretation: SMS and email use very different text patterns. Models learn dataset-specific spam signals, not universal ones.

**Bias Check (Enron)**
- Removing company-specific tokens (enron, ect, hou, corp, vince) changed F1 by < 0.003.
- The models learned real spam features (remove, http, software, money, click, invest), not company artifacts.

**Neural Network (CNN+BiLSTM, 3 seeds)**
- SMS: F1 0.912 ± 0.024, AUC 0.986, training time 21s per seed.
- Enron: F1 0.988 ± 0.002, AUC 0.999, training time 527s per seed.
- Comparison: On SMS, Logistic Regression (0.943) outperforms NN (0.912 ± 0.024; seeds 0.934 / 0.887 / 0.914); on Enron, Linear SVM (0.994) slightly outperforms NN (0.988 ± 0.002). NN has lower recall on SMS (0.869).
- Training time: NN 21 s (SMS) / 527 s (Enron) vs. classical models: 0.04–0.1 s (SMS) / 5–12 s (Enron) per fit in the learning-curve runs (learning_curve_summary.csv); final fits with the selected settings (model_comparison_*.csv): 0.02–0.03 s (SMS) and 1.1–6.9 s (Enron).
- Caveat: NN spread over only 3 seeds; classical models have no seed spread on the single final test run, so the std values are not directly comparable.

**Key Takeaways for the Report**
1. More data helps, with diminishing returns (supports Process criterion).
2. Cross-dataset failure shows the problem is harder than a single-dataset score suggests (supports Creativity and Transfer criteria).
3. Classical and neural methods are equivalent here; the choice between them is pragmatic, not principled (supports Quality criterion).

## Step 7: Prediction Demo

A small script (src/step7_predict_demo.py) demonstrates classification on new messages with the trained SMS and Enron models. All 8 hand-written example messages (4 SMS, 4 e-mail) are classified correctly. The text must be pre-processed with mode="stem" like the training data (a first version used lemmatization and wrongly flagged e-mails as spam - an input mismatch, fixed).

## 2026-10-14 – Enron-Spam copy checked against the original paper (Metsis et al., 2006; PDF has 9 pages, no printed page numbers -> PDF page numbers cited)
- Construction (PDF p. 4): ham = mailboxes of six Enron employees (farmer-d, kaminski-v, kitchen-l, williams-w3, beck-s, lokay-m); spam from four sources (SpamAssassin corpus, Honeypot project, Bruce Guenter's collection, spam collected by one author); ham:spam ratio set to about 3:1 in three datasets and 1:3 in the other three by random subsampling; each dataset 5-6 thousand messages. Pre-processing by the authors: removed messages sent by the mailbox owner, HTML tags and headers.
- Table 1 (PDF p. 5): ham:spam per dataset 3672:1500, 4361:1496, 4012:1500, 1500:4500, 1500:3675, 1500:4500. Sums: ham 16,545, spam 17,171, total 33,716 = exactly the GitHub copy used here -> the copy is the six datasets combined.
- Consequence: the nearly balanced class share (50.9 % spam) of our Enron data results from combining datasets with 3:1 and 1:3 ratios.
- SMS Spam Collection (UCI page): spam 425 from Grumbletext forum + 322 from SMS Spam Corpus v0.1 Big; ham 3,375 NUS SMS Corpus + 1,002 SMS Spam Corpus v0.1 Big + 450 from a PhD thesis (Tag).
- Word counts: Tab. 1 counts letter sequences ([a-z']+); a plain space split gives medians 11/25 (SMS) and 176/138 (Enron) - same conclusion.

## 2026-10-14 – Which text variant is used where
- Classical models (NB, SVM, LR) use the STEMMED text (column text_stem): chosen in step 5 by highest mean CV F1 (E stemmed 0.895 vs. D lemmatized 0.890, difference within fold noise 0.02-0.05). Saved models and the step 7 demo also use stemming.
- The CNN + BiLSTM (step6b_neural_network.py) uses the LEMMATIZED text (column text_lemma): design decision (word embeddings work on whole words); NOT tested by CV -> mention as limitation.

## 2026-10-15 – Porter (stemming) source checked
- PDF checked = 2006 republication: Porter, M. F. (2006). An algorithm for suffix stripping. Program: Electronic Library and Information Systems, 40(3), 211-218, https://doi.org/10.1108/00330330610681286 (original work published 1980, Program 14(3), 130-137). Cite as (Porter, 1980/2006, p. 211).
- p. 211: abstract (complex suffixes treated as compounds of simple suffixes, removed in a number of steps) and introduction (CONNECT/CONNECTED/... -> stem). p. 217: step-4 table (ER: airliner -> airlin). p. 218: GENERALIZATIONS stripped step by step.
- Miller (1995), Communications of the ACM, Vol. 38, No. 11, first page = p. 39 (checked on the page image): p. 39 introduces WordNet as an online lexical database in which English nouns, verbs, adjectives and adverbs are organised into sets of synonyms (synsets) linked by semantic relations. Page range 39–41 confirmed on the WordNet site (Princeton, 'Citing WordNet'); DOI 10.1145/219717.219748 seen in two listings, not yet checked on the ACM page.

## 2026-10-25 – Reference and number checks for the report
- Joachims (1998): Springer lists pp. 137-142, DOI 10.1007/BFb0026683; the author copy (6 text pages) has no printed page numbers -> page 2 of the PDF = p. 138 (IDF scaling), page 4 = p. 140 (sparse document vectors).
- Metsis et al. (2006): the authors' own citation lists no page numbers (CEAS 2006 proceedings); page numbers in the report are PDF pages: p. 1 (popularity of Naive Bayes), p. 2 (Bayes' theorem, threshold T = 0.5), p. 4 (construction of Enron-Spam), p. 5 (Table 1, TP/FP definitions).
- SMS copy: the downloaded copy has 5,572 messages (4,825 legitimate, 747 spam); the original UCI collection has 5,574 (4,827 legitimate, 747 spam). Spam share 13.4 % in both.
- All numbers in Chapters 3-7 of the report were compared with results/tables/*.csv on this date; no differences found.
