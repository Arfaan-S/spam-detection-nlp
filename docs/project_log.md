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
