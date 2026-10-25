# Spam detection with NLP

Project report for the course *DLBAIPNLP01 – Project: NLP* (Task 2: Text Classification for Spam Detection).
Binary classification of messages as **legitimate (ham)** or **spam** on two public datasets
(SMS Spam Collection, Enron-Spam) with Naive Bayes, Linear SVM, Logistic Regression and a small CNN + BiLSTM neural network.

> The final report (PDF) is in the folder `report/`.

## Main results (test sets, spam = positive class)
| Dataset | Naive Bayes | Linear SVM | Logistic Regression | CNN + BiLSTM (3 seeds) |
|---|---|---|---|---|
| SMS (1,010 test messages) – F1 | 0.914 | 0.924 | 0.943 | 0.912 ± 0.024 |
| Enron (5,795 test emails) – F1 | 0.990 | 0.994 | 0.991 | 0.988 ± 0.002 |

Models trained on one dataset fail on the other (F1 0.23–0.50), see `results/tables/cross_dataset.csv`.
All numbers are produced by the scripts below; full tables are in `results/tables/`, figures in `results/figures/`.

## Project structure
| Folder / file | Content |
|---|---|
| `src/download_data.py` | downloads the two datasets into `data/raw/` |
| `src/data_loader.py` | loads both datasets into one common format |
| `src/explore_data.py` | dataset statistics and figures |
| `src/preprocessing.py` | text cleaning, stop-word removal, lemmatizing / stemming |
| `src/features.py` | TF-IDF encoding |
| `src/prepare_data.py` | duplicate removal, stratified train/test split |
| `src/evaluation.py` | metrics, bootstrap confidence interval |
| `src/step5_sms_models.py` | SMS: pre-processing study, model selection by cross-validation, test evaluation, error analysis |
| `src/step6_scaling_generalization.py` | Enron models, learning curves, cross-dataset test, bias check |
| `src/step6b_neural_network.py` | CNN + BiLSTM neural network (PyTorch) |
| `src/step7_predict_demo.py` | classifies new messages with the saved models |
| `models/` | saved models (see below) |
| `results/` | tables and figures produced by the scripts |
| `docs/project_log.md` | log of decisions and findings |

## How to run
```bash
pip install -r requirements.txt
python src/download_data.py             # downloads the data into data/raw/
python src/explore_data.py              # statistics and figures
python src/prepare_data.py              # pre-processing and train/test split -> data/processed/
python src/step5_sms_models.py          # SMS models
python src/step6_scaling_generalization.py   # Enron models, learning curves, cross-dataset test (~7 min)
python src/step6b_neural_network.py     # neural network (~30 min on CPU)
python src/step7_predict_demo.py "Congratulations, you won a free prize! Click here"   # classify new text
```
The NLTK word lists (stop words, WordNet) are downloaded automatically on the first run.

## Text variants
The classical models use stemmed text (chosen by cross-validation in step 5); the neural network uses lemmatized text (design decision, not tested separately).

## Data and saved models
* `data/raw/` and `data/processed/` are **not stored** in the repository (public datasets with their own licences;
  `processed` is about 60 MB). Run `download_data.py` and `prepare_data.py` to recreate them.
* `models/` contains the trained models, so that `step7_predict_demo.py` runs without retraining:
  `spam_model_sms.joblib` (Logistic Regression), `spam_model_enron.joblib` (Linear SVM) – scikit-learn pipelines –
  and `spam_nn_sms.pt`, `spam_nn_enron.pt` (PyTorch weights, vocabulary included).
  The `.joblib` files require the scikit-learn version from `requirements.txt` (or a close one).

## Data sources
- Almeida & Hidalgo (2011). *SMS Spam Collection* [Data set]. UCI Machine Learning Repository, https://doi.org/10.24432/C5CC84 (CC BY 4.0).
  The copy used here (`sms_spam.csv`, downloaded from the GitHub mirror `mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-`) has 5,572 messages (747 spam); the original collection has 5,574.
- Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006). Spam filtering with Naive Bayes – Which Naive Bayes? *Proceedings of the 3rd Conference on Email and Anti-Spam (CEAS 2006)* (Enron-Spam).
  The copy used here (`enron_spam_data.zip`, GitHub repository `MWiechmann/enron_spam_data`) has 33,716 e-mails; this equals the sum of the six datasets in Table 1 of the paper.
