# Spam detection with NLP

Project report for the course *DLBAIPNLP01 – Project: NLP* (Task 2: Text Classification for Spam Detection).
Binary classification of messages as **legitimate (ham)** or **spam** with classical machine learning
on two public datasets (SMS Spam Collection, Enron-Spam).

> Work in progress – the final report (PDF) and the performance metrics are added when the project is finished.

## Project structure
| Folder / file | Content |
|---|---|
| `src/download_data.py` | downloads the two datasets into `data/raw/` |
| `src/data_loader.py` | loads both datasets into one common format |
| `src/explore_data.py` | dataset statistics and figures |
| `src/preprocessing.py` | text cleaning, stop-word removal, lemmatizing / stemming |
| `src/features.py` | TF-IDF encoding |
| `src/prepare_data.py` | duplicate removal, stratified train/test split |
| `results/` | tables and figures produced by the scripts |
| `docs/project_log.md` | log of decisions and findings |

## How to run
```bash
pip install -r requirements.txt
python src/download_data.py     # downloads the data (not stored in this repository)
python src/explore_data.py      # step 1: exploration
python src/prepare_data.py      # step 2: pre-processing and train/test split
```
The NLTK word lists (stop words, WordNet) are downloaded automatically on the first run.

## Data sources
- Almeida, T. A., & Hidalgo, J. M. G. (2011). *SMS Spam Collection*. UCI Machine Learning Repository, https://doi.org/10.24432/C5CC84 (CC BY 4.0).
- Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006). Spam filtering with naive Bayes – which naive Bayes? *CEAS 2006* (Enron-Spam).
