"""Encoding: turn processed text into numbers (TF-IDF vectors)."""
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(ngram_range=(1, 1), min_df=2, max_features=None) -> TfidfVectorizer:
    """TF-IDF = term frequency x inverse document frequency.
    Words that are frequent in a message but rare in the whole collection get a high weight.
    The vectorizer must be fitted on TRAINING data only (to avoid data leakage)."""
    return TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features,
                           sublinear_tf=True, lowercase=False, token_pattern=r"\S+")
