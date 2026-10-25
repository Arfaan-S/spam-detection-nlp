"""Text pre-processing for spam detection.

Pipeline (each step can be inspected separately):
  1. lowercase
  2. strip HTML tags
  3. replace URLs, e-mail addresses, currency symbols and numbers with placeholder tokens
     (spam often contains links, money and numbers; the placeholder keeps that signal)
  4. keep alphabetic tokens only
  5. remove stop words (NLTK English list)
  6. lemmatize (WordNet) OR stem (Porter)
"""
import re
from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer


def _ensure_nltk() -> None:
    """Make sure the NLTK word lists exist (downloads them on a normal computer)."""
    for res, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)


_ensure_nltk()
from nltk.corpus import stopwords  # noqa: E402  (imported after the resources are available)

STOP_WORDS = frozenset(stopwords.words("english"))
_lemmatizer, _stemmer = WordNetLemmatizer(), PorterStemmer()

HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MAIL_RE = re.compile(r"\S+@\S+\.\S+")
MONEY_RE = re.compile(r"[£$€]")
NUM_RE = re.compile(r"\d+")
TOKEN_RE = re.compile(r"[a-z]{2,}")


@lru_cache(maxsize=None)
def lemmatize(word: str) -> str:
    return _lemmatizer.lemmatize(word)


@lru_cache(maxsize=None)
def stem(word: str) -> str:
    return _stemmer.stem(word)


def clean(text: str, placeholders: bool = True) -> str:
    """Steps 1-3: lowercase, remove HTML, mark URLs / e-mail addresses / money / numbers.
    With placeholders=False only lowercasing and HTML removal are applied (used for the ablation study)."""
    text = text.lower()
    text = HTML_RE.sub(" ", text)
    if not placeholders:
        return text
    text = URL_RE.sub(" urltoken ", text)
    text = MAIL_RE.sub(" mailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    return NUM_RE.sub(" numtoken ", text)


def tokenize(text: str, placeholders: bool = True) -> list:
    """Step 4: alphabetic tokens with at least two letters."""
    return TOKEN_RE.findall(clean(text, placeholders))


def preprocess(text: str, mode: str = "lemma", remove_stop: bool = True, placeholders: bool = True) -> str:
    """Full pipeline. mode = 'lemma' | 'stem' | 'none'. Returns the processed text as one string."""
    tokens = tokenize(text, placeholders)
    if remove_stop:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if mode == "lemma":
        tokens = [lemmatize(t) for t in tokens]
    elif mode == "stem":
        tokens = [stem(t) for t in tokens]
    return " ".join(tokens)
