"""
Module with constants
"""

LANGUAGE = "en"
STOPWORDS = [
    "a", "the", "and", "but", "or", "so", "to", "of", "this", "that", "them",
    "than", "in", "up", "for", "as", "et", "al", "is", "by", "on", "are", "am"
    "I", "you", "we", "tey", "he", "she", "it", "their", "theirs", "et", "al",
    "use", "uses", "using", "an", "from", "with", "be", "not", "our", "used", "can",
    "could", "would", "will", "were", "was", "yes", "no", "also", "how", "these",
    "at"
    ]
METHODS = ["TF-IDF", "YAKE", "KeyBERT"]
MIN_VAL = 1
MAX_VAL = 100
DEFAULT_VAL = 10
COLUMNS = ["Keyword", "Score"]

# pylint: disable=too-few-public-methods
# pylint: disable=missing-class-docstring
class TFIDF:
    TOP_N = 10


class YAKE:
    N = 2
    DEDUP_THRESHOLD = 0.9
    WINDOW_SIZE = 2


class KEYBERT:
    TOP_N = 10
    DIVERSITY = 0.5
    MODEL_NAME = "all-MiniLM-L6-v2"
