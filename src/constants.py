class GENERAL:
    TOP_N = 10
    LANGUAGE = "en"
    # STOPWORDS = "english"
    STOPWORDS = [
        "a", "the", "and", "but", "or", "so", "to", "of", "this", "that", "them",
        "than", "in", "up", "for", "as", "et", "al", "is", "by", "on", "are", "am"
        "I", "you", "we", "tey", "he", "she", "it", "their", "theirs", "et", "al",
        "use", "uses", "using", "an", "from", "with", "be", "not", "our", "used", "can",
        "could", "would", "will", "were", "was"
        ]
    METHODS = ["TF-IDF", "YAKE", "KeyBERT"]
    MIN_VAL = 1
    MAX_VAL = 100
    DEFAULT_VAL = 10
    COLUMNS = ["Keyword", "Score"]


class TFIDF:
    MAX_FEATURES = 5_000
    NGRAM_RANGE = (1, 2)
    MAX_FEATURES_MIN_VALUE = 100
    MAX_FEATURES_MAX_VALUE = 10_000
    MAX_FEATURES_VALUE = 5_000
    MAX_FEATURES_STEP = 100
    NGRAM_MIN_MIN_VALUE = 1
    NGRAM_MIN_MAX_VALUE = 3
    NGRAM_MIN_VALUE = 1
    NGRAM_MAX_MIN_VALUE = 1
    NGRAM_MAX_MAX_VALUE = 3
    NGRAM_MAX_VALUE = 1
    STOPWORDS = GENERAL.STOPWORDS


class YAKE:
    N = 2
    DEDUP_THRESHOLD = 0.9
    WINDOW_SIZE = 2
    LANGUAGE = GENERAL.LANGUAGE
    STOPWORDS = GENERAL.STOPWORDS
    NGRAM_MIN_VALUE = 1
    NGRAM_MAX_VALUE = 3
    NGRAM_VALUE = 2
    DEDUP_MIN_VALUE = 0.0
    DEDUP_MAX_VALUE = 1.0
    DEDUP_VALUE = 0.9
    DEDUP_STEP = 0.05
    WINDOW_MIN_VALUE = 1
    WINDOW_MAX_VALUE = 5
    WINDOW_VALUE = 2


class KEYBERT:
    NGRAM_RANGE = (1, 2)
    USE_MMR = False
    DIVERSITY = 0.5
    MODEL_NAME = "all-MiniLM-L6-v2"
    STOPWORDS = GENERAL.STOPWORDS
    KEY_MIN_MIN_VALUE = 1
    KEY_MIN_MAX_VALUE = 3
    KEY_MIN_VALUE = 1
    KEY_MAX_MIN_VALUE = 1
    KEY_MAX_MAX_VALUE = 3
    KEY_MAX_VALUE = 2
    USE_MMR = False
    DIV_MIN_VALUE = 0.0
    DIV_MAX_VALUE = 1.0
    DIV_VALUE = 0.5
    DIV_STEP = 0.05
