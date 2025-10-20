class GENERAL:
    TOP_N = 10             # Default number of keywords
    LANGUAGE = "en"        # Default language for YAKE
    STOPWORDS = "english"  # Stopwords for all methods


class TFIDF:
    MAX_FEATURES = 5000          # Maximum features for TfidfVectorizer
    NGRAM_RANGE = (1, 2)         # Default n-gram range (min, max)


class YAKE:
    N = 2                        # Maximum words in a keyword
    DEDUP_THRESHOLD = 0.9        # Deduplication threshold
    WINDOW_SIZE = 2               # Context window size
    LANGUAGE = GENERAL.LANGUAGE
    STOPWORDS = GENERAL.STOPWORDS


class KEYBERT:
    NGRAM_RANGE = (1, 2)         # n-gram range for keyphrases
    USE_MMR = False               # Whether to use Maximal Marginal Relevance
    DIVERSITY = 0.5               # Diversity factor if MMR is used
    MODEL_NAME = "all-MiniLM-L6-v2"  # Model name for KeyBERT
    STOPWORDS = GENERAL.STOPWORDS
