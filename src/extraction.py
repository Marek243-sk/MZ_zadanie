import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
from keybert import KeyBERT
from constants import GENERAL, TFIDF, YAKE, KEYBERT

# Default parameters
TOP_N = 10
LANGUAGE = "en"
STOPWORDS = "english"
YAKE_N = 2
DEDUP_THRESHOLD = 0.9
WINDOW_SIZE = 2

def extract_tfidf(
        docs,
        top_n: int = GENERAL.TOP_N,
        max_features: int = TFIDF.MAX_FEATURES,
        ngram_range: tuple = TFIDF.NGRAM_RANGE
        ) -> pd.DataFrame:

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
    x = vectorizer.fit_transform(docs)
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = x.toarray().sum(axis=0).argsort()[::-1]
    keywords = [(feature_array[i], x.toarray().sum(axis=0)[i]) for i in tfidf_sorting[:top_n]]
    return pd.DataFrame(keywords, columns=["Keyword", "Score"])


def extract_yake(
        text,
        top_n: int = GENERAL.TOP_N,
        n: int = YAKE.N,
        dedup_threshold: float = YAKE.DEDUP_THRESHOLD,
        window_size: int = YAKE.WINDOW_SIZE
        ,language: str = YAKE.LANGUAGE
        ) -> pd.DataFrame:

    kw_extractor = yake.KeywordExtractor(lan=language, n=n, top=top_n,
                                         dedupLim=dedup_threshold, windowsSize=window_size,
                                         stop_words=STOPWORDS)
    keywords = kw_extractor.extract_keywords(text)
    return pd.DataFrame(keywords, columns=["Keyword", "Score"])


def extract_keybert(
        text,
        top_n: int = GENERAL.TOP_N,
        keyphrase_ngram_range: tuple = KEYBERT.NGRAM_RANGE,
        use_mmr: bool = KEYBERT.USE_MMR,
        diversity: float = KEYBERT.DIVERSITY,
        model_name: str = KEYBERT.MODEL_NAME
        )-> pd.DataFrame:

    kw_model = KeyBERT(model_name)
    if use_mmr:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words="english",
            use_mmr=True,
            diversity=diversity,
            top_n=top_n
        )
    else:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words="english",
            top_n=top_n
        )
    return pd.DataFrame(keywords, columns=["Keyword", "Score"])
