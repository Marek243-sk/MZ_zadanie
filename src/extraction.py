import pandas as pd
import numpy as np
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT

TOP_N = 10
LANGUAGE = "sk"
MAX_FEATURES = 5000
STOPWORDS = ["a", "the", "and", "but", "or", "so", "to", "of", "this", "that", "them", "than", "in", "up", "for", "as", "et", "al", "is", "by", "on", "are"]


def extract_yake(text, top_n=TOP_N, language=LANGUAGE) -> pd.DataFrame:
    kw_extractor = yake.KeywordExtractor(lan=language, n=2, top=top_n, stop_words=STOPWORDS)
    keywords = kw_extractor.extract_keywords(text)
    return pd.DataFrame(keywords, columns=["Keyword", "Score"])

def extract_tfidf(texts, top_n=TOP_N, max_features=MAX_FEATURES) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(stop_words=STOPWORDS, max_features=max_features)
    x = vectorizer.fit_transform(texts)
    tfidf_names = np.asarray(x.mean(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()
    scores = sorted(zip(feature_names, tfidf_names), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(scores[:top_n], columns=["Keyword", "Score"])

def extract_keybert(text, top_n: str = TOP_N, model: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    kw_model = KeyBERT(model=model)
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n
    )
    return pd.DataFrame(keywords, columns=["Keyword", "Score"])
