"""
Module for keyword extraction
"""

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
from keybert import KeyBERT
from constants import STOPWORDS, COLUMNS, LANGUAGE


@dataclass
class TFIDFConfig:
    """Class for TF-IDF cinfiguration"""

    top_n: int = 20
    max_features: int = 10_000
    ngram_range: Tuple[int, int] = (1, 2)


def extract_tfidf(
    docs,
    config: TFIDFConfig,
) -> pd.DataFrame:
    """
    Extract TF-IDF keywords from a collection of documents.

    Args:
        docs (Iterable[str]): Input documents.
        config (TFIDFConfig): Configuration object for TF-IDF extraction.

    Returns:
        pd.DataFrame: DataFrame containing keywords and their TF-IDF scores.
    """

    # Convert a collection of raw documents to a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        stop_words=STOPWORDS,
    )

    # Fit and transform documents into a TF-IDF matrix
    x = vectorizer.fit_transform(docs)

    # Extract feature names
    feature_array = vectorizer.get_feature_names_out()

    # Compute total TF-IDF score for each token
    tfidf_scores = x.sum(axis=0).A1

    # Sort tokens by score (descending)
    tfidf_sorting = tfidf_scores.argsort()[::-1]

    # Select top N keywords
    top_indices = tfidf_sorting[: config.top_n]
    keywords = [(feature_array[i], tfidf_scores[i]) for i in top_indices]

    return (
        pd.DataFrame(keywords, columns=COLUMNS)
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
    )


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
@dataclass
class YAKEConfig:
    """Class for YAKE configuration"""

    top_n: int = 20
    ngram_size: int = 3
    dedup_threshold: float = 0.9
    window_size: int = 1
    language: str = LANGUAGE


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def extract_yake(text: str, config: YAKEConfig) -> pd.DataFrame:
    """
    Extract keywords from a text using the YAKE algorithm.

    Args:
        text (str): Input text.
        config (YAKEConfig): Configuration object for YAKE extraction.

    Returns:
        pd.DataFrame: DataFrame with extracted keywords and scores.
    """

    # Initialize YAKE keyword extractor
    kw_extractor = yake.KeywordExtractor(
        lan=config.language,
        n=config.ngram_size,
        top=config.top_n,
        dedupLim=config.dedup_threshold,
        windowSize=config.window_size,
        stopwords=STOPWORDS,
    )

    # Extract keywords from text
    keywords = kw_extractor.extract_keywords(text)

    # Convert to DataFrame
    return (
        pd.DataFrame(keywords, columns=COLUMNS)
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
    )


@dataclass
class KEYBERTConfig:
    """Class for KEYBERT configuration"""

    top_n: int = 10
    keyphrase_ngram_range: Tuple[int, int] = (1, 2)
    use_mmr: bool = False
    diversity: float = 0.5
    model_name: str = "all-MiniLM-L6-v2"


def extract_keybert(
    text,
    config: KEYBERTConfig,
) -> pd.DataFrame:
    """
    Extract keywords from a text using the KEYBERT algorithm.

    Args:
        text (str): Input text.
        config (KEYBERTConfig): Configuration object for YAKE extraction.

    Returns:
        pd.DataFrame: DataFrame with extracted keywords and scores.
    """

    kw_model = KeyBERT(config.model_name)
    if config.use_mmr:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=config.keyphrase_ngram_range,
            stop_words="english",
            use_mmr=True,
            diversity=config.diversity,
            top_n=config.top_n,
        )
    else:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=config.keyphrase_ngram_range,
            stop_words="english",
            top_n=config.top_n,
        )
    return (
        pd.DataFrame(keywords, columns=COLUMNS)
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
    )
