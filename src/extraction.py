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


# TF-IDF KWs extraction function
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
        pd.DataFrame: DataFrame with extracted keywords, scores and doc name.
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

    # Conwert results to DataFrame
    return (
        pd.DataFrame(keywords, columns=COLUMNS)
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
        .assign(ID=lambda df: df.index + 1)
        .set_index("ID")
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
# YAKE KWs extraction function
def extract_yake(text: str, config: YAKEConfig) -> pd.DataFrame:
    """
    Extract keywords from a text using the YAKE algorithm.

    Args:
        text (str): Input text.
        config (YAKEConfig): Configuration object for YAKE extraction.

    Returns:
        pd.DataFrame: DataFrame with extracted keywords, scores and doc name.
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
        .assign(ID=lambda df: df.index + 1)
        .set_index("ID")
    )


@dataclass
class KEYBERTConfig:
    """Class for KEYBERT configuration"""

    top_n: int = 10
    keyphrase_ngram_range: Tuple[int, int] = (1, 2)
    use_mmr: bool = False
    diversity: float = 0.5
    model_name: str = "all-MiniLM-L6-v2"


# KeyBERT KWs extraction function
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
        pd.DataFrame: DataFrame with extracted keywords, scores and doc name.
    """

    # Initialize chosen model
    kw_model = KeyBERT(config.model_name)

    # Whether to use Maximal Marginal Relevance (MMR) for the selection of keywords/keyphrases
    if config.use_mmr:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=config.keyphrase_ngram_range,
            stop_words="english",
            top_n=config.top_n,
            use_mmr=True,
            diversity=config.diversity,
        )
    else:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=config.keyphrase_ngram_range,
            stop_words="english",
            top_n=config.top_n,
        )

    # Convert results to DataFrame
    return (
        pd.DataFrame(keywords, columns=COLUMNS)
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
        .assign(ID=lambda df: df.index + 1)
        .set_index("ID")
    )
