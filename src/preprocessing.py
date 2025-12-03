"""
Module for data preprocessing
"""

import re


def preprocess_text(text: str):
    """
    Basic text preprocessing steps
    Args:
        text(str): text data
    Returns:
        Preprocessed text
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", " ", text)
    return text.strip()
