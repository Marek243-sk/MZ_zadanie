"""
Module for preprocessing testing
"""

from src.preprocessing import preprocess_text

def test_preprocess_text():
    """
    Testing if preprocess_text eliminates numbers, spaces
    """
    sample_text = "  Sample 4 text. "
    result = preprocess_text(sample_text)

    assert isinstance(result, str)
    assert "  Sample" not in result
    assert "text. " not in result
    assert "4" not in result
