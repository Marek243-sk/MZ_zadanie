"""
Module for loaders testing
"""

import io
import pytest
from src.loaders import read_pdf, read_txt

# pylint: disable=redefined-outer-name

def test_read_text_returns_string():
    """Testing if read_txt returns string"""
    sample_text = "Sample text."
    fake_file = io.StringIO(sample_text)
    result = read_txt(file=fake_file)

    assert isinstance(result, str)
    assert "Sample text " in result


def test_read_text_removes_latex_commands():
    """Test if read_txt removes latex commands"""
    sample_latex = "Sample \\textbf{latex}"
    fake_file = io.StringIO(sample_latex)
    result = read_txt(file=fake_file)

    assert "textbf" not in result
    assert "latex" in result


@pytest.fixture
def sample_pdf_file():
    """Sample pdf file path"""
    sample_pdf_path = "tests/samples/sample_pdf.pdf"
    with open(sample_pdf_path, "rb") as f:
        yield f


def test_read_pdf_returns_string(sample_pdf_file):
    """Testing if read_pdf returns string"""
    result = read_pdf(file=sample_pdf_file)
    assert isinstance(result, str)
    assert len(result) > 0


def test_read_pdf_content(sample_pdf_file):
    """Testing if read_pdf returns correct content"""
    result = read_pdf(file=sample_pdf_file)
    assert "Sample text" in result
