"""
Module for data loaders
"""

import re
import fitz
from unidecode import unidecode


def read_pdf(file, filetype: str = "pdf") -> str:
    """
    Reads pdf file
    Args:
        file: file to extract data from
        filetype(str): type of file - pdf
    Returns:
        text(str): extracted text data from given file
    """
    if isinstance(file, bytes):
        doc = fitz.open(stream=file, filetype=filetype)
    elif isinstance(file, str):
        doc = fitz.open(file)
    else:
        doc = fitz.open(stream=file.read(), filetype=filetype)

    text = ""
    for page in doc:
        text += page.get_text("text")
    doc.close()
    return text


def read_txt(file):
    """
    Reads txt file
    Args:
        file: file to extract data from
    Returns:
        text(str): extracted text data from given file
    """
    if isinstance(file, bytes):
        text = file.decode("utf-8")
    elif isinstance(file, str):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = file.read().decode("utf-8")

    text = re.sub(r"\\(begin|end)\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    text = re.sub(r"[^a-zA-ZáéíóúäčďľňôŕšťžÁÉÍÓÚÄČĎĽŇÔŔŠŤŽ\s]", " ", text)

    return unidecode(text)
