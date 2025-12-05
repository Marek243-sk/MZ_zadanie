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
    # if file is bytes
    if isinstance(file, bytes):
        text = file.decode("utf-8")
    #if file is string (path)
    elif isinstance(file, str):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
    # file is file-like object
    else:
        text = file.read()
        # decode if it is bytes
        if isinstance(text, bytes):
            text = text.decode("utf-8")

    # remove latex commands
    text = re.sub(r"\\(begin|end)\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\s*", " ", text)


    text = re.sub(r"[^a-zA-ZáéíóúäčďľňôŕšťžÁÉÍÓÚÄČĎĽŇÔŔŠŤŽ\s]", " ", text)

    return unidecode(text)
