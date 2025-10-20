import fitz
import re
from unidecode import unidecode

def read_pdf(file, filetype: str = "pdf"):
    text = ""
    with fitz.open(stream=file.read(), filetype=filetype) as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def read_txt(file):
    text = file.read().decode("utf-8")
    text = re.sub(r"\\(begin|end)\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-zA-ZáéíóúäčďľňôŕšťžÁÉÍÓÚÄČĎĽŇÔŔŠŤŽ\s]", " ", text)
    return unidecode(text)

def read_txt(file):
    return file.read().decode("utf-8")
