import fitz
import re
from unidecode import unidecode

def read_pdf(file, filetype: str = "pdf"):
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
