"""
Module for the Streamlit app
"""

import os
import pandas as pd
import streamlit as st

from loaders import read_pdf, read_txt
from preprocessing import preprocess_text
from extraction import (
    extract_tfidf,
    extract_yake,
    extract_keybert,
    TFIDFConfig,
    YAKEConfig,
    KEYBERTConfig
)
from visualization import plot_wordcloud, plot_barh_chart
from utils import export_csv
from constants import METHODS, MIN_VAL, MAX_VAL, DEFAULT_VAL


st.set_page_config(page_title="Keyword Extraction", layout="wide")
st.title("Keyword Extraction from Documents")

st.sidebar.header("File Selection")
source_option = st.sidebar.radio(
    "Choose input source:",
    ("Upload files", "Use sample files from DATA/"),
)

data_path = os.path.join(os.path.dirname(__file__), "..", "DATA")

uploaded_files = []
if source_option == "Upload files":
    uploaded_files = st.file_uploader(
        "Upload documents (.txt, .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
else:
    if not os.path.exists(data_path):
        st.warning("DATA/ directory not found.")
    else:
        choices = [
            f for f in os.listdir(data_path)
            if f.lower().endswith((".txt", ".pdf"))
        ]
        selected = st.multiselect("Select sample files:", choices)
        uploaded_files = [
            os.path.join(data_path, f) for f in selected
        ]

method = st.selectbox("Choose Extraction Method:", METHODS)
top_n = st.number_input(
    "Number of Keywords:",
    min_value=MIN_VAL,
    max_value=MAX_VAL,
    value=DEFAULT_VAL
)

def tfidf_ui():
    return {
        "max_features": st.number_input(
            "TF-IDF max features", 100, 10000, 5000, step=100),
        "ngram_range": (
            st.number_input("ngram min", 1, 3, 1),
            st.number_input("ngram max", 1, 3, 2),
        ),
    }


def yake_ui():
    return {
        "ngram_size": st.number_input("YAKE n-gram size", 1, 3, 2),
        "dedup_threshold": st.number_input(
            "YAKE deduplication threshold", 0.0, 1.0, 0.9, step=0.05),
        "window_size": st.number_input("YAKE window size", 1, 5, 2),
    }


def keybert_ui():
    return {
        "ngram_range": (
            st.number_input("KeyBERT n-gram min", 1, 3, 1),
            st.number_input("KeyBERT n-gram max", 1, 3, 2),
        ),
        "use_mmr": st.checkbox("Use MMR (diversity)?", value=False),
        "diversity": st.number_input(
            "MMR Diversity", 0.0, 1.0, 0.5, step=0.05),
    }


ui_params = (
    tfidf_ui() if method == "TF-IDF"
    else yake_ui() if method == "YAKE"
    else keybert_ui()
)

if not uploaded_files:
    st.info("Please upload or select at least one file to begin.")
    st.stop()

all_results = []

for file in uploaded_files:

    if isinstance(file, str):
        filename = os.path.basename(file)
        ext = filename.split(".")[-1].lower()
        with open(file, "rb") as f:
            filedata = f.read()
    else:
        filename = file.name
        ext = filename.split(".")[-1].lower()
        filedata = file.read()

    text = read_pdf(filedata) if ext == "pdf" else read_txt(filedata)
    text = preprocess_text(text)

    st.divider()
    st.subheader(f"Document: `{filename}`")

    if method == "TF-IDF":
        config = TFIDFConfig(
            top_n=top_n,
            max_features=ui_params["max_features"],
            ngram_range=ui_params["ngram_range"],
        )
        keywords = extract_tfidf(docs=[text], config=config)

    elif method == "YAKE":
        config = YAKEConfig(
            top_n=top_n,
            ngram_size=ui_params["ngram_size"],
            dedup_threshold=ui_params["dedup_threshold"],
            window_size=ui_params["window_size"],
        )
        keywords = extract_yake(text=text, config=config)

    else:
        config = KEYBERTConfig(
            top_n=top_n,
            keyphrase_ngram_range=ui_params["ngram_range"],
            use_mmr=ui_params["use_mmr"],
            diversity=ui_params["diversity"],
            # model_name ostáva default
        )
        keywords = extract_keybert(text=text, config=config)

    keywords["Document"] = filename
    all_results.append(keywords)

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(keywords)

    with col2:
        plot_barh_chart(keywords)
        plot_wordcloud(keywords)


st.divider()
st.subheader("Export results")
export_csv(all_results)
