import os
import streamlit as st
import pandas as pd

from loaders import read_pdf, read_txt
from preprocessing import preprocess_text
from extraction import extract_yake, extract_tfidf, extract_keybert
from visualization import plot_bar_chart, plot_wordcloud
from utils import export_csv

# Streamlit setup
st.set_page_config(page_title="Keywords Extraction", layout="wide")
st.title("Keywords Extraction from Documents")

# File sources
st.sidebar.header("File Selection")
source_option = st.sidebar.radio(
    "Choose input source:",
    ("Upload files", "Use sample files from DATA/"),
)

# Upload or select files
uploaded_files = []
data_path = os.path.join(os.path.dirname(__file__), "..", "DATA")

if source_option == "Upload files":
    uploaded_files = st.file_uploader(
        "Upload your documents (.txt, .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
else:
    if not os.path.exists(data_path):
        st.warning("No DATA directory found. Please create it.")
    else:
        available_files = [
            f for f in os.listdir(data_path) if f.lower().endswith((".txt", ".pdf"))
        ]
        if not available_files:
            st.warning("No sample files found in DATA/.")
        else:
            selected_files = st.multiselect(
                "Select one or more sample files from DATA/:",
                available_files,
            )
            for f in selected_files:
                uploaded_files.append(os.path.join(data_path, f))

# Keyword extraction settings
METHODS = ["TF-IDF", "YAKE", "KeyBERT"]
MIN_VAL = 1
MAX_VAL = 100
DEFAULT_VAL = 10

method = st.selectbox("Choose Extraction Method:", METHODS)
top_n = st.number_input("Number of Keywords:", min_value=MIN_VAL, max_value=MAX_VAL, value=DEFAULT_VAL)

# Additional method-specific parameters
if method == "TF-IDF":
    max_features = st.number_input("TF-IDF max features:", min_value=100, max_value=10000, value=5000, step=100)
    ngram_min = st.number_input("TF-IDF ngram min:", min_value=1, max_value=3, value=1)
    ngram_max = st.number_input("TF-IDF ngram max:", min_value=1, max_value=3, value=2)
    ngram_range = (ngram_min, ngram_max)

elif method == "YAKE":
    ngram_range = st.number_input("YAKE n-gram:", min_value=1, max_value=3, value=2)
    dedup_threshold = st.number_input("YAKE deduplication threshold:", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
    window_size = st.number_input("YAKE window size:", min_value=1, max_value=5, value=2)

elif method == "KeyBERT":
    keyphrase_min = st.number_input("KeyBERT ngram min:", min_value=1, max_value=3, value=1)
    keyphrase_max = st.number_input("KeyBERT ngram max:", min_value=1, max_value=3, value=2)
    keyphrase_range = (keyphrase_min, keyphrase_max)
    use_mmr = st.checkbox("Use Maximal Marginal Relevance (MMR)?", value=False)
    diversity = st.number_input("Diversity (if MMR used):", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# Main process
if uploaded_files:
    all_results = []

    for file in uploaded_files:
        if isinstance(file, str):
            filename = os.path.basename(file)
            ext = filename.split(".")[-1].lower()
            with open(file, "rb") as f:
                file_data = f.read()
        else:
            filename = file.name
            ext = filename.split(".")[-1].lower()
            file_data = file.read()

        # Read document
        if ext == "pdf":
            text = read_pdf(file_data)
        else:
            text = read_txt(file_data)

        text = preprocess_text(text)
        st.divider()
        st.subheader(f"Document: `{filename}`")

        # Extraction with method-specific parameters
        if method == "YAKE":
            keywords = extract_yake(text, top_n=top_n, n=ngram_range, dedup_threshold=dedup_threshold, window_size=window_size)
        elif method == "TF-IDF":
            keywords = extract_tfidf([text], top_n=top_n, max_features=max_features, ngram_range=ngram_range)
        elif method == "KeyBERT":
            keywords = extract_keybert(
                text,
                top_n=top_n,
                keyphrase_ngram_range=keyphrase_range,
                use_mmr=use_mmr,
                diversity=diversity
            )

        keywords["Document"] = filename
        all_results.append(keywords)

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(keywords)
        with col2:
            plot_bar_chart(keywords)
            plot_wordcloud(keywords)

    # Export all results
    st.divider()
    st.subheader("Store your Results")
    export_csv(all_results)
else:
    st.info("Please upload or select at least one document to proceed.")
