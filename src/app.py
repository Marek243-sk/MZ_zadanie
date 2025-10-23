import os
import streamlit as st
from loaders import read_pdf, read_txt
from preprocessing import preprocess_text
from extraction import extract_yake, extract_tfidf, extract_keybert
from visualization import plot_bar_chart, plot_wordcloud
from utils import export_csv
from constants import GENERAL, TFIDF, YAKE, KEYBERT

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


method = st.selectbox("Choose Extraction Method:", GENERAL.METHODS)
top_n = st.number_input("Number of Keywords:", min_value=GENERAL.MIN_VAL, max_value=GENERAL.MAX_VAL, value=GENERAL.DEFAULT_VAL)

# Additional method-specific parameters
if method == "TF-IDF":
    max_features = st.number_input(
        label="TF-IDF max features:",
        min_value=TFIDF.MAX_FEATURES_MIN_VALUE,
        max_value=TFIDF.MAX_FEATURES_MAX_VALUE,
        value=TFIDF.MAX_FEATURES_VALUE,
        step=TFIDF.MAX_FEATURES_STEP
        )
    ngram_min = st.number_input(
        label="TF-IDF ngram min:",
        min_value=TFIDF.NGRAM_MIN_MIN_VALUE,
        max_value=TFIDF.NGRAM_MIN_MAX_VALUE,
        value=TFIDF.NGRAM_MIN_VALUE
        )
    ngram_max = st.number_input(
        label="TF-IDF ngram max:",
        min_value=TFIDF.NGRAM_MAX_MIN_VALUE,
        max_value=TFIDF.NGRAM_MAX_MAX_VALUE,
        value=TFIDF.NGRAM_MAX_VALUE
        )
    ngram_range = (ngram_min, ngram_max)

elif method == "YAKE":
    ngram_range = st.number_input(
        label="YAKE n-gram:",
        min_value=YAKE.NGRAM_MIN_VALUE,
        max_value=YAKE.NGRAM_MAX_VALUE,
        value=YAKE.NGRAM_VALUE
        )
    dedup_threshold = st.number_input(
        label="YAKE deduplication threshold:",
        min_value=YAKE.DEDUP_MIN_VALUE,
        max_value=YAKE.DEDUP_MAX_VALUE,
        value=YAKE.DEDUP_VALUE,
        step=YAKE.DEDUP_STEP
        )
    window_size = st.number_input(
        label="YAKE window size:",
        min_value=YAKE.WINDOW_MIN_VALUE,
        max_value=YAKE.WINDOW_MAX_VALUE,
        value=YAKE.WINDOW_VALUE
        )

elif method == "KeyBERT":
    keyphrase_min = st.number_input(
        label="KeyBERT ngram min:",
        min_value=KEYBERT.KEY_MIN_MIN_VALUE,
        max_value=KEYBERT.KEY_MIN_MAX_VALUE,
        value=KEYBERT.KEY_MIN_VALUE
        )
    keyphrase_max = st.number_input(
        label="KeyBERT ngram max:",
        min_value=KEYBERT.KEY_MAX_MIN_VALUE,
        max_value=KEYBERT.KEY_MAX_MAX_VALUE,
        value=KEYBERT.KEY_MAX_VALUE
        )
    keyphrase_range = (keyphrase_min, keyphrase_max)
    use_mmr = st.checkbox(
        label="Use Maximal Marginal Relevance (MMR)?",
        value=KEYBERT.USE_MMR
        )
    diversity = st.number_input(
        label="Diversity (if MMR used):",
        min_value=KEYBERT.DIV_MIN_VALUE,
        max_value=KEYBERT.DIV_MAX_VALUE,
        value=KEYBERT.DIV_VALUE,
        step=KEYBERT.DIV_STEP
        )

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
            keywords = extract_yake(
                text=text,
                top_n=top_n,
                n=ngram_range,
                dedup_threshold=dedup_threshold,
                window_size=window_size
                )
        elif method == "TF-IDF":
            keywords = extract_tfidf(
                docs=[text],
                top_n=top_n,
                max_features=max_features,
                ngram_range=ngram_range
                )
        elif method == "KeyBERT":
            keywords = extract_keybert(
                text=text,
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
