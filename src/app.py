"""
Module for the Streamlit app
"""

import os
import streamlit as st

from loaders import read_pdf, read_txt
from preprocessing import preprocess_text
from extraction import (
    extract_tfidf,
    extract_yake,
    extract_keybert,
    TFIDFConfig,
    YAKEConfig,
    KEYBERTConfig,
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

# List with uploaded files
uploaded_files = []

# If user chooses to uplod own file
if source_option == "Upload files":
    uploaded_files = st.file_uploader(
        "Upload documents (.txt, .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
# If user chooses to use preloaded file(s)
else:
    if not os.path.exists(data_path):
        st.warning("DATA/ directory not found.")
    else:
        choices = [
            f for f in os.listdir(data_path) if f.lower().endswith((".txt", ".pdf"))
        ]
        selected = st.multiselect("Select sample files:", choices)
        uploaded_files = [os.path.join(data_path, f) for f in selected]

method = st.selectbox("Choose Extraction Method:", METHODS)
top_n = st.number_input(
    "Number of Keywords:", min_value=MIN_VAL, max_value=MAX_VAL, value=DEFAULT_VAL
)


# KWs extraction method UIs
def tfidf_ui():
    """UI elements for TF-IDF"""
    return {
        "max_features": st.number_input(
            "TF-IDF max features",
            100, 10000, 5000, step=100,
            help="Maximum number of terms considered by TF-IDF. "
                 "Higher values = more accurate but slower."
        ),
        "ngram_range": (
            st.number_input(
                "N-gram min", 1, 3, 1,
                help="Minimum number of words in an n-gram. 1 = unigram."
            ),
            st.number_input(
                "N-gram max", 1, 3, 2,
                help="Maximum number of words in an n-gram. 2 = bigram, 3 = trigram."
            ),
        ),
    }

def yake_ui():
    """UI elements for YAKE"""
    return {
        "ngram_size": st.number_input(
            "YAKE n-gram size", 1, 3, 2,
            help="Length of extracted phrases (number of words). "
                 "1 = single words only."
        ),
        "dedup_threshold": st.number_input(
            "YAKE deduplication threshold", 0.0, 1.0, 0.9, step=0.05,
            help="Similarity threshold for merging near-duplicate phrases. "
                 "Higher values reduce duplicates."
        ),
        "window_size": st.number_input(
            "YAKE window size", 1, 5, 2,
            help="Context window size used to evaluate co-occurrence of terms."
        ),
    }

def keybert_ui():
    """UI elements for KeyBERT"""
    return {
        "ngram_range": (
            st.number_input(
                "KeyBERT n-gram min", 1, 3, 1,
                help="Minimum number of words in keyphrases."
            ),
            st.number_input(
                "KeyBERT n-gram max", 1, 3, 2,
                help="Maximum number of words in keyphrases."
            ),
        ),
        "use_mmr": st.checkbox(
            "Use MMR (Maximal Marginal Relevance)?", value=False,
            help="Enable MMR to increase diversity in extracted keywords."
        ),
        "diversity": st.number_input(
            "MMR diversity", 0.0, 1.0, 0.5, step=0.05,
            help="Controls how diverse the keywords should be. "
                 "Only applied if MMR is enabled."
        ),
    }

# UIs switch based on chosen method
ui_params = (
    tfidf_ui()
    if method == "TF-IDF"
    else yake_ui() if method == "YAKE" else keybert_ui()
)

# Cache and results handling in session state, if non existent - create new one
if "cache" not in st.session_state:
    st.session_state["cache"] = {}
if "all_results" not in st.session_state:
    st.session_state["all_results"] = {}

if not uploaded_files:
    st.info("Please upload or select at least one file to begin.")
    st.stop()

# List to store all KWs estraction results
st.session_state["all_results"] = []

# For each file in the list of uploaded files
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

    # Using read file method based on file type (.pdf or .txt)
    text = read_pdf(filedata) if ext == "pdf" else read_txt(filedata)
    # Prepare data
    text = preprocess_text(text)

    st.divider()
    st.subheader(f"Document: `{filename}`")

    # Creating cache keys based on chosen KWs extraction method
    if method == "TF-IDF":
        cache_key = f"{filename}|TFIDF|top={top_n}|maxf={ui_params['max_features']}|ng={ui_params['ngram_range']}"
    elif method == "YAKE":
        cache_key = f"{filename}|YAKE|top={top_n}|ng={ui_params['ngram_size']}|dedup={ui_params['dedup_threshold']}|win={ui_params['window_size']}"
    else:
        cache_key = f"{filename}|KEYBERT|top={top_n}|ng={ui_params['ngram_range']}|mmr={ui_params['use_mmr']}|div={ui_params['diversity']}"

    # If cache key exist, load cached results -> no need to run KWs extraction again
    if cache_key in st.session_state["cache"]:
        keywords = st.session_state["cache"][cache_key]

    # Otherwise run KWs extraction with chosen method
    else:
        # Methods and UIs switches based on users choice of KWs extraction method
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
            )
            keywords = extract_keybert(text=text, config=config)

        # Name of the document that can be downloaded
        keywords = keywords.copy()
        keywords["Document"] = filename

        # Save cache
        st.session_state["cache"][cache_key] = keywords

    # Adding results (extracted KW, score and doc name)
    st.session_state["all_results"].append(keywords)

    # UI elements
    col1, col2 = st.columns(2)

    # DataFrame with results
    with col1:
        st.dataframe(keywords)
    # barh plot and wordcloud
    with col2:
        plot_barh_chart(keywords)
        plot_wordcloud(keywords)

# Option to save results as a .csv file
st.divider()
st.subheader("Export results")
export_csv(st.session_state["all_results"])
