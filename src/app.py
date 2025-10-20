import streamlit as st

from loaders import read_pdf, read_txt
from preprocessing import preprocess_text
from extraction import extract_yake, extract_tfidf, extract_keybert
from visualization import plot_bar_chart, plot_wordcloud
from utils import export_csv

st.set_page_config(page_title="Keywords Extraction", layout="centered")
st.title("Keywords Extraction from Documents")

uploaded_files = st.file_uploader(
    "Upload a Document (.txt, .pdf)",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

method = st.selectbox("Choose Extraction Method:", ["TF-IDF", "YAKE", "KeyBERT"])
top_n = st.slider("Number of Keywords:", 5, 20, 10)

if uploaded_files:
    all_results = []

    for file in uploaded_files:
        filename = file.name
        ext = filename.split(".")[-1].lower()

        if ext == "pdf":
            text = read_pdf(file)
        else:
            text = read_txt(file)

        text = preprocess_text(text)
        st.divider()
        st.subheader(f"Document: `{filename}`")

        if method == "YAKE":
            keywords = extract_yake(text, top_n)
        elif method == "TF-IDF":
            keywords = extract_tfidf([text], top_n)
        elif method == "KeyBERT":
            keywords = extract_keybert(text, top_n)

        keywords["Document"] = filename
        all_results.append(keywords)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(keywords)
        with col2:
            plot_bar_chart(keywords)
            plot_wordcloud(keywords)

    st.divider()
    st.subheader("Store your Results")
    export_csv(all_results)