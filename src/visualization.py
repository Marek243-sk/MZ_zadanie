import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

def plot_wordcloud(keywords, width: int = 800, height: int = 400, bg_col: str = "white"):
    wc = WordCloud(width=width, height=height, background_color=bg_col)
    wc.generate(" ".join(keywords["Keyword"]))
    st.image(wc.to_array())

def plot_bar_chart(keywords, figzise: tuple = (6, 4), ascending: bool = True, legend: bool = True):
    fig, ax = plt.subplots(figsize=figzise)
    keywords.sort_values("Score", ascending=ascending).plot.barh(
        x="Keyword", y="Score", ax=ax, legend=legend
    )
    st.pyplot(fig)
