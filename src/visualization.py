import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

def plot_wordcloud(
        keywords,
        width: int = 800,
        height: int = 400,
        bg_col: str = "white"
        ):
    wc = WordCloud(width=width, height=height, background_color=bg_col)
    wc.generate(" ".join(keywords["Keyword"]))
    st.image(wc.to_array())

def plot_bar_chart(
        keywords,
        figsize: tuple = (6, 4),
        ascending: bool = True,
        legend: bool = False
        ):
    fig, ax = plt.subplots(figsize=figsize)
    keywords_sorted = keywords.sort_values("Score", ascending=ascending)
    keywords_sorted.plot.barh(
        x="Keyword", y="Score", ax=ax, legend=legend
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlabel("Score")
    ax.set_ylabel("Keywords")
    ax.set_title("Keywords score")
    st.pyplot(fig)
