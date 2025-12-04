"""
Module with functions to visualize model results
"""

from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st


def plot_wordcloud(keywords: pd.DataFrame, width: int = 800, height: int = 400) -> None:
    """
    Function that creates wordclou from extracted keywords
    Args:
        keywords(pd.DataFrame): dataframe with extracted kw, theirs score and origin
        width(int): width of the plot
        height(int): height of the plot
    Returns:
        None
    """
    wc = WordCloud(width=width, height=height, background_color="white")
    wc.generate(" ".join(keywords["Keyword"]))
    st.image(wc.to_array())


def plot_barh_chart(
    keywords: pd.DataFrame,
    figsize: Tuple[int, int] = (6, 4),
    ascending: bool = True,
) -> None:
    """
    Function that creates barh chart
    Args:
        keywords(pd.DataFrame): dataframe with extracted kw, theirs score and origin
        figsize(Tuple[int, int]): size of the chart
        ascending(bool): scores ordering
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)
    keywords_sorted = keywords.sort_values("Score", ascending=ascending)
    keywords_sorted.plot.barh(x="Keyword", y="Score", ax=ax)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlabel("Score")
    ax.set_ylabel("Keywords")
    ax.set_title("Keywords score")
    st.pyplot(fig)
