"""
Modeule with utils
"""

from io import BytesIO
from typing import List
import pandas as pd
import streamlit as st


def export_csv(all_results: List):
    """
    Exports results of extraction as csv file
    Args:
        all_results(List): dataframes with kw, scores and their origin
    Returns:
        None
    """
    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
        csv_buffer = BytesIO()
        result_df.to_csv(csv_buffer, index=False, encoding="utf-8")
        st.download_button(
            label="Download results as CSV file",
            data=csv_buffer.getvalue(),
            file_name="Keywords_results.csv",
            mime="text/csv",
        )
