import streamlit as st
from src.utils import utils_streamlit


def main(title):
    st.header(title)
    status = utils_streamlit.health_check_apps()

    st.dataframe(status.reset_index(drop=True), hide_index=True)
