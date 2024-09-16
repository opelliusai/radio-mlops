import streamlit as st
from src.utils import utils_streamlit


def main(title, cookies):
    st.header(title)
    # status = utils_streamlit.health_check_apps()
    status = utils_streamlit.health_check_apps_run()
    st.dataframe(status.reset_index(drop=True), hide_index=True)
