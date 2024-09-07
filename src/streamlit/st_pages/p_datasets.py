'''
Créé le 14/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Page pour les informations sur le dataset et le téléchargement d'un nouveau dataset
'''

import streamlit as st
from src.utils import utils_streamlit
from src.config.log_config import setup_logging

logger = setup_logging("STREAMLIT_ADMIN")


def main(title):
    st.title(title)
    list_datasets = utils_streamlit.admin_get_datasets()
