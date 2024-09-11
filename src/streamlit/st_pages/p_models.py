'''
Créé le 14/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Page pour le déploiement d'un modèle
Affichage de la liste de tous les modèles disponibles sur MLFlow
Sélection du modèle à déployer
'''

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.utils import utils_streamlit
from src.config.log_config import setup_logging

logger = setup_logging("STREAMLIT_ADMIN")


def main(title):
    st.header(title)

    st.subheader("Liste des modèles")

    list_models = utils_streamlit.admin_get_models()
    print(f"list_models {list_models}")
    # Tri par date de création
    list_models.sort(key=lambda x: x['Date de création'], reverse=True)
    model_names = [f"{item['Name']}-{item['Version']} ({item['Phase']})" if item['Phase']
                   == 'Production' else f"{item['Name']}-{item['Version']}" for item in list_models]
    # Affichage des modèles dans une select box
    select_model = st.selectbox(
        'Sélection du Modèle(Tri par date de création', model_names)
    # On supprime aussi (Production) qui a été ajouté
    selected_model_name = select_model.replace(" (Production)", "")
    print(f"selected_model_name {selected_model_name}")
    # Resplit pour récupérer le nom et la version du modèle
    selected_model_name, selected_model_version = selected_model_name.split(
        '-')

    print(f"selected_model_name {selected_model_name}")
    print(f"selected_model_version {selected_model_version}")
    # Find the selected model in the list_models
    selected_item = next(
        (item for item in list_models if item['Name'] == selected_model_name and item['Version'] == selected_model_version), None)

    # Display the selected model's content
    if selected_item:

        st.write(
            f"Selected Model: {selected_item['Name']}-{selected_item['Version']}")
        st.write(f"Date de création: {selected_item['Date de création']}")
        st.write(f"Phase: {selected_item['Phase']}")
        # st.write(f"Données d'entrainment {selected_item['Dataset']}")
        st.write(f"Lien {selected_item['Link']}")
        if selected_item['Phase'] != "Production":
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                        label="Préparer au déploiement",
                        key=f"make_prod {selected_item['Version']}"
                ):
                    status = utils_streamlit.admin_make_model_ready(
                        selected_item['Version'])
                    st.write(f"Status {status}")
            with col2:
                if st.button(label="Forcer le déploiement",
                             key=f"force_prod {selected_item['Version']}"):
                    status = utils_streamlit.admin_force_model_serving(
                        selected_item['Version'])
                    st.write(f"Status {status}")
        # Add more fields as needed
        params = selected_item["PARAMS"]
        print(f"params {params}")
        # st.write(f"Dataset {params['Dataset']}")
        # st.write(f"params {params}")
        # Créer un DataFrame
        df = pd.DataFrame(list(params.items()), columns=[
                          'Paramètre', 'Valeur'])
        # On supprime les colonnes qu'on ne veut pas afficher
        df = df[df['Paramètre'] != 'Dataset']
        df = df[df['Paramètre'] != 'Description']

        # Afficher le DataFrame dans Streamlit
        st.dataframe(df.reset_index(
            drop=True), hide_index=True)

    else:
        st.write("Model not found.")
