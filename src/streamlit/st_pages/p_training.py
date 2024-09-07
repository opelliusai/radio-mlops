'''
Créé le 14/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Page pour l'entrainement de nouveau modèles
Affiche la liste des modèles disponibles 
et permet de lancer l'entrainement d'un nouveau modèle en sélectionnant un dataset et des hyperparamètres
'''

import streamlit as st
from src.utils import utils_streamlit
from src.config.log_config import setup_logging

logger = setup_logging("STREAMLIT_ADMIN")


def main(title):
    st.header("Entrainement/Réentrainement d'un modèle")

    st.radio("", [
             "Entrainement complet", "Réentrainement (avec les données de production)"])

    list_datasets = utils_streamlit.admin_get_datasets()
    list_prod_datasets = utils_streamlit.admin_get_prod_datasets()

    dataset_names = [item['Dataset Name'] for item in list_datasets]
    dataset_names.sort()
    selected_dataset = st.selectbox('Sélection du Dataset', dataset_names)

    st.write("Hyperparamètres")
    col_epochs, col_trials = st.columns([1, 1])
    with col_epochs:
        max_epochs = st.text_input("Max Epochs", value=1)
    with col_trials:
        num_trials = st.slider(
            "Nombre d'essais / Num Trials", min_value=1, max_value=10)

    if st.button("Lancer l'entrainement"):
        st.write(
            "Entrainement lancé, un message vous sera envoyé à la fin de l'entrainement")
        st.write("Récapitulatif:")
        st.write(f"Selected Dataset: {selected_dataset}")
        st.write(f"Max Epochs: {max_epochs}")
        st.write(f"Number of Trials: {num_trials}")
        # Lancer l'entrainement du modèle
        run_id, model_name, model_version = utils_streamlit.admin_train_model(
            selected_dataset, int(max_epochs), int(num_trials))

        st.write("RUN ID: {run_id}")
        st.write("Model name: {model_name}")
        st.write("Model version: {model_version}")
