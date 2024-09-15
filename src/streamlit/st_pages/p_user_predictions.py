'''
Créé le 08/09/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Page pour l'historique des demandes de prédiction par un utilisateur
'''

import streamlit as st
import os
import pandas as pd
from src.config.run_config import init_paths, model_info
from src.config.log_config import setup_logging

logger = setup_logging("STREAMLIT_USER")


def main(title, cookies):
    # st.header(title)
    print("Début de la fonction main")
    prediction_path = os.path.join(init_paths["main_path"],
                                   init_paths["PRED_logging_folder"], model_info["PRED_logging_filename"])
    if not os.path.exists(prediction_path):
        st.warning(f"Le fichier {prediction_path} n'existe pas.")
    else:
        username = cookies.get("username")
        st.subheader(f"Historique de vos prédictions - {username}")
        data = pd.read_csv(prediction_path, sep=",")
        # Filtrer les données pour l'utilisateur connecté
        filtered_data = data[data["Username"] == username]
        # Suppression des colonnes inutiles
        data_pred = filtered_data.drop(["UID", "Chemin de l'image"], axis=1)
        # tri par date de prediction descendante
        data_pred = data_pred.sort_values(
            by='Date de prédiction', ascending=False)
        columns = {
            'Nom du modèle': 'Nom du modèle',
            'Prédiction': 'Prediction',
            'Indice de confiance': 'Confiance',
            'Prédiction validée': 'Prédiction validée',
            'Temps de prédiction': 'Temps',
            'Date de prédiction': 'Date'
        }
        data_pred = data_pred.rename(columns=columns)
        data_pred = data_pred.reindex(columns=list(columns.values()))
        data_pred = data_pred.fillna("X")
        st.dataframe(data_pred.reset_index(drop=True), hide_index=True)

        st.subheader(f"Historique de vos contributions - {username}")
