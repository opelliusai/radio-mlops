'''
Créé le 14/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Page pour le monitoring des prédictions et des performances
'''

import streamlit as st
import os
import pandas as pd
from src.config.run_config import init_paths, model_info
from src.config.log_config import setup_logging

logger = setup_logging("STREAMLIT_ADMIN")


def main(title, cookies):
    st.header(title)
    prediction_path = os.path.join(init_paths["main_path"],
                                   init_paths["PRED_logging_folder"], model_info["PRED_logging_filename"])
    if not os.path.exists(prediction_path):
        st.warning(f"Le fichier {prediction_path} n'existe pas.")
    else:
        st.subheader("Historique des prédictions")
        data = pd.read_csv(prediction_path, sep=",")
        # Suppression des colonnes inutiles
        data_pred = data.drop(["UID", "Chemin de l'image"], axis=1)
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

        st.subheader("Performance - Temps de prédiction")
        temps_pred = data[["Nom du modèle",
                           "Temps de prédiction", "Date de prédiction"]]
        model_names = temps_pred["Nom du modèle"].unique()
        selected_model = st.selectbox("Selectionner un modèle", model_names)
        filtered_data = temps_pred[temps_pred["Nom du modèle"]
                                   == selected_model]
        st.line_chart(
            filtered_data, x=temps_pred.columns[2], y=temps_pred.columns[1])

        st.subheader("Performance - Précision des prédictions")
        perf_pred = data[["Nom du modèle",
                          "Perf Prédiction", "Date de prédiction"]]
        model_names = temps_pred["Nom du modèle"].unique()
        filtered_data = perf_pred[perf_pred["Nom du modèle"] == selected_model]
        st.line_chart(
            filtered_data, x=perf_pred.columns[2], y=perf_pred.columns[1])
