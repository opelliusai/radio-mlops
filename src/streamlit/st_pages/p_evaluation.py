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
from src.utils import utils_streamlit
logger = setup_logging("STREAMLIT_ADMIN")


def main(title):
    st.header(title, cookies)
    drift_filepath = os.path.join(
        init_paths["main_path"], init_paths["model_drift_folder"], model_info["MODEL_DRIFT_logging_filename"])
    st.subheader("Historique des drift")
    if os.path.exists(drift_filepath):
        data_drift = pd.read_csv(drift_filepath, sep=",")
        data_drift = data_drift.drop("UID", axis=1)
        data_drift['Drift'] = data_drift['Drift'].map(
            {True: 'OUI', False: 'NON'})
        st.dataframe(data_drift.reset_index(drop=True), hide_index=True)
    else:
        st.warning(f"Le fichier {drift_filepath} n'existe pas.")

    # Lancer une détection de drift manuellement
    retrain = st.checkbox("Avec réentrainement en cas de drift")

    if st.button("Lancer un calcul de drift"):
        status, model_name, drift, new_mean, original_mean, new_std, original_std, mean_diff, std_diff, status_retrain_diff, diff_run_id, diff_model_version, diff_experiment_link, status_retrain_comb, comb_run_id, comb_model_version, comb_experiment_link = utils_streamlit.lancer_drift_detection(
            retrain)

        if retrain:
            st.write('Infos réentrainement du modèle')
            st.write(f"Statut: {status_retrain_diff}")
            data_train = {
                'Type': ['Run ID', 'Model Name', 'Version', "Lien Experiment"],
                'DIFF': [diff_run_id, model_name, diff_model_version, diff_experiment_link],
                'Combiné': [comb_run_id, model_name, comb_model_version, comb_experiment_link]
            }
            logger.debug(f"data_train {data_train}")
            df_train = pd.DataFrame(data_train)
            st.dataframe(df_train.reset_index(drop=True), hide_index=True)

        st.success("Calcul de Drift exécuté")
        data = {
            'Nom': ['status', 'model_name', 'drift', 'new_mean', 'original_mean', 'new_std', 'original_std', 'mean_diff', 'std_diff'],
            'Valeur': [status, model_name, drift, new_mean, original_mean, new_std, original_std, mean_diff, std_diff]
        }
        df = pd.DataFrame(data)

        st.dataframe(df.reset_index(drop=True), hide_index=True)
