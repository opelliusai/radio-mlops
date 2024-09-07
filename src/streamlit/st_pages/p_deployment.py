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

    list_models, runs_info = utils_streamlit.admin_get_models()
    print(f"list_models {list_models}")
    print(f"runs_info {runs_info}")
    df = pd.DataFrame(list_models)
    print(f"df {df}")
    # df_m = df.drop(['Confusion Matrix', 'Classification Report'], axis=1)
    df_r = pd.DataFrame(runs_info)
    print(f"df_r {df_r}")
    merged_df = pd.merge(df, df_r, on='RUN', suffixes=('', ''))

    sorted_df = merged_df.sort_values(
        by=['Phase', 'Date de création'], ascending=[False, False])
    sorted_df_table = merged_df.drop(
        columns=['RUN', 'PARAMS', 'Etat', 'STATUS'])
    st.dataframe(sorted_df_table.reset_index(drop=True), hide_index=True)

    # Tab Version
    # Tab title = Model Name
    # si PROD (ajout entre parenthese PROD)
    # On click :
    # tabs = st.tabs(sorted_df["Name"]+sorted_df["Version"].tolist())
    tabs = st.tabs([f"{name} {version}" for name, version in zip(
        sorted_df["Name"], sorted_df["Version"])])

    for tab, line in zip(tabs, sorted_df.itertuples()):
        with tab:
            st.write(f"{line.Name}-{line.Version} - Phase {line.Phase}")
            if line.Phase != "Production":
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(label="Préparer au déploiement", key=f"make_prod {line.Version}"):
                        status = utils_streamlit.admin_make_model_ready(
                            line.Version)
                        st.write(f"Status {status}")
                with col2:
                    if st.button(label="Forcer le déploiement", key=f"force_prod {line.Version}"):
                        status = utils_streamlit.admin_force_model_serving(
                            line.Version)
                        st.write(f"Status {status}")
            params = line.PARAMS
            print(f"params {params}")
            st.write(f"Description {params['Description']}")
            st.write(f"Dataset {params['Dataset']}")

            col_hp, col_perf = st.columns(2)
            with col_hp:
                st.subheader("Paramètres")
                hparams = {
                    "Parameter": ["dropout_connect_rate", "dropout_rate", "img_dim", "img_size", "l2_lambda", "learning_rate", "max_epochs", "num_dropout_layers", "num_trials", "units"],
                    "Value": ["0.2", "0.1", "3", "224", "0.001", "0.0010972846436569073", "1", "5", "1", "224"]
                }
                df_hparams = pd.DataFrame(hparams)
                st.dataframe(df_hparams.reset_index(
                    drop=True), hide_index=True)
            with col_perf:
                '''
                st.subheader("Performances")
                perf = {
                    "Parameter": ["Accuracy", "F1-score", "Perte", "Rappel"],
                    "Value": [params['Accuracy'], params['F1-score'], params['Loss'], params['Recall']]
                }
                '''

                st.subheader("Matrice de confusion")
                df_confusion_matrix = df["Confusion Matrix"]
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.heatmap(df_confusion_matrix, annot=True,
                            cmap="YlGnBu", fmt='g', ax=ax)
                ax.set_xlabel('Prédiction')
                ax.set_ylabel('Réel')

                st.pyplot(fig)

                st.subheader("Classification report")
                df_classification_report = df["Classification Report"]
                # Display the DataFrame as a table in Streamlit
                st.dataframe(df_classification_report.reset_index(
                    drop=True), hide_index=True)
