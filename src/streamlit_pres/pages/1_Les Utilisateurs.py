import streamlit as st
import os
# from src.config.run_config import init_paths
# assets_path = os.path.join(
#    init_paths["main_path"], init_paths["streamlit_assets_folder"])


st.title("Les utilisateurs")
st.subheader("Les utilisateurs métier - Internes et Radiologues")
st.write("Disposent d'une application web pour charger les images et obtenir un diagnostic")
st.subheader("Workflow métier")
st.image("data/raw/streamlit_assets/MLOps_Utilisateurs_metier_workflow.png")
st.subheader("Les Administrateurs - Ops Engineer et Datascientists")
st.write("Disposent de plusieurs outils pour administrer et gérer l'application et les modèles")
st.subheader("Workflow Administrateur")
st.image("data/raw/streamlit_assets/MLOps_Administrateurs_workflow.png")
