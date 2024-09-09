import streamlit as st
from src.config.run_config import init_paths
import os
st.title("Le produit")

st.write("- Une application web qui permet aux utilisateurs d'obtenir rapidement un diagnostic sur l'anomalie pulmonaire")

st.header("Les utilisateurs métier")
st.write("- Internes/Contributeurs")

assets_path = os.path.join(
    init_paths["main_path"], init_paths["streamlit_assets_folder"])
st.image(os.path.join(assets_path, "UC_metier.png"))

st.write("- Administrateurs")
st.image(os.path.join(assets_path, "UC_administrateur.png"))

st.header("La solution proposée")
st.subheader("L'interface métier")

st.subheader("Le workflow")

st.header("La solution technique")
st.subheader("Workflow")
st.image(os.path.join(assets_path, "Workflow_global.png"))

st.subheader("Architecture")
st.image(os.path.join(assets_path, "Archi_Docker.png"))

st.subheader("Règles de gestion - Données/Contribution")
st.write("- Labellisation des données de production")
st.write("Les données non validées par les utilisateurs/contributeurs sont automatiquement labellisées")
st.write("Les données validées ou corrigées par les utilisateurs/contributeurs sont automatiquement labellisées")

st.write("- Intégration de nouvelles données")
st.write("Basées sur le hash MD5")

st.write("- Utilisation des données pour l'entrainement des modèles")
st.write("Equilibrage sur la classe minimale")
st.write("Information sur les données rééllement utilisées accessible depuis le RUN d'entrainement")

st.subheader("Règles de gestion - Monitoring")
st.write("- Drift monitoring - Calcul")
st.write("Suivi de la moyenne et de l'écart type sur les données de référence et les données de production")
st.write("- Drift monitoring - Seuil de drift")
st.write("Moyenne: 0.01")
st.write("Ecart type: 0.05")
st.write("Ces valeurs sont définies dans le fichier de configuration")

st.write("- Drift monitoring - Détection de drift et réentrainement")
st.write("Ré-entrainement du modèle de Production sur les données de production")
st.write("Entrainement d'un nouveau modèle sur les nouvelles données de références raffraichies avec les données de production")
