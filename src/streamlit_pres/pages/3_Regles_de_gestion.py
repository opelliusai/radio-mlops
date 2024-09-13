import streamlit as st
# from src.config.run_config import init_paths
import os

st.title("Les règles de gestion")
st.header("Gestion des données")
st.subheader("- Labellisation des données de production")
st.write("Les données validées ou corrigées par les utilisateurs/contributeurs sont automatiquement labellisées")
st.write("Les données non validées par les utilisateurs/contributeurs sont automatiquement stockées dans un répertoire 'UNLABELED' pour être soumises à contribution")

st.subheader("- Intégration de nouvelles données")
st.write("- Différentiel basé sur le hash MD5")
st.write("- Utilisation des données pour l'entrainement des modèles")
st.write("Equilibrage sur la classe minimale")
st.write("Information sur les données rééllement utilisées accessible depuis le RUN d'entrainement")

st.header("Monitoring")

st.subheader("- Batchs et alerte")
st.write("- Batch 1: Healthcheck")
st.write("- Batch 2: Déploiement d'un modèle prêt pour la Production")
st.write("- Batch 3: Calcul de drift et réentrainement si drift détecté")

st.subheader("- Drift monitoring - Calcul")
st.write("Suivi de la moyenne et de l'écart type de la métrique RECALL sur les données de référence et les données de production")
st.subheader("- Drift monitoring - Seuil de drift")
st.write("Moyenne: 0.01")
st.write("Ecart type: 0.05")
st.write("Ces valeurs sont définies dans le fichier de configuration")

st.subheader("- Drift monitoring - Détection de drift et réentrainement")
st.write("Entrainement d'un nouveau modèle sur les nouvelles données de références raffraichies avec les données de production")
