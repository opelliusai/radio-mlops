'''
Créé le 08/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Page pour la prédiction d'une Anomalie Pulmonaire (COVID ou Pneumonia Virale)
-- Gestion du compte
    -- Inscription (à venir) 
    -- Authentification (Avec Profil simple ou admin)
    -- Modification du mot de passe (à venir)
    -- Suppression du compte (à venir)
-- Prédiction
    -- Historique des prédictions de l'utilisateur (à venir)
    -- Exécution d'une prédiction et visualisation du résultat avec indice de confiance
    -- Action: Valider/Invalider/Modifier la prédiction
    Action en Backend:
    - Mettre à jour le log des predictions
    - Mettre à jour de la BDD locale pour ajouter les bonnes prédictions à une nouvelle version du dataset
'''

# Import des modules
import streamlit as st
from fastapi import UploadFile
from io import BytesIO
# Fin des imports


# Imports des modules internes
from src.utils import utils_streamlit
from src.config.run_config import current_dataset_label_correspondance
from src.config.log_config import setup_logging

logger = setup_logging("STREAMLIT_USER")


def main(title, uploaded_file=None):
    # Affichage du titre
    st.title("Détection d'une anomalie Pulmonaire")
    # Configuration de 2 colonnes : Upload d'une prédiction et affichage du résultat
    col_upload, col_res = st.columns([1, 1])

    # COLUMN 1
    with col_upload:
        if uploaded_file is None:
            uploaded_file = st.file_uploader(
                "Choisissez une image de Radiologie Pulmonaire...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            logger.debug(f"Type de Fichier {type(uploaded_file)}")
            logger.debug(f"filename {uploaded_file.name}")

    # COLUMN 2 :
    with col_res:
        if 'prediction' not in st.session_state:
            if uploaded_file:
                logger.debug("Conversion en UploadFile lisible par FastAPI")
                file_like = BytesIO(uploaded_file.read())
                filename = uploaded_file.name
                model_name, prediction, confiance, temps_prediction, image_upload_path, pred_id = utils_streamlit.lancer_une_prediction(
                    file_like, filename)
                st.session_state.prediction = prediction
                st.session_state.confiance = confiance
                st.session_state.temps_prediction = temps_prediction
                st.session_state.image_upload_path = image_upload_path
                st.session_state.pred_id = pred_id

        if 'prediction' in st.session_state:
            st.image(
                uploaded_file, caption=f"{st.session_state.prediction}, à {st.session_state.confiance}% en {st.session_state.temps_prediction} s")

            col_val, col_inval = st.columns([1, 1])
            with col_val:
                if st.button("Valider", on_click=on_button_click):
                    print("Valider button clicked")
                    status_ajout_image = utils_streamlit.ajout_image(
                        st.session_state.image_upload_path, st.session_state.prediction)
                    status_log_prediction = utils_streamlit.admin_log_prediction(
                        st.session_state.pred_id, st.session_state.prediction)
                    if status_ajout_image == "OK" and status_log_prediction == "OK":
                        st.success("La prédiction a été validée !")
                    else:
                        st.error(
                            f"Erreur lors de la validation de la prédiction. status_ajout_image {status_ajout_image} / status_log_prediction {status_log_prediction} ")
                    st.session_state.clear()
                    uploaded_file = None

            with col_inval:
                if st.button("Je ne sais pas", on_click=on_button_click):
                    print("Je ne sais pas button clicked")
                    utils_streamlit.ajout_image(
                        st.session_state.image_upload_path, "UNLABELED")
                    uploaded_file = None
                    st.success(
                        "La reconnaissance n'a pas été validée sans proposition!")
                    st.session_state.clear()

            proposition = st.selectbox(
                'Proposition', current_dataset_label_correspondance.keys())
            if st.button("Soumettre", on_click=on_button_click_proposition):
                print("Soumettre button clicked")
                print(f"Proposition {proposition}")
                utils_streamlit.ajout_image(
                    st.session_state.image_upload_path, proposition)
                utils_streamlit.admin_log_prediction(
                    st.session_state.pred_id, proposition)
                st.success("La proposition a été validée !")
                st.session_state.clear()


def on_button_click():
    st.session_state.button_clicked = True

# Fonction pour gérer le clic du bouton


def on_button_click_proposition():
    st.session_state.button_clicked = True
    # st.session_state.input_value =
    # print(f"session_state input value {st.session_state.input_value}")


if __name__ == "__main__":
    main()
