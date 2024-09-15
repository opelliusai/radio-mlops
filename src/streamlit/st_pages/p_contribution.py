'''
Créé le 08/09/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Page pour la contribution des utilisateurs de Profil Radiologue pour la labellisation d'images
-- Contribution
    -- Affichage aléatoire d'une image non labellisée (du répertoire UNLABELED des données de production)
    -- Proposition de classes pour le contributeur
    -- Action: Proposer une classe (labellisation et déplacement de l'image dans le bon répertoire)
    -- ou ne rien proposer: Ne rien faire
    Action en Backend:
    - Mettre à jour le log des predictions
    - Mettre à jour de la BDD locale pour déplacer l'image dans le bon label
    Action Front
    - Afficher la prochaine image
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


def main(title, cookies):
    # Affichage du titre
    st.title("Contribution Détection d'une anomalie Pulmonaire")
    # Configuration de 2 colonnes : Upload d'une prédiction et affichage du résultat
    col_image, col_proposition = st.columns([1, 1])

    status, image_uid, image_name, image_path, pred_id = utils_streamlit.get_unlabeled_image()
    if status == "OK":
        # COLUMN 1
        with col_image:

            st.image(image_path, caption=image_name, use_column_width=True)
            st.write(image_name)
            st.session_state.image_uid = image_uid
            st.session_state.image_name = image_name
            st.session_state.image_upload_path = image_path
            st.session_state.pred_id = pred_id
        # COLUMN 2 :
        with col_proposition:
            col_val, col_inval = st.columns([1, 1])
            with col_inval:
                if st.button("Je ne sais pas", on_click=on_button_click):
                    print("Je ne sais pas button clicked")
                    # ne rien faire (image déjà dans la classe UNLABELED
                    st.success(
                        "Aucune proposition!")
                    st.session_state.clear()
                proposition = st.selectbox(
                    'Proposition', current_dataset_label_correspondance.keys())
                if st.button("Soumettre", on_click=on_button_click_proposition):
                    print("Soumettre button clicked")
                    print(f"Proposition {proposition}")
                    # Fonction qui met à jour la classe et déplace l'image
                    utils_streamlit.update_image_label(
                        st.session_state.image_uid, st.session_state.pred_id, proposition)

                    st.success("La proposition a été validée !")
                    st.session_state.clear()
    else:
        st.warning("Aucune image trouvée")


def on_button_click():
    st.session_state.button_clicked = True

# Fonction pour gérer le clic du bouton


def on_button_click_proposition():
    st.session_state.button_clicked = True
    # st.session_state.input_value =
    # print(f"session_state input value {st.session_state.input_value}")


if __name__ == "__main__":
    main()
