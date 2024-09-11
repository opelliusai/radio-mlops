import streamlit as st
from src.utils import utils_streamlit
from src.streamlit.st_pages import p_predict, p_user_predictions, p_contribution, p_services, p_performance, p_evaluation, p_training, tests, p_models

# Titre du menu latéral
st.sidebar.title("Analyse de radiographies pulmonaires")


def main():
    # Initialisation des variables de session
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        st.title("Page de Connexion")
        # Formulaire de connexion
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            access_token, username, user_name, user_role = utils_streamlit.login(
                username, password)
            if access_token:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_name = user_name
                st.session_state.access_token = access_token
                st.session_state.user_role = user_role
                # st.success(f"Bienvenue {user_name} (Profil {user_role})")
                st.rerun()  # raffraichissement de la page
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect")
    else:
        app_pages = get_menu_list(st.session_state.user_role)
        # Affichage du menu latéral
        page = st.sidebar.radio("Navigation", app_pages, index=0)

        # Affichage du contenu des pages
        if page == "Mon compte":
            st.title(
                f"Votre compte - Bienvenue {st.session_state.user_name}")
            st.subheader(f"(Profil {st.session_state.user_role})")
            # st.success(
            #    f"Bienvenue {st.session_state.user_name} ! (Profil {st.session_state.user_role})")
            # st.write(f"Identifiant : {st.session_state.username}")
            if st.button("Se déconnecter"):
                utils_streamlit.logout()
                st.rerun()  # raffraichissement de la page

            if st.button("Modifier mot de passe"):
                st.text_input("Ancien mot de passe", type="password")
                st.text_input("Nouveau mot de passe", type="password")
                st.text_input("Resaisir le nouveau mot de passe",
                              type="password")
                if st.button("Valider"):
                    # st.success("Mot de passe modifié avec succès!")
                    # st.success("To be implemented!")
                    utils_streamlit.logout()
                    st.rerun()  # raffraichissement de la page
            if st.button("Supprimer votre compte"):
                st.success("To be implemented!")
                utils_streamlit.logout()
                st.rerun()  # raffraichissement de la page
        elif page == "Demande d'analyse":
            # p_predict.main(page)
            p_predict.main(page)
        elif page == "Mes analyses":
            p_user_predictions.main(page)
        elif page == "Contribution":
            p_contribution.main(page)
        elif page == "Etat des services" and st.session_state.user_role == 'admin':
            p_services.main(page)
        elif page == "Performance du modèle de Production" and st.session_state.user_role == 'admin':
            p_performance.main(page)
        elif page == "Evaluation du modèle de Production" and st.session_state.user_role == 'admin':
            p_evaluation.main(page)
        elif page == "Entrainement/Réentrainement" and st.session_state.user_role == 'admin':
            p_training.main(page)
        elif page == "Informations et déploiement de modèles" and st.session_state.user_role == 'admin':
            p_models.main(page)
        elif page == "Tests":
            tests.main(page)


def get_menu_list(user_role):
    if user_role == 'admin':
        # Pages pour l'administrateur
        app_pages = ["Mon compte",
                     "Demande d'analyse",
                     "Mes analyses",
                     "Contribution",
                     "Etat des services",
                     "Performance du modèle de Production",
                     "Evaluation du modèle de Production",
                     "Entrainement/Réentrainement",
                     "1Informations et déploiement de modèles",
                     "Informations et déploiement de modèles",
                     "Based de données",
                     "Tests"]
    elif user_role == 'user':
        # Pages pour l'utilisateur
        app_pages = ["Mon compte",
                     "Demande d'analyse",
                     "Mes analyses",
                     "Contribution",
                     "Tests"]

    return app_pages


if __name__ == "__main__":
    main()
