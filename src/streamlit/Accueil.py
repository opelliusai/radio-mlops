import streamlit as st
from src.utils import utils_streamlit
from src.streamlit.st_pages import p_predict, p_user_predictions, p_evaluation_model, p_contribution, p_services, p_performance, p_evaluation, p_training, p_models
from streamlit_cookies_manager import EncryptedCookieManager

# Titre du menu latéral
st.sidebar.title("Analyse de radiographies pulmonaires")

# Créer un gestionnaire de cookies
cookies = EncryptedCookieManager(prefix="radio_mlops", password="radio_mlops")

# Vérifiez que le gestionnaire de cookies est prêt
if not cookies.ready():
    st.stop()

# Fonction pour vérifier si l'utilisateur est déjà connecté


def is_logged_in():
    return cookies.get('logged_in') == 'True'


def login():
    with st.form(key='login_form'):
        username = st.text_input('Nom d’utilisateur')
        password = st.text_input('Mot de passe', type='password')
        submit = st.form_submit_button('Se connecter')

        if submit:
            access_token, username, user_name, user_role = utils_streamlit.login(
                username, password)
            if access_token:
                cookies['logged_in'] = 'True'
                cookies['username'] = username
                cookies['user_name'] = user_name
                cookies['access_token'] = access_token
                cookies['user_role'] = user_role
                cookies.save()  # Sauvegarder les cookies après modification
                st.rerun()  # raffraichissement de la page
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect")


def main():
    if not is_logged_in():
        login()
    else:
        app_pages = get_menu_list(cookies.get('user_role'))
        # Affichage du menu latéral
        page = st.sidebar.radio("Navigation", app_pages, index=0)

        # Affichage du contenu des pages
        if page == "Mon compte":
            st.title(
                f"Votre compte - Bienvenue {cookies.get('user_name')}")
            st.subheader(f"(Profil {cookies.get('user_role')})")
            # st.success(
            #    f"Bienvenue {st.session_state.user_name} ! (Profil {st.session_state.user_role})")
            # st.write(f"Identifiant : {st.session_state.username}")
            if st.button("Se déconnecter"):
                # utils_streamlit.logout()
                logout()
                # st.rerun()  # raffraichissement de la page

            if st.button("Modifier mot de passe"):
                st.text_input("Ancien mot de passe", type="password")
                st.text_input("Nouveau mot de passe", type="password")
                st.text_input("Resaisir le nouveau mot de passe",
                              type="password")
                if st.button("Valider"):
                    # st.success("Mot de passe modifié avec succès!")
                    # st.success("To be implemented!")
                    # utils_streamlit.logout()
                    logout()
                    # st.rerun()  # raffraichissement de la page
            if st.button("Supprimer votre compte"):
                st.success("To be implemented!")
                # utils_streamlit.logout()
                logout()
                # st.rerun()  # raffraichissement de la page
        elif page == "Demande d'analyse":
            # p_predict.main(page)
            p_predict.main(page, cookies)
        elif page == "Mes analyses":
            p_user_predictions.main(page, cookies)
        elif page == "Contribution":
            p_contribution.main(page, cookies)
        elif page == "Etat des services" and cookies.get('user_role') == 'admin':
            p_services.main(page, cookies)
        elif page == "Performance du modèle de Production" and cookies.get('user_role') == 'admin':
            p_performance.main(page, cookies)
        elif page == "Evaluation data drift" and cookies.get('user_role') == 'admin':
            p_evaluation.main(page, cookies)
        elif page == "Evaluation model drift" and cookies.get('user_role') == 'admin':
            p_evaluation_model.main(page, cookies)
        elif page == "Entrainement/Réentrainement" and cookies.get('user_role') == 'admin':
            p_training.main(page, cookies)
        elif page == "Informations et déploiement de modèles" and cookies.get('user_role') == 'admin':
            p_models.main(page, cookies)


def get_menu_list(user_role):
    if user_role == 'admin':
        # Pages pour l'administrateur
        app_pages = ["Mon compte",
                     "Demande d'analyse",
                     "Mes analyses",
                     "Contribution",
                     "Etat des services",
                     "Performance du modèle de Production",
                     "Evaluation data drift",
                     "Evaluation model drift",
                     "Entrainement/Réentrainement",
                     "Informations et déploiement de modèles"]
    elif user_role == 'user':
        # Pages pour l'utilisateur
        app_pages = ["Mon compte",
                     "Demande d'analyse",
                     "Mes analyses",
                     "Contribution"]

    return app_pages


def logout():
    cookies['logged_in'] = 'False'
    cookies.save()
    st.success("Déconnexion réussie!")
    st.rerun()


if __name__ == "__main__":
    main()
