'''
Créé le 08/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fonctions utiles pour streamlit:
--

'''

# IMPORTS
from src.config.log_config import setup_logging
import requests
import pandas as pd
import os
import streamlit as st
# Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths, user_api_info
from src.config.run_config import admin_api_info, monitoring_api_info

logger = setup_logging("UTILS_STREAMLIT")

# Construction des URL
API_URL = user_api_info["USER_API_URL"]
# URL prédiction
API_URL_PREDICT = API_URL + user_api_info["PREDICT_URL"]
logger.debug(f"API_FULL_URL_PREDICT : {API_URL_PREDICT}")
API_URL_LOG_PREDICTION = API_URL + user_api_info["LOG_PREDICTION"]
logger.debug(f"API_FULL_URL_LOG_PREDICTION : {API_URL_LOG_PREDICTION}")
API_URL_ADD_IMAGE = API_URL + user_api_info["ADD_IMAGE"]
logger.debug(f"API_URL_ADD_IMAGE : {API_URL_ADD_IMAGE}")
API_URL_GET_UNLABELED_IMAGE = API_URL + user_api_info["GET_UNLABELED_IMAGE"]
logger.debug(f"GET_UNLABELED_IMAGE : {API_URL_GET_UNLABELED_IMAGE}")
API_URL_UPDATE_IMAGE = API_URL + user_api_info["UPDATE_IMAGE"]
logger.debug(f"API_URL_UPDATE_IMAGE : {API_URL_UPDATE_IMAGE}")
API_URL_LOGIN = API_URL + user_api_info["LOGIN"]
logger.debug(f"API_FULL_URL_LOGIN : {API_URL_LOGIN}")

# Construction des URL ADMIN
API_ADMIN_URL = admin_api_info["ADMIN_API_URL"]
# URL Téléchargement
API_URL_DOWNLOAD_DATASET = API_ADMIN_URL + \
    admin_api_info["DOWNLOAD_DATASET_URL"]
logger.debug(f"API_FULL_URL_DOWNLOAD_DATASET : {API_URL_DOWNLOAD_DATASET}")
API_URL_CLEAN_DATASET = API_ADMIN_URL + admin_api_info["CLEAN_DATASET_URL"]
logger.debug(f"API_FULL_URL_CLEAN_DATASET : {API_URL_CLEAN_DATASET}")
API_URL_UPDATE_DATASET = API_ADMIN_URL + admin_api_info["UPDATE_DATASET_URL"]
logger.debug(f"API_FULL_URL_UPDATE_DATASET : {API_URL_UPDATE_DATASET}")
API_URL_TRAIN_MODEL = API_ADMIN_URL + admin_api_info["TRAIN_MODEL_URL"]
logger.debug(f"API_FULL_URL_TRAIN_MODEL : {API_URL_TRAIN_MODEL}")
API_URL_MAKE_MODEL_PROD_READY = API_ADMIN_URL + \
    admin_api_info["MAKE_MODEL_PROD_READY_URL"]
logger.debug(
    f"API_FULL_URL_MAKE_MODEL_PROD_READY : {API_URL_MAKE_MODEL_PROD_READY}")
API_URL_DEPLOY_READY_MODEL = API_ADMIN_URL + \
    admin_api_info["DEPLOY_READY_MODEL"]
logger.debug(f"API_FULL_URL_DEPLOY_READY_MODEL : {API_URL_DEPLOY_READY_MODEL}")
API_URL_FORCE_MODEL_SERVING = API_ADMIN_URL + \
    admin_api_info["FORCE_MODEL_SERVING"]
logger.debug(
    f"API_FULL_URL_FORCE_MODEL_SERVING : {API_URL_FORCE_MODEL_SERVING}")
API_URL_GET_LIST_MODELS = API_ADMIN_URL + admin_api_info["GET_LIST_MODELS"]
logger.debug(f"API_FULL_URL_GET_LIST_MODELS : {API_URL_GET_LIST_MODELS}")
API_URL_GET_LIST_DATASETS = API_ADMIN_URL + admin_api_info["GET_LIST_DATASETS"]
logger.debug(f"API_FULL_URL_GET_LIST_DATASETS : {API_URL_GET_LIST_DATASETS}")
API_URL_GET_PROD_LIST_DATASETS = API_ADMIN_URL + \
    admin_api_info["GET_PROD_LIST_DATASETS"]
logger.debug(
    f"API_URL_GET_PROD_LIST_DATASETS : {API_URL_GET_PROD_LIST_DATASETS}")
API_URL_GET_RUNS_INFO = API_ADMIN_URL + admin_api_info["GET_RUNS_INFO"]
logger.debug(f"API_FULL_URL_GET_RUNS_INFO : {API_URL_GET_RUNS_INFO}")
API_URL_ADD_IMAGES = API_ADMIN_URL + admin_api_info["ADD_IMAGES"]
logger.debug(f"API_URL_ADD_IMAGES : {API_URL_ADD_IMAGES}")


# MONITORING API
API_MONITORING_URL = monitoring_api_info["MONITORING_API_URL"]
# URL Téléchargement
API_URL_DRIFT_METRICS = API_MONITORING_URL + \
    monitoring_api_info["DRIFT_METRICS_URL"]
logger.debug(f"API_URL_DRIFT_METRICS : {API_URL_DRIFT_METRICS}")

# Utils USER


def lancer_une_prediction(file, filename, username):
    logger.debug(
        f"-----------lancer_une_prediction(filename = {filename}, username = {username})----------")
    logger.debug(f"prediction_url = {API_URL_PREDICT}")
    files = {"image": (filename, file, "image/jpeg")}
    response = requests.post(API_URL_PREDICT, files=files, params={
                             "username": username})

    if response.status_code == 200:
        model = response.json().get("model_name")
        prediction = response.json().get('prediction')
        confiance = response.json().get('confiance')
        temps_pred = response.json().get('temps_prediction')
        image_path = response.json().get('image_upload_path')
        pred_id = response.json().get('pred_id')
        logger.debug(f"model_name = {model}")
        logger.debug(f"prediction = {prediction}")
        logger.debug(f"confiance = {confiance}")
        logger.debug(f"temps_prediction = {temps_pred}")
        logger.debug(f"image_upload_path = {image_path}")
        logger.debug(f"pred_id = {pred_id}")

        return model, prediction, confiance, temps_pred, image_path, pred_id
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


def ajout_image(image_path, pred_id, label):
    logger.debug(
        f"----ajout_image(image_path={image_path},pred_id={pred_id}label={label})---")
    logger.debug(f"add_image_url = {API_URL_ADD_IMAGE}")
    data = {
        "image_path": image_path,
        "pred_id": pred_id,
        "label": label
    }

    logger.debug(f"data {data}")
    response = requests.post(API_URL_ADD_IMAGE, params=data)

    if response.status_code == 200:
        status = response.json().get("status")
        return status
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


def get_unlabeled_image():
    logger.debug(
        f"----get_unlabeled_image()---")
    logger.debug(f"get_unlabeled_image_url = {API_URL_GET_UNLABELED_IMAGE}")

    response = requests.get(API_URL_GET_UNLABELED_IMAGE)

    if response.status_code == 200:
        status = response.json().get("status")
        image_uid = response.json().get("image_uid")
        image_name = response.json().get("image_name")
        image_path = response.json().get("image_path")
        pred_id = response.json().get("pred_id")
        return status, image_uid, image_name, image_path, pred_id
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


def update_image_label(image_uid, pred_id, label):
    logger.debug(
        f"----ajout_image_dataset(image_uid={image_uid},pred_id={pred_id},label={label})---")
    logger.debug(f"update_image_url = {API_URL_UPDATE_IMAGE}")
    data = {
        "image_uid": image_uid,
        "pred_id": pred_id,
        "label": label
    }

    logger.debug(f"data {data}")
    response = requests.post(API_URL_UPDATE_IMAGE, params=data)

    if response.status_code == 200:
        status = response.json().get("status")
        return status
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


# Utils ADMIN

def admin_get_datasets():
    logger.debug("----------------admin_get_datasets()------------")
    # Essayer de se connecter pour obtenir le token
    logger.debug(f"get_list_datasets = {API_URL_GET_LIST_DATASETS}")

    response = requests.post(API_URL_GET_LIST_DATASETS)

    if response.status_code == 200:
        list_datasets = response.json().get('list_datasets')
        return list_datasets
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


def admin_get_prod_datasets():
    logger.debug("----------------admin_get_prod_datasets()------------")
    # Connexion pour obtenir le token
    logger.debug(f"admin_get_prod_datasets = {API_URL_GET_LIST_DATASETS}")
    data = {
        "type": "PROD"
    }
    logger.debug(f"data {data}")
    response = requests.post(API_URL_GET_LIST_DATASETS, params=data)

    if response.status_code == 200:
        list_datasets = response.json().get('list_datasets')
        return list_datasets
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


def admin_train_model(dataset_version, max_epochs, num_trials, retrain=False, include_prod_data=False):

    logger.debug(
        "--admin_train_model(dataset_version,max_epochs,num_trials)--")

    logger.debug(
        f"--admin_train_model({dataset_version},{max_epochs},{num_trials})-")
    # Connexion pour obtenir un token
    logger.debug(f"train_url = {API_URL_TRAIN_MODEL}")
    data = {
        "dataset_version": dataset_version,
        "max_epochs": max_epochs,
        "num_trials": num_trials,
        "retrain": retrain,
        "include_prod_data": include_prod_data
    }
    logger.debug(f"data {data}")
    response = requests.post(API_URL_TRAIN_MODEL, params=data)

    if response.status_code == 200:
        run_id = response.json().get("run_id")
        model_name = response.json().get("model_name")
        model_version = response.json().get("model_version")
        experiment_link = response.json().get("experiment_link")
        return run_id, model_name, model_version, experiment_link
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


def admin_get_models():
    logger.debug(f"----------------admin_get_models(model_name=)----------")
    # Essayer de se connecter pour obtenir le token
    logger.debug(f"get_models_list = {API_URL_GET_LIST_MODELS}")

    response = requests.get(API_URL_GET_LIST_MODELS)

    if response.status_code == 200:
        list_models = response.json().get('list_models')
        return list_models
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


def admin_make_model_ready(num_version):
    url = API_URL_MAKE_MODEL_PROD_READY
    logger.debug(f"url = {url}")
    data = {
        "num_version": num_version
    }
    logger.debug(f"data {data}")
    response = requests.post(API_URL_MAKE_MODEL_PROD_READY, params=data)

    if response.status_code == 200:
        status = response.json().get("status")
        return status
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


def admin_force_model_serving(num_version):
    url = API_URL_FORCE_MODEL_SERVING
    logger.debug(f"url = {url}")
    data = {
        "num_version": num_version
    }
    logger.debug(f"data {data}")
    response = requests.post(API_URL_FORCE_MODEL_SERVING, params=data)

    if response.status_code == 200:
        status = response.json().get("status")
        return status
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


def admin_log_prediction(pred_id, label):
    logger.debug(
        f"-------admin_log_prediction(pred_id={pred_id},label={label})----")
    url = API_URL_LOG_PREDICTION
    logger.debug(f"url = {url}")
    data = {
        "pred_id": pred_id,
        "label": label
    }
    logger.debug(f"data {data}")
    response = requests.post(API_URL_LOG_PREDICTION, params=data)

    if response.status_code == 200:
        status = response.json().get("status")
        return status
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


def admin_ajout_images(images_labels):
    logger.debug(
        f"--------ajout_image_dataset(image_labels={images_labels})-------")
    logger.debug(f"add_image_url = {API_URL_ADD_IMAGES}")
    data = {
        "images_labels": images_labels
    }
    logger.debug(f"data {data}")
    response = requests.post(API_URL_ADD_IMAGES, params=data)

    if response.status_code == 200:
        status = response.json().get("status")
        return status
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


# Monitoring


def lancer_drift_detection(retrain=False):
    logger.debug(f"----------------lancer_drift_detection()------------")

    logger.debug(f"drift_metrics_launch_url = {API_URL_DRIFT_METRICS}")
    data = {
        "retrain": retrain
    }
    response = requests.post(API_URL_DRIFT_METRICS, params=data)

    if response.status_code == 200:
        status = response.json().get("status")
        model_name = response.json().get("model_name")
        drift = response.json().get('drift')
        new_mean = response.json().get('new_mean')
        original_mean = response.json().get('original_mean')
        new_std = response.json().get('new_std')
        original_std = response.json().get('original_std')
        mean_diff = response.json().get('mean_diff')
        std_diff = response.json().get('std_diff')
        status_retrain_diff = response.json().get('status_retrain_diff')
        diff_run_id = response.json().get('diff_run_id')
        diff_model_version = response.json().get('diff_model_version')
        diff_experiment_link = response.json().get('diff_experiment_link')
        status_retrain_comb = response.json().get('status_retrain_comb')
        comb_run_id = response.json().get('comb_run_id')
        comb_model_version = response.json().get('comb_model_version')
        comb_experiment_link = response.json().get('comb_experiment_link')

        logger.debug(f"status = {status}")
        logger.debug(f"model_name = {model_name}")
        logger.debug(f"drift = {drift}")
        logger.debug(f"new_mean = {new_mean}")
        logger.debug(f"original_mean = {original_mean}")
        logger.debug(f"new_std = {new_std}")
        logger.debug(f"original_std = {original_std}")
        logger.debug(f"mean_diff = {mean_diff}")
        logger.debug(f"std_diff = {std_diff}")
        logger.debug(f"status_retrain_diff = {status_retrain_diff}")
        logger.debug(f"diff_run_id = {diff_run_id}")
        logger.debug(f"diff_model_version = {diff_model_version}")
        logger.debug(f"diff_experiment_link = {diff_experiment_link}")
        logger.debug(f"status_retrain_comb = {status_retrain_comb}")
        logger.debug(f"comb_run_id = {comb_run_id}")
        logger.debug(f"comb_model_version = {comb_model_version}")
        logger.debug(f"comb_experiment_link = {comb_experiment_link}")
        return status, model_name, drift, new_mean, original_mean, new_std, original_std, mean_diff, std_diff, status_retrain_diff, diff_run_id, diff_model_version, diff_experiment_link, status_retrain_comb, comb_run_id, comb_model_version, comb_experiment_link

    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


def lancer_drift_detection_avec_reentrainement():
    logger.debug(f"----------------lancer_drift_detection()------------")

    logger.debug(f"drift_metrics_launch_url = {API_URL_DRIFT_METRICS}")
    response = requests.get(API_URL_DRIFT_METRICS)

    # logger.debug(f"train_url = {API_URL_TRAIN_MODEL}")
    # response = requests.get(API_URL_TRAIN_MODEL)

    if response.status_code == 200:
        status = response.json().get("status")
        model_name = response.json().get("model_name")
        drift = response.json().get('drift')
        new_mean = response.json().get('new_mean')
        original_mean = response.json().get('original_mean')
        new_std = response.json().get('new_std')
        original_std = response.json().get('original_std')
        mean_diff = response.json().get('mean_diff')
        std_diff = response.json().get('std_diff')
        logger.debug(f"status = {status}")
        logger.debug(f"model_name = {model_name}")
        logger.debug(f"drift = {drift}")
        logger.debug(f"new_mean = {new_mean}")
        logger.debug(f"original_mean = {original_mean}")
        logger.debug(f"new_std = {new_std}")
        logger.debug(f"original_std = {original_std}")
        logger.debug(f"mean_diff = {mean_diff}")
        logger.debug(f"std_diff = {std_diff}")
        diff_run_id, diff_model_name, diff_model_version, comb_run_id, comb_model_name, comb_model_version = "na", "na", "na", "na", "na", "na"
        # Lancer un réentrainement si drift est true
        if drift == True:
            logger.debug(f"Drift détecté - Lancement d'un réentrainement DIFF")
            diff_run_id, diff_model_name, diff_model_version = admin_retrain_model(
                option="diff")
            logger.debug(
                f"Drift détecté - Lancement d'un réentrainement COMBINED")
            comb_run_id, comb_model_name, comb_model_version = admin_retrain_model(
                option="combined")
        else:
            logger.debug("Pas de drift détecté - Pas de réentrainement")
        return status, model_name, drift, new_mean, original_mean, new_std, original_std, mean_diff, std_diff, diff_run_id, diff_model_name, diff_model_version, comb_run_id, comb_model_name, comb_model_version
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


# Health Check
def health_check_apps():
    df_urls = pd.read_csv(os.path.join(
        init_paths["main_path"],
        init_paths["streamlit_assets_folder"],
        "url_info.csv"))

    df_urls['Status'] = ''

    for index, row in df_urls.iterrows():
        url = monitoring_api_info["URLS_PREFIX"] + ":" + str(row['Port'])
        logger.debug(f"URL Service {row['Service']} / {url}")

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                df_urls.at[index, 'Status'] = 'UP'
            else:
                df_urls.at[index, 'Status'] = 'ERROR'
        except requests.exceptions.Timeout:
            df_urls.at[index, 'Status'] = 'Timeout'
        except requests.exceptions.ConnectionError:
            df_urls.at[index, 'Status'] = 'DOWN'
    df_urls = df_urls.reset_index(drop=True)
    df_urls['Port'] = df_urls['Port'].astype(str)
    return df_urls

# Fonction de connexion


def login(username, password):
    # Essayer de se connecter pour obtenir le token
    login_url = API_URL_LOGIN
    auth_data = {
        'username': username,  # Remplacer par votre nom d'utilisateur
        'password': password   # Remplacer par votre mot de passe
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.post(login_url, data=auth_data, headers=headers)
    if response.status_code == 200:
        access_token = response.json().get('access_token')
        username = response.json().get('username')
        user_name = response.json().get('user_name')
        user_role = response.json().get('role')

        logger.debug(f"Token d'accès : {access_token}")
        logger.debug(f"Username : {username}")
        logger.debug(f"User_name : {user_name}")
        logger.debug(f"User Role : {user_role}")
        return access_token, username, user_name, user_role
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None

# Fonction pour se déconnecter


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Déconnexion réussie")
    st.rerun()  # raffraichissement de la page
