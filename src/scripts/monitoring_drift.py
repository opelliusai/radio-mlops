'''
Créé le 08/09/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Script monitoring - Model et Data Drift
-- Détection de model drift basé sur le différentiel de la moyenne du recall entre les données de référence et les données de production
-- Détection du data drift basé sur la différence sur la moyenne ou l'écart type entre les données de référence et les données de production
-- Envoie le contenu du mail pour les DAG associés
'''
from src.config.log_config import setup_logging
import requests

# Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths, monitoring_api_info

logger = setup_logging("monitoring_modeldrift",
                       init_paths["monitoring_log_folder"])

# MONITORING API
API_MONITORING_URL = monitoring_api_info["MONITORING_API_URL"]
# URL Téléchargement
API_URL_DRIFT_METRICS = API_MONITORING_URL + \
    monitoring_api_info["DRIFT_METRICS_URL"]
logger.debug(f"API_URL_DRIFT_METRICS : {API_URL_DRIFT_METRICS}")
API_URL_DRIFT_METRICS_MODEL = API_MONITORING_URL + \
    monitoring_api_info["DRIFT_METRICS_URL_MODEL"]
logger.debug(f"API_URL_DRIFT_METRICS : {API_URL_DRIFT_METRICS_MODEL}")

API_URL_DRIFT_METRICS_DATA = API_MONITORING_URL + \
    monitoring_api_info["DRIFT_METRICS_URL_DATA"]
logger.debug(f"API_URL_DRIFT_METRICS : {API_URL_DRIFT_METRICS_DATA}")


# Fonction de détection de data drift qui appelle l'API monitoring
def data_drift(retrain=True):
    logger.debug(f"----------------data_drift(retrain={retrain})------------")

    logger.debug(f"drift_metrics_launch_url = {API_URL_DRIFT_METRICS_DATA}")
    data = {
        "retrain": retrain
    }
    response = requests.post(API_URL_DRIFT_METRICS_DATA, params=data)
    message_objet = None
    html_message_corps = None

    if response.status_code == 200:
        status = response.json().get("status")
        model_name = response.json().get("model_name")
        drift = response.json().get('drift')
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
        if drift:
            status = "KO"
            message_objet = "[Radio-MLOps] Monitoring - Data Drift détecté"
            html_message_corps = f"Le monitoring a détecté un data drift <br>"
            for content in response.json().keys():
                html_message_corps += f"- {content} : {response.json().get(content)} <br>"
        logger.debug(f"status - {status}")
        logger.debug(f"message_objet - {message_objet}")
        logger.debug(f"html_message_corps - {html_message_corps}")
        return status, message_objet, html_message_corps

    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None

# Fonction de détection de model drift qui appelle l'API monitoring


def model_drift(retrain=True):
    logger.debug(f"----------------model_drift(retrain={retrain})------------")

    logger.debug(
        f"drift_metrics_model_launch_url = {API_URL_DRIFT_METRICS_MODEL}")
    data = {
        "retrain": retrain
    }
    response = requests.post(API_URL_DRIFT_METRICS_MODEL, params=data)

    if response.status_code == 200:
        status = response.json().get("status")
        model_name = response.json().get("model_name")
        drift = response.json().get('drift')
        recall_diff = response.json().get('recall_diff')
        status_retrain_comb = response.json().get('status_retrain_comb')
        comb_run_id = response.json().get('comb_run_id')
        comb_model_version = response.json().get('comb_model_version')
        comb_experiment_link = response.json().get('comb_experiment_link')

        logger.debug(f"status = {status}")
        logger.debug(f"model_name = {model_name}")
        logger.debug(f"drift = {drift}")
        logger.debug(f"recall_diff = {recall_diff}")
        logger.debug(f"status_retrain_comb = {status_retrain_comb}")
        logger.debug(f"comb_run_id = {comb_run_id}")
        logger.debug(f"comb_model_version = {comb_model_version}")
        logger.debug(f"comb_experiment_link = {comb_experiment_link}")
        if drift:
            status = "KO"
            message_objet = "[Radio-MLOps] Monitoring - Model Drift détecté"
            html_message_corps = f"Le monitoring a détecté un Model drift <br>"
            for content in response.json().keys():
                html_message_corps += f"- {content} : {response.json().get(content)} <br>"
        return status, message_objet, html_message_corps
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None
