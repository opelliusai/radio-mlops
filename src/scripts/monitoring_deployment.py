'''
Créé le 08/09/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Script monitoring - Déploiement de modèle
-- Déploie le modèle ayant un Tag 
--  

'''
from src.config.log_config import setup_logging
import requests
import pandas as pd
import uuid
from datetime import datetime
import time
# Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths, monitoring_api_info
from src.utils import utils_data
logger = setup_logging("monitoring_deployment",
                       init_paths["monitoring_log_folder"])

# MONITORING API
API_MONITORING_URL = monitoring_api_info["MONITORING_API_URL"]

# URL Deploy ready models
API_URL_DEPLOY_READY_MODEL = API_MONITORING_URL + \
    monitoring_api_info["DEPLOY_READY_MODEL"]
logger.debug(f"DEPLOY_READY_MODEL : {API_URL_DEPLOY_READY_MODEL}")


def deploy_ready_model():
    """
    Fait appel à l'API de monitoring pour déployer le modèle ayant le bon flag
    """
    logger.debug(
        f"----deploy_ready_model()---")
    logger.debug(f"deploy_ready_model_url = {API_URL_DEPLOY_READY_MODEL}")
    start_time = time.time()
    response = requests.post(API_URL_DEPLOY_READY_MODEL)
    details = "None"
    if response.status_code == 200:
        status_details = response.json().get("status")
        model_name = response.json().get("model_name")
        model_version = response.json().get("model_version")
        details = f"{status_details} - Modèle {model_name}-{model_version}"
        status = "OK"

    else:
        logger.debug(f"Erreur : {response.status_code} - {response.json()}")
        status = "KO"

    end_time = time.time()
    duree = end_time - start_time
    # Logging history
    logging_exists, logging_path = utils_data.check_logging_exists(
        "monitoring")
    if not logging_exists:
        df_logging, logging_path = utils_data.initialize_logging_file(
            "monitoring")
    # Enregistrement des résultats dans le fichier de logging
    else:
        new_line = {}
        df_logging = pd.read_csv(logging_path)
        new_line["UID"] = uuid.uuid4()
        new_line["Type d'exécution"] = "Monitoring - Déploiement de modèle"
        new_line["Date d'exécution"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        new_line["Temps d'exécution"] = duree
        new_line["Status"] = status
        new_line["Détails"] = details
        new_line_df = pd.DataFrame([new_line])
        df_logging = pd.concat([df_logging, new_line_df], ignore_index=True)
        df_logging.to_csv(logging_path, index=False)

    return status, details


if __name__ == "__main__":
    services = deploy_ready_model()
