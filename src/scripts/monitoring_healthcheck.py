'''
Créé le 08/09/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Script monitoring - Healthcheck
-- Vérification du statut des différents services
-- Envoie le contenu du mail pour le DAG si au moins un des services est HS

'''
from src.config.log_config import setup_logging
import requests
import pandas as pd
import uuid
from datetime import datetime
import time
# Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths, urls_info
from src.utils import utils_data
logger = setup_logging("monitoring_healthcheck",
                       init_paths["monitoring_log_folder"])


def global_heathcheck():
    """
    Lance un appel à toutes les url de urls_info et renvoit un status
    """
    logger.debug(
        f"----global_heathcheck()---")
    start_time = time.time()
    services = {}
    message_objet = None
    html_message_corps = None
    status = "OK"
    for service in urls_info.keys():
        url = urls_info[service]
        logger.debug(f"Check service {service} on URL {url}")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                services[service] = "UP"
            else:
                services[service] = "ERROR"
                status = "KO"
        except requests.exceptions.Timeout:
            services[service] = "TIMEOUT"
            status = "KO"
        except requests.exceptions.ConnectionError:
            services[service] = "DOWN"
            status = "KO"
        logger.debug(f"Service {service} - {services[service]}")
    end_time = time.time()
    duree = end_time - start_time
    # Logging history
    logging_exists, logging_path = utils_data.check_logging_exists(
        "monitoring")
    if not logging_exists:
        logger.debug(f"Logging {logging_path} does not exist, initialisation")
        df_logging, logging_path = utils_data.initialize_logging_file(
            "monitoring")
        logger.debug(f"Logging {logging_path} created - df {df_logging}")
    # Enregistrement des résultats dans le fichier de logging
    else:
        logger.debug(f"Logging {logging_path} exists, adding new line")
        new_line = {}
        df_logging = pd.read_csv(logging_path)
        new_line["UID"] = uuid.uuid4()
        new_line["Type d'exécution"] = "Monitoring - Healthcheck"
        new_line["Date d'exécution"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        new_line["Temps d'exécution"] = duree
        new_line["Status"] = status
        new_line["Détails"] = services
        new_line_df = pd.DataFrame([new_line])
        df_logging = pd.concat([df_logging, new_line_df], ignore_index=True)
        df_logging.to_csv(logging_path, index=False)

    # Préparer le message à envoyer si 1 service est down
    if status == "KO":
        message_objet = "[Radio-MLOps] Monitoring - Healthcheck - KO"
        html_message_corps = f"Le monitoring a détecté un problème sur un des services : <br>"
        for service in services.keys():
            html_message_corps += f"- {service} : {services[service]} <br>"
    logger.debug(f"status - {status}")
    logger.debug(f"message_objet - {message_objet}")
    logger.debug(f"html_message_corps - {html_message_corps}")
    return status, message_objet, html_message_corps
