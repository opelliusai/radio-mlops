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
import os
import streamlit as st
# Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths
from src.config.run_config import monitoring_api_info

logger = setup_logging("monitoring_modeldrift",
                       init_paths["monitoring_log_folder"])

# MONITORING API
API_MONITORING_URL = monitoring_api_info["MONITORING_API_URL"]

# URL Deploy ready models
API_URL_DRIFT_METRICS_URL = API_MONITORING_URL + \
    monitoring_api_info["DRIFT_METRICS_URL"]
logger.debug(f"DRIFT_METRICS_URL : {API_URL_DRIFT_METRICS_URL}")


def model_drift_metrics():
    """
    Fait appel à l'API de monitoring pour détecter un model drift
    """
    logger.debug(
        f"----model_drift_metrics()---")
    logger.debug(f"model_drift_metrics = {API_URL_DRIFT_METRICS_URL}")

    response = requests.get(API_URL_DRIFT_METRICS_URL)

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
            # diff_run_id, diff_model_name, diff_model_version = admin_retrain_model(option="diff")
            logger.debug(
                f"Drift détecté - Lancement d'un réentrainement COMBINED")
            # comb_run_id, comb_model_name, comb_model_version = admin_retrain_model(option="combined")
        else:
            logger.debug("Pas de drift détecté - Pas de réentrainement")
        return status, model_name, drift, new_mean, original_mean, new_std, original_std, mean_diff, std_diff, diff_run_id, diff_model_name, diff_model_version, comb_run_id, comb_model_name, comb_model_version
    else:
        print(f"Erreur : {response.status_code} - {response.json()}")
        return None


if __name__ == "__main__":
    services = model_drift_metrics()
