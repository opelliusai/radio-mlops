'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fichier de configuration des logs
Les logs sont gérés par package src/ et initialisés en entête des fichiers python
'''

# set the absolute path
import os
import sys
sys.path.insert(0, os.path.abspath("."))

######
# Informations sur les répertoires, noms de fichiers et conventions de nommage
# On évite de mettre en dur les version des données, modèles etc.
# Ces informations seront stockés dans des fichiers de logging csv
# Pour les répertoires, un script utils/utils_init_folders.py permet de créer tous les répertoires déclarés
######

init_paths = {
    "main_path": ".",
    # tous les chemins sont basés sur main_path données, logs, archives et docs
    "data_folder": "data",
    "archive_folder": "data/archives",
    "logs_folder": "logs",
    "docs": "docs",
    "references": "references",
    "notebooks": "notebooks",
    "pytest_report": "data/pytest_reports",
    # Infra
    "airflow_folder": "airflow",
    "prometheus_folder": "prometheus",
    "grafana_folder": "grafana",
    "scripts_folder": "scripts",
    "docker_folder": "docker-compose",
    "github_actions": ".github/workflows",

    # Chemin des datasets et logging
    "KAGGLE_datasets_folder": "data/raw/kaggle_datasets",
    "REF_datasets_folder": "data/processed/ref_datasets",
    "PROD_datasets_folder": "data/processed/prod_datasets",
    "test_images": "data/raw/test_images",
    "dataset_logging_folder": "data/processed/dataset_logging",

    # Chemin d'entrainement: RUN, Keras tuner
    "mlflow_folder": "data/processed/mlruns",
    "run_folder": "data/processed/mflow_runs",
    "keras_tuner_folder": "data/processed/keras_tuner",

    # Chemin post entrainement: Modeles
    "models_path": "data/models",
    "models_data_path": "data/processed/models_data",
    "PRED_images_folder": "data/raw/prediction_images",
    "PRED_logging_folder": "data/processed/predictions",

    # Post deploiement - Drift
    "model_drift_folder": "data/processed/model_drift",

    # Streamlit - chemin des ressources pour la présentation et certains affichage (récup url)
    "streamlit_assets_folder": "data/raw/streamlit_assets",

    # API - Gestion des utilisateurs / A ne pas inclure dans les répertoires applicatifs, sera stockés dans un répertoire caché
    "streamlit_cache_folder": "data/streamlit_cache",
}

# Base de données utilisée, URL Kaggle qui sera chargée via l'API
# Attention: Il faut récupérer un token d'accès à stocker dans un fichier .kaggle
dataset_info = {
    "KAGGLE_dataset_url": "tawsifurrahman/covid19-radiography-database",
    "KAGGLE_dataset_prefix": "COVID-19_Radiography_Dataset",
    "REF_dataset_prefix": "RadioPulmonaire_REF-",
    "PROD_dataset_prefix": "RadioPulmonaire_PROD-",
    # dataset_logging_folder/KAGGLE_dataset_logging_filename
    "KAGGLE_dataset_logging_filename": "kaggle_dataset_logging.csv",
    # dataset_logging_folder/REF_dataset_logging_filename
    "REF_dataset_logging_filename": "ref_dataset_logging.csv",
    # dataset_logging_folder/PROD_dataset_logging_filename
    "PROD_dataset_logging_filename": "prod_dataset_logging.csv",
}

current_dataset_label_correspondance = {
    'Normal': 0,
    'Viral Pneumonia': 1,
    'COVID': 2
}

model_info = {
    "auteur": "Jihane EL GASMI - Radio MLOps",
    "model_desc": "Projet MLOps - Détection d'anomalie pulmonaire",
    "model_name_prefix": "MLOps_Radio_Model",
    # prediction_logging_folder/PRED_logging_filename
    "PRED_logging_filename": "prediction_logging.csv",
    # prediction_logging_folder/MODEL_DRIFT_logging_filename
    "MODEL_DRIFT_logging_filename": "model_drift_logging.csv",

}

mlflow_info = {
    "mlflow_tracking_uri": "http://127.0.0.1:5001",
    # "mlflow_tracking_uri": "http://mlflow:8090"
}

model_hp = {
    "num_trials": 1,  # init 5
    "max_epochs": 1,      # init 18
    "img_size": 224,
    "img_dim": 3
}


drift_info = {
    "mean_threshold": 0.01,
    "std_threshold": 0.05
}

user_api_info = {
    "USER_API_URL": "http://127.0.0.1:8081",
    # "USER_API_URL": "http://user_api:8081",
    "PREDICT_URL": "/predict",
    "LOG_PREDICTION": "/update_log_prediction",
    "ADD_IMAGE": "/add_image",
    "LOGIN": "/login"
}

admin_api_info = {
    "ADMIN_API_URL": "http://127.0.0.1:8082",
    # "ADMIN_API_URL": "http:/admin_api:8082",
    "DOWNLOAD_DATASET_URL": "/download_dataset",
    "CLEAN_DATASET_URL": "/clean_dataset",
    "UPDATE_DATASET_URL": "/update_dataset",
    "TRAIN_MODEL_URL": "/train_model",
    "RETRAIN_MODEL_URL": "/retrain_model",
    "MAKE_MODEL_PROD_READY_URL": "/make_model_prod_ready",
    "DEPLOY_READY_MODEL": "/deploy_ready_model",
    "FORCE_MODEL_SERVING": "/force_model_serving",
    "GET_LIST_MODELS": "/get_models_list",
    "GET_LIST_DATASETS": "/get_datasets_list",
    "GET_PROD_LIST_DATASETS": "/get_prod_datasets_list",
    "GET_RUNS_INFO": "/get_runs_info",
    "ADD_IMAGES": "/add_images"
}

monitoring_api_info = {
    "URLS_PREFIX": "http://localhost",
    "MONITORING_API_URL": "http://127.0.0.1:8083",
    # "MONITORING_API_URL": "http://monitoring_api:8083",
    "DRIFT_METRICS_URL": "/drift_metrics",
}


user_mgt = {
    "user_filename": "streamlit_users.json"
}
######
# LOG management
######
infolog = {
    "project_name": "Covid19_MLOps",
    "logfile_name": "covid19-mlops.log",
    "logfile_prefix": "radio-mlops_",  # prefix pour le fichier de log par package src
}
