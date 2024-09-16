import numpy as np
import mlflow
import os
import time
from sklearn.metrics import recall_score
from src.datasets.image_preprocessing import preprocess_data, preprocess_unlabeled_data
from src.config.run_config import init_paths, model_info, dataset_info, mlflow_info, drift_info
from mlflow import MlflowClient
from datetime import datetime
from src.config.log_config import setup_logging
from src.utils import utils_models, utils_data
from src.config.run_config import current_dataset_label_correspondance
logger = setup_logging("MODELS")
mlflow_uri = mlflow_info["mlflow_tracking_uri"]
client = MlflowClient(
    registry_uri=mlflow_uri, tracking_uri=mlflow_uri)

mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)


def monitor_new_data_predict(model, new_images, dataset_path):
    """
    Detection de drift en se basant sur l'average_recall: moyenne des rappels entre les données d'entraînement et les nouvelles données (si labels disponibles).
    """

    # Prétraitement des données
    new_data = preprocess_unlabeled_data(new_images, 224, 3)
    train_data = preprocess_data(dataset_path, size=224, dim=3)
    X_train, y_train = map(list, zip(*train_data))

    # Conversion des données en numpy array
    new_data = np.array(new_data)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    logger.debug("Début de la prédiction")
    # Extraction des caractéristiques des nouvelles images et des données d'entraînement
    new_features = model.predict(new_data, batch_size=32)
    original_features = model.predict(X_train, batch_size=32)

    logger.debug(f"new_features {new_features}")
    logger.debug(f"original_features {original_features}")
    logger.debug(f"y_train {y_train}")
    logger.debug(f"Conversion de y_train en numerique")
    y_train = utils_data.label_to_numeric_np(
        y_train, current_dataset_label_correspondance)
    # Calcul des rappels si les labels sont disponibles
    if y_train is not None and len(y_train) > 0:
        avg_recall = calculate_recall_for_new_and_train(
            model, new_data, X_train, y_train)
    else:
        logger.warning(
            "Les labels pour les nouvelles données ne sont pas disponibles. Le rappel ne sera pas calculé.")
        avg_recall = None

    return new_features, original_features, avg_recall


def calculate_recall_for_new_and_train(model, new_data, X_train, y_train):
    """
    Calcule la moyenne des rappels entre les données de référence et les nouvelles données.
    """
    # Prédictions sur les nouvelles données
    new_pred_proba = model.predict(new_data, batch_size=32)
    logger.debug(f"new_pred_proba {new_pred_proba}")
    new_pred = np.argmax(new_pred_proba, axis=1)
    logger.debug(f"new_pred {new_pred}")
    # Prédictions sur les données d'entraînement
    train_pred_proba = model.predict(X_train, batch_size=32)
    # Conversion en numeriques de np.array
    num_corr = utils_data.label_to_numeric_np(
        train_pred_proba, current_dataset_label_correspondance)
    logger.debug(f"Numeric correspondance {num_corr}")
    logger.debug(f"train_pred_proba {train_pred_proba}")
    train_pred = np.argmax(train_pred_proba, axis=1)
    logger.debug(f"train_pred {train_pred}")

    # Calcul du rappel pour les données d'entraînement et les nouvelles données
    recall_train = recall_score(y_train, train_pred, average='macro')

    recall_new = recall_score(
        y_train[:len(new_pred)], new_pred, average='macro')

    average_recall = (recall_train + recall_new) / 2
    return average_recall


def detect_recall_drift(new_recall, original_recall, drift_info):
    """
    Détecte un drift basé sur la différence entre la moyenne des recall des nouvelles données et des données de référence.
    Retourne le differentiel et un boolean a True si un drift est détecté
    """

    # Comparaison des statistiques des rappels
    logger.debug(f"new_recall_mean: {new_recall}")
    logger.debug(f"original_recall_mean: {original_recall}")

    # Calcul de la différence des moyennes de recall
    recall_mean_diff = abs(new_recall - original_recall)
    logger.debug(f"Recall mean difference: {recall_mean_diff}")

    # Détection de dérive basée sur un seuil défini dans drift_info du fichier de config
    mean_threshold = drift_info["recall_mean_threshold"]
    logger.debug(f"Threshold {mean_threshold}")
    # Vérification si la différence de moyenne dépasse le seuil
    if recall_mean_diff > mean_threshold:
        logger.debug(
            f"Drift détecté sur la moyenne des recalls. Différence: {recall_mean_diff}")
        return recall_mean_diff, True
    else:
        logger.debug(
            "Pas de drift significatif détecté sur la moyenne des recalls.")
        return recall_mean_diff, False


def drift_detection_main(log=True):
    start_time = time.time()
    # Récupération des informations sur les bases courantes de reference et production
    ref_dataset = utils_data.get_latest_dataset_info()
    ref_dataset_path = ref_dataset["Chemin du Dataset"]

    prod_dataset = utils_data.get_latest_dataset_info("PROD")
    prod_dataset_path = prod_dataset["Chemin du Dataset"]

    model, model_name, model_version = utils_models.get_mlflow_prod_model()
    model_infos = f"{model_name}-V{model_version}"
    logger.debug(f"ref dataset_path {ref_dataset_path}")
    logger.debug(f"prod dataset_path {prod_dataset_path}")
    logger.debug(f"model_name {model_infos}")

    new_features, original_features, avg_recall = monitor_new_data_predict(
        model, prod_dataset_path, ref_dataset_path
    )

    # Détection de drift basé sur le rappel (si applicable)
    if avg_recall is not None:
        # Rappel moyen pour les données d'entraînement et nouvelles données
        avg_recall_train = avg_recall
        recall_mean_diff, drift_detected = detect_recall_drift(
            avg_recall_train, avg_recall, drift_info)
    else:
        drift_detected = False

    logger.debug(f"recall drift_detected {drift_detected}")
    # Détection de drift basé sur les caractéristiques extraites

    drift = False
    if drift_detected:
        logger.debug(
            f"Un model drift a été détecté. recall_mean_diff {recall_mean_diff}")
        drift = False
    else:
        drift = False
        logger.debug("Pas de drift significatif détecté.")

    drift_calculation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_time = time.time()
    # Calcul du temps de prédiction
    temps_calcul = round(end_time-start_time, 2)
    if log == True:
        # log MLFlow
        log_mlflow_metrics(recall_mean_diff)
        utils_models.save_drift_metrics_model(
            model_infos, recall_mean_diff, drift, drift_calculation_date, temps_calcul)

    return recall_mean_diff, drift


def log_mlflow_metrics(recall_diff):
    experiment_name = "First Model Tracking"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    logger.info(f"MLFlow - Model Tracking Experiment_ID {experiment_id}")
    # Logging des statistiques dans MLflow
    with mlflow.start_run(experiment_id=experiment_id) as run:
        logger.debug(f"recall_diff {recall_diff}")
        mlflow.log_metric("recall_diff", recall_diff)


if __name__ == "__main__":
    drift_detection_main()
