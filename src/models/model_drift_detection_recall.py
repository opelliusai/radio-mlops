import numpy as np
import mlflow
import os
import time
from sklearn.metrics import recall_score
from src.datasets.image_preprocessing import preprocess_data, preprocess_unlabeled_data
from src.config.LOCAL_run_config import init_paths, model_info, dataset_info, mlflow_info, drift_info
from mlflow import MlflowClient
from datetime import datetime
from src.config.log_config import setup_logging
from src.utils import utils_models, utils_data

logger = setup_logging("MODELS")
mlflow_uri = mlflow_info["mlflow_tracking_uri"]
client = MlflowClient(
    registry_uri=mlflow_uri, tracking_uri=mlflow_uri)

mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)


def monitor_new_data_predict(model, new_images, dataset_path):
    """
    Surveille les nouvelles données en comparant les caractéristiques extraites 
    des nouvelles images et des données d'entraînement, et calcule les rappels.

    Args:
    - model: modèle complet utilisé pour faire les prédictions.
    - new_images: nouvelles images à surveiller.
    - dataset_path: chemin vers les données d'entraînement.

    Retour:
    - new_features: caractéristiques des nouvelles images.
    - original_features: caractéristiques des images d'entraînement.
    - average_recall: moyenne des rappels entre les données d'entraînement et les nouvelles données (si labels disponibles).
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
    # Extraire les caractéristiques des nouvelles images et des données d'entraînement
    new_features = model.predict(new_data, batch_size=32)
    original_features = model.predict(X_train, batch_size=32)

    logger.debug(f"new_features {new_features}")
    logger.debug(f"original_features {original_features}")
    logger.debug(f"y_train {y_train}")
    logger.debug(f"Conversion de y_train en numerique")
    y_train = utils_data.convert_labels_to_numeric(y_train)
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

    Args:
    - model: modèle TensorFlow complet.
    - new_data: nouvelles images prétraitées.
    - X_train: images d'entraînement prétraitées.
    - y_train: étiquettes d'entraînement.

    Retour:
    - average_recall: la moyenne des rappels.
    """
    # Prédictions sur les nouvelles données
    new_pred_proba = model.predict(new_data, batch_size=32)
    logger.debug(f"new_pred_proba {new_pred_proba}")
    new_pred = np.argmax(new_pred_proba, axis=1)
    logger.debug(f"new_pred {new_pred}")
    # Prédictions sur les données d'entraînement
    train_pred_proba = model.predict(X_train, batch_size=32)
    num_corr = utils_data.generate_numeric_correspondance_np(train_pred_proba)
    logger.debug(f"Numeric correspondance {num_corr}")
    logger.debug(f"train_pred_proba {train_pred_proba}")
    train_pred = np.argmax(train_pred_proba, axis=1)
    logger.debug(f"train_pred {train_pred}")

    # Calculer le rappel pour les données d'entraînement et les nouvelles données
    recall_train = recall_score(y_train, train_pred, average='macro')

    # Hypothèse : on a des labels pour les nouvelles données (sinon gestion nécessaire en amont)
    recall_new = recall_score(
        y_train[:len(new_pred)], new_pred, average='macro')

    # Moyenne des rappels
    average_recall = (recall_train + recall_new) / 2
    return average_recall


def detect_recall_drift(new_recall, original_recall, drift_info):
    """
    Détecte un drift basé sur la différence entre la moyenne des recall des nouvelles données et des données de référence.

    Args:
    - new_recall: rappel moyen sur les nouvelles données.
    - original_recall: rappel moyen sur les données de référence.
    - drift_info: dictionnaire contenant les seuils de détection de drift pour la moyenne et l'écart type.

    Retour:
    - True si un drift est détecté, False sinon.
    """
    # Comparaison des statistiques des rappels
    logger.debug(f"new_recall_mean: {new_recall}")
    logger.debug(f"original_recall_mean: {original_recall}")

    # Calcul de la différence des moyennes de recall
    recall_mean_diff = abs(new_recall - original_recall)
    logger.debug(f"Recall mean difference: {recall_mean_diff}")

    # Détection de dérive basée sur un seuil défini dans drift_info
    mean_threshold = drift_info.get("mean_threshold", 0.05)

    # Vérification si la différence de moyenne dépasse le seuil
    if recall_mean_diff > mean_threshold:
        logger.debug(
            f"Drift détecté sur la moyenne des recalls! Différence: {recall_mean_diff}")
        return True
    else:
        logger.debug(
            "Pas de drift significatif détecté sur la moyenne des recalls.")
        return False


def detect_feature_drift(new_features, original_features, drift_info):
    """
    Détecte un drift en comparant les caractéristiques des nouvelles données et celles des données de référence.

    Args:
    - new_features: caractéristiques des nouvelles images.
    - original_features: caractéristiques des images d'entraînement.
    - drift_info: dictionnaire contenant les seuils de détection de drift pour la distance des caractéristiques.

    Retour:
    - True si un drift est détecté, False sinon.
    """
    # Calcul d'une distance (par exemple, distance euclidienne moyenne) entre les caractéristiques
    distance = np.linalg.norm(new_features - original_features, axis=1).mean()
    logger.debug(f"Distance moyenne entre les caractéristiques : {distance}")

    # Seuil de détection de drift (0.1 par défaut)
    distance_threshold = drift_info.get("distance_threshold", 0.1)

    if distance > distance_threshold:
        logger.debug(
            f"Drift détecté sur les caractéristiques! Distance: {distance}")
        return True
    else:
        logger.debug(
            "Pas de drift significatif détecté sur les caractéristiques.")
        return False


def drift_detection_main(log=True):
    # Récupération des informations sur les bases courantes
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

    drift_info = {
        "mean_threshold": 0.05,  # Seuil de 5% de différence pour le recall
        "distance_threshold": 0.1,  # Seuil de différence pour les caractéristiques
    }

    # Détection de drift basé sur le rappel (si applicable)
    if avg_recall is not None:
        # Rappel moyen pour les données d'entraînement et nouvelles données
        avg_recall_train = avg_recall
        drift_detected = detect_recall_drift(
            avg_recall_train, avg_recall, drift_info)
    else:
        drift_detected = False

    # Détection de drift basé sur les caractéristiques extraites
    feature_drift_detected = detect_feature_drift(
        new_features, original_features, drift_info)

    if drift_detected or feature_drift_detected:
        print("Un drift a été détecté.")
    else:
        print("Pas de drift significatif détecté.")


if __name__ == "__main__":
    drift_detection_main()
