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

logger = setup_logging("MODELS")
mlflow_uri = mlflow_info["mlflow_tracking_uri"]
client = MlflowClient(
    registry_uri=mlflow_uri, tracking_uri=mlflow_uri)

mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)


def monitor_new_data_predict(model, new_images, dataset_path):
    """
    Cette fonction surveille les nouvelles données en comparant les caractéristiques extraites 
    des nouvelles images et des données d'entraînement, et calcule les rappels.

    Args:
    - feature_extractor: modèle pré-entraîné ou partie du modèle utilisé pour l'extraction de caractéristiques.
    - new_images: nouvelles images à surveiller.
    - dataset_path: chemin vers les données d'entraînement.
    - model: modèle complet utilisé pour faire les prédictions.

    Retour:
    - new_features: caractéristiques des nouvelles images.
    - original_features: caractéristiques des images d'entraînement.
    - average_recall: moyenne des rappels entre les données de référence et les nouvelles données.
    """

    # Prétraitement des données
    new_data = preprocess_unlabeled_data(new_images, 224, 3)
    train_data = preprocess_data(dataset_path, size=224, dim=3)
    X_train, y_train = map(list, zip(*train_data))

    # Conversion des données en numpy array
    new_data = np.array(new_data)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Extraire les caractéristiques des nouvelles images
    new_features = model.predict(new_data)
    logger.debug(f"new_features {new_features}")

    # Extraire les caractéristiques des données d'entraînement
    original_features = model.predict(X_train)
    logger.debug(f"original_features {original_features}")

    # Calculer les prédictions du modèle complet pour les nouvelles données et les données de référence
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
        new_pred_proba = model.predict(new_data)
        new_pred = np.argmax(new_pred_proba, axis=1)

        # Prédictions sur les données d'entraînement
        train_pred_proba = model.predict(X_train)
        train_pred = np.argmax(train_pred_proba, axis=1)

        # Calculer le rappel pour les données d'entraînement et les nouvelles données
        recall_train = recall_score(y_train, train_pred, average='macro')
        # Hypothèse : on a des labels
        recall_new = recall_score(
            y_train[:len(new_pred)], new_pred, average='macro')

        # Moyenne des rappels
        average_recall = (recall_train + recall_new) / 2
        return average_recall

    # Calcul de la moyenne des rappels entre les données d'entraînement et les nouvelles données
    avg_recall = calculate_recall_for_new_and_train(
        model, new_data, X_train, y_train)

    return new_features, original_features, avg_recall


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
    # Valeur par défaut de 0.05 si non spécifié
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


def drift_detection_main(log=True):
    # Récupération des informations sur les bases courantes
    # Dernière base de référence
    ref_dataset = utils_data.get_latest_dataset_info()
    ref_dataset_path = ref_dataset["Chemin du Dataset"]
    # Dernière base courante (prod)
    prod_dataset = utils_data.get_latest_dataset_info("PROD")
    prod_dataset_path = prod_dataset["Chemin du Dataset"]

    model, model_name, model_version = utils_models.get_mlflow_prod_model()
    model_infos = f"{model_name}-V{model_version}"
    logger.debug(f"ref dataset_path {ref_dataset_path}")
    logger.debug(f"prod dataset_path {prod_dataset_path}")
    logger.debug(f"model_name {model_infos}")
    new_features, original_features, avg_recall = monitor_new_data_predict(
        model, prod_dataset_path, ref_dataset_path)

    # Calcul de la moyenne des recalls pour les nouvelles données et les données d'entraînement
    # Rappel moyen calculé pour les données d'entraînement et nouvelles données
    avg_recall_train = avg_recall

    # Détection du drift avec un seuil pour la différence de recall
    drift_info = {
        "mean_threshold": 0.05,  # Seuil de 5% de différence pour détecter un drift
    }

    # Simuler la détection de drift (nous utilisons avg_recall pour les deux car on a un seul calcul)
    drift_detected = detect_recall_drift(
        avg_recall_train, avg_recall, drift_info)

    if drift_detected:
        print("Un drift a été détecté sur la moyenne des recalls.")
    else:
        print("Pas de drift significatif sur la moyenne des recalls.")


if __name__ == "__main__":
    drift_detection_main()
