import numpy as np
import mlflow
import os
import time
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


def monitor_new_data_predict(feature_extractor, new_images, dataset_path):

    new_data = preprocess_unlabeled_data(new_images, 224, 3)
    train_data = preprocess_data(dataset_path, size=224, dim=3)
    X_train, _ = map(list, zip(*train_data))
    new_data = np.array(new_data)
    X_train = np.array(X_train)

    # Extrait les caractéristiques des nouvelles images en utilisant le modèle (par exemple, les activations d'une couche cachée)
    new_features = feature_extractor.predict(new_data, batch_size=32)
    logger.debug(f"new_features {new_features}")
    # Comparer la distribution des nouvelles caractéristiques avec celles des données d'entraînement initiales
    original_features = feature_extractor.predict(X_train, batch_size=32)
    logger.debug(f"original_features {original_features}")

    return new_features, original_features


def monitor_new_data_mean_std(feature_extractor, new_images, dataset_path):
    logger.debug(
        f"----------monitor_new_data_mean_std(feature_extractor,new_images,dataset_path)---------")
    start_time = time.time()
    new_features, original_features = monitor_new_data_predict(
        feature_extractor, new_images, dataset_path)
    # Calcul des statistiques pour détecter la dérive
    new_mean = np.mean(new_features, axis=0)
    original_mean = np.mean(original_features, axis=0)

    new_std = np.std(new_features, axis=0)
    original_std = np.std(original_features, axis=0)
    end_time = time.time()
    # Calcul du temps de prédiction
    temps_calcul = round(end_time-start_time, 2)
    return temps_calcul, new_mean, original_mean, new_std, original_std


def log_mlflow_metrics(new_mean, original_mean, new_std, original_std):
    experiment_name = "First Model Tracking"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    logger.info(f"MLFlow - Model Tracking Experiment_ID {experiment_id}")
    # Enregistrer ces statistiques dans MLflow
    with mlflow.start_run(experiment_id=experiment_id) as run:
        logger.debug(f"new_mean.mean() {new_mean.mean()}")
        logger.debug(f"original_mean.mean() {original_mean.mean()}")
        logger.debug(f"new_std.mean() {new_std.mean()}")
        logger.debug(f"original_std.mean() {original_std.mean()}")

        mlflow.log_metric("new_mean", new_mean.mean())
        mlflow.log_metric("original_mean", original_mean.mean())
        mlflow.log_metric("new_std", new_std.mean())
        mlflow.log_metric("original_std", original_std.mean())


def detect_drift(new_mean, original_mean, new_std, original_std):
    # Comparaison des statistiques
    logger.debug(f"new_mean {new_mean.mean()}")
    logger.debug(f"original_mean {original_mean.mean()}")
    logger.debug(f"new_std {new_std.mean()}")
    logger.debug(f"original_std {original_std.mean()}")

    mean_diff = abs(new_mean.mean() - original_mean.mean())
    std_diff = abs(new_std.mean() - original_std.mean())

    logger.debug(f"Mean difference: {mean_diff}")
    logger.debug(f"Standard deviation difference: {std_diff}")

    # Détection de dérive basée sur un seuil
    mean_threshold = drift_info["mean_threshold"]
    std_threshold = drift_info["std_threshold"]

    if mean_diff > mean_threshold:
        logger.debug(f"Drift détecté sur la moyenne! {mean_diff}")
        return True
    elif std_diff > std_threshold:
        logger.debug(f"Drift détecté sur l'écart type! {std_diff}")
        return True
    else:
        print("Pas de drift significatif détecté.")
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
    temps_calcul, new_mean, original_mean, new_std, original_std = monitor_new_data_mean_std(
        model, prod_dataset_path, ref_dataset_path)
    drift = detect_drift(new_mean, original_mean, new_std, original_std)
    drift_calculation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Calcul de la dérive (drift)
    mean_diff = np.abs(new_mean.mean() - original_mean.mean())
    std_diff = np.abs(new_std.mean() - original_std.mean())

    if log == True:
        # log MLFlow
        log_mlflow_metrics(new_mean, original_mean, new_std, original_std)
        utils_models.save_drift_metrics_data(model_infos, new_mean, original_mean, new_std, original_std, mean_diff, std_diff,
                                             drift, drift_calculation_date, temps_calcul)

    return model_infos, new_mean, original_mean, new_std, original_std, mean_diff, std_diff, drift


if __name__ == "__main__":
    drift_detection_main()
