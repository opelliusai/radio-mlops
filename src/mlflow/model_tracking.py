'''
Créé le 08/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: MLFlow Model Tracking
-- Execution de l'entrainement (appel aux fonctions src/models)
-- Evaluation du modèle
-- Déclaration des hyperparamètres
-- Enregistrement des métriques
-- Log du modèle
'''

# Imports
import os
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.keras
from sklearn.metrics import f1_score
from mlflow.tracking import MlflowClient
from mlflow import MlflowException
import numpy as np
import pandas as pd
import json
# Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths, dataset_info, current_dataset_label_correspondance, model_hp, model_info, mlflow_info, infolog
from src.utils import utils_data
from src.utils import utils_models
# Import de modules internes: ici les modules liés à la construction /entrainement du modele
from src.models import build_model, train_model
from src.datasets import update_dataset, clean_dataset, image_preprocessing

from src.config.log_config import setup_logging

logger = setup_logging("MLFLOW")
mlflow_uri = mlflow_info["mlflow_tracking_uri"]
client = MlflowClient(
    registry_uri=mlflow_uri, tracking_uri=mlflow_uri)

mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)


def model_tracking(dataset_infos, balance=True,
                   max_epochs=model_hp["max_epochs"],
                   num_trials=model_hp["num_trials"],
                   experiment_name="First Model Tracking"):
    """
    Fonction qui effectue l'entrainement et l'enregistrement d'un modèle

    :param dataset_version_path: Le chemin vers la version du dataset à utiliser.
    :type dataset_version_path: str

    :param balance: Indique si le dataset doit être équilibré avant l'entraînement. Par défaut, True.
    :type balance: bool

    :param max_epochs: Le nombre maximal d'époques pour l'entraînement du modèle.
    :type max_epochs: int

    :param num_trials: Le nombre d'essais pour la recherche des meilleurs hyperparamètres.
    :type num_trials: int

    :param experiment_name: Le nom de l'expérience MLflow dans laquelle les informations du modèle seront enregistrées. Par défaut, "First Model Tracking".
    :type experiment_name: str

    :param init_model_name: Le nom du modèle initial à utiliser, si spécifié. Par défaut, None.
    :type init_model_name: str, optional

    :param init_model_version: La version du modèle initial à utiliser, si spécifié. Par défaut, None.
    :type init_model_version: str, optional

    Si init_model_name et init_model_version ne sont pas spécifiés (None), on récupère le nom et version du modèle en production.

    :returns: L'ID du run MLflow, le nom du modèle, et la version du nouveau modèle enregistré.
    :rtype: tuple

    :raises MlflowException: En cas de problème lors de l'enregistrement du modèle dans MLflow.
    :raises Exception: En cas d'erreur non gérée.
    """
    logger.debug(
        f"MLFLOW - Model tracking initiated with parameters: dataset_infos={dataset_infos}, balance={balance}, max_epochs={max_epochs}, num_trials={num_trials}, experiment_name={experiment_name}")

    model_hp.update({"max_epochs": max_epochs, "num_trials": num_trials})
    logger.debug(
        f"Définition d'une version de modèle pour nommage du fichier metadata")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(
        experiment_name)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Set the model URI
        model_uri = "runs:/{run_id}/model".format(run_id=run.info.run_id)
        logger.debug(f"model_uri {model_uri}")
        # Register the model
        model_details = mlflow.register_model(
            model_uri=model_uri, name=model_info["model_name_prefix"])
        logger.debug(f"model_details {model_details}")
        new_model_info = client.create_model_version(model_info["model_name_prefix"], run.info.artifact_uri + "/model", run.info.run_id, tags={
                                                     "Auteur": model_info["auteur"], "Run_id": run.info.run_id}, description=model_info["model_desc"])
        model_name = new_model_info.name
        logger.debug(f"model_name = {model_name}")
        model_version = new_model_info.version
        logger.debug(f"Nouveau Modele {model_name}-{model_version}")
        if dataset_infos is None:
            dataset_infos = utils_data.get_latest_dataset_info("REF")
        if balance:
            dataset_infos, target_metadata_path = clean_dataset.prepare_dataset_for_model(
                dataset_infos, f"{model_name}-{model_version}")
            preprocessed_data = image_preprocessing.preprocess_data(
                dataset_infos["Chemin du Dataset"], target_metadata_path, model_hp["img_size"], model_hp["img_dim"])
            mlflow.log_artifact(
                target_metadata_path, artifact_path="Dataset_balance")
        else:
            logger.debug(
                f"dataset_infos['Chemin du Dataset'] {dataset_infos}")
            logger.debug(f"model_hp['img_size'] {model_hp['img_size']}")
            logger.debug(f"model_hp['img_dim'] {model_hp['img_dim']}")
            preprocessed_data = image_preprocessing.preprocess_data(
                dataset_infos["Chemin du Dataset"], None, model_hp["img_size"], model_hp["img_dim"])
            logger.debug(f"model_hp['img_size'] {model_hp['img_size']}")
            logger.debug(f"model_hp['img_dim'] {model_hp['img_dim']}")
            logger.debug(f"model_hp['max_epochs'] {model_hp['max_epochs']}")
            logger.debug(f"model_hp['num_trials'] {model_hp['num_trials']}")

        X, y = map(list, zip(*preprocessed_data))
        y = utils_data.label_to_numeric(
            y, current_dataset_label_correspondance)

        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=0.2, random_state=1234)

        mlflow.log_params(
            {"Description": "EfficientNetB0 - Détection d'anomalie pulmonaire"})

        num_classes = 3
        full_run_folder = os.path.join(
            init_paths["main_path"], init_paths["run_folder"])
        full_kt_folder = os.path.join(
            init_paths["main_path"], init_paths["keras_tuner_folder"])
        keras_tuning_history_filename = f"effnetb0_tuning_history_run_{run.info.run_id}.csv"

        best_model, best_hp = build_model.tuner_randomsearch(
            model_hp, full_run_folder, full_kt_folder, keras_tuning_history_filename, X_train, y_train, num_classes)
        mlflow.log_params(best_hp.values)

        model_training_history_csv = f"Covid19_3C_EffnetB0_model_history_run_{run.info.run_id}.csv"
        trained_model, metrics, history = train_model.train_model(
            best_model, model_hp, X_train, y_train, full_run_folder, model_training_history_csv)

        mlflow.log_params(model_hp)

        # Enregistrement des métriques d'évaluation
        X_eval, y_eval = np.array(X_eval), np.array(y_eval)
        eval_metrics = utils_models.get_prediction_metrics(
            trained_model, X_eval, y_eval)

        # Extraction de la matrice de confusion et du rapport de classification du dictionnaire des métriques
        confusion_matrix = eval_metrics.pop('Confusion Matrix', None)
        classification_report = eval_metrics.pop('Classification report', None)

        if confusion_matrix is not None:
            # Enregistrement de la matrice de confusion en tant qu'artefact
            confusion_matrix_df = pd.DataFrame(confusion_matrix)
            confusion_matrix_csv_path = os.path.join(
                full_run_folder, 'confusion_matrix.csv')
            confusion_matrix_df.to_csv(confusion_matrix_csv_path, index=False)
            mlflow.log_artifact(confusion_matrix_csv_path)

        if classification_report is not None:
            # Enregistrment des éléments du rapport de classification en tant que métriques individuelles
            for label, metrics_dict in classification_report.items():
                if isinstance(metrics_dict, dict):
                    for metric_name, value in metrics_dict.items():
                        mlflow.log_metric(f"eval_{label}_{metric_name}", value)
                else:
                    mlflow.log_metric(f"eval_{label}", metrics_dict)

            # Enregistrement du rapport de classification complet en tant qu'artefact JSON
            classification_report_json_path = os.path.join(
                full_run_folder, 'classification_report.json')
            with open(classification_report_json_path, 'w') as f:
                json.dump(classification_report, f)
            mlflow.log_artifact(classification_report_json_path)

        # Enregistrement des autres métriques
        mlflow.log_metrics({f"eval_{k}": v for k, v in eval_metrics.items()})

        mlflow.log_params({"Dataset": dataset_infos["Chemin du Dataset"]})
        mlflow.log_artifact(
            dataset_infos["Chemin du Dataset"], artifact_path="Dataset")
        mlflow.keras.log_model(trained_model, "model")

        return new_model_info.run_id, new_model_info.name, new_model_info.version, experiment_id


def model_retrain(model, dataset_infos,
                  init_model_name,
                  init_model_version,
                  max_epochs=model_hp["max_epochs"],
                  num_trials=model_hp["num_trials"],
                  experiment_name="Model Retraining",

                  ):
    logger.debug(
        f"MLFLOW - Model Retraining - model_retrain(dataset_infos={dataset_infos},max_epochs={max_epochs},num_trials={num_trials},init_model_name={init_model_name},init_model_version={str(init_model_version)})")
    # Etapes
    # 1. Chargement des données
    # 2. Split des données 80/20 Train/Eval (Un resplit est fait lors de l'entrainement)
    # 3. Construction du modèle
    # 4. Recherche des meilleurs paramètres sur 80% des données
    # 5. Entrainement du MEILLEUR modèle sur 80% des données (60% train, 20% val)
    # 5b. Enregistrement des données d'entrainement (training history, classification report etc.)
    # 6. Test du modèle (20% eval)
    # 7. Enregistrement des hyperparamètres
    # 8. Enregistrement des métriques
    # 9. Log du modèle
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    logger.info(f"MLFlow - Model Retraining Experiment_ID {experiment_id}")
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_param("init_model", init_model_name +
                         "_V"+str(init_model_version))
        model_name = model_info["model_name_prefix"]
        run_id = run.info.run_id
        logger.debug(f"RUN ID {run_id}")
        mlflow.log_param(
            "Description", "RETRAIN - EfficientNetB0 - Détection d'anomalie pulmonaire")
        logger.info("Chargement des données")
        dataset_path = dataset_infos["Chemin du Dataset"]
        logger.debug(f"Chemin du dataset {dataset_path}")
        logger.info("Preprocessing et labellisation X,y")
        model_hp["max_epochs"] = max_epochs
        model_hp["num_trials"] = num_trials
        preprocessed_data = image_preprocessing.preprocess_data(
            dataset_path, None,  model_hp["img_size"], model_hp["img_dim"])
        X, y = map(list, zip(*preprocessed_data))
        logger.debug("Conversion numerique des labels")
        logger.debug("Chargement du tableau de correspondance")
        correspondance = current_dataset_label_correspondance
        y = utils_data.label_to_numeric(y, correspondance)
        logger.debug(f"y numeric")
        logger.debug(f"{y}")

        logger.info(
            "Split des données 80/20 Train/Eval (Un resplit est fait lors de l'entrainement)")
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=0.2, random_state=1234)
        logger.info("Construction du modèle")
        logger.debug("Chargement des chemins")

        logger.info("Recherche des meilleurs paramètres sur 80p des données ")
        full_run_folder = os.path.join(
            init_paths["main_path"], init_paths["run_folder"])  # ./data/processed/mflow
        # ./data/processed/keras_tuner
        logger.info(
            "Entrainement du MEILLEUR modèle sur 80p des données (60% train, 20% val)")
        model_training_history_csv = f"Covid19_3C_EffnetB0_model_history_run_{run_id}.csv"
        trained_model, metrics, history = train_model.train_model(
            model, model_hp, X_train, y_train, full_run_folder, model_training_history_csv)

        logger.debug(f"Metrics {metrics}")

        logger.info(
            "Enregistrement des données d'entrainement (training history, classification report etc.)")
        logger.debug("PLUS TARD")

        logger.info("Test du modèle (20% eval)")
        logger.debug("PLUS TARD")

        logger.info("Enregistrement des hyperparamètres")
        for key, value in model_hp.items():
            logger.debug(f"{key}: {value}")
            mlflow.log_param(key, value)

        # Métriques supplémentaires

        logger.info("Récupération des métriques d'évaluation")
        logger.debug(
            f"Type X_eval {type(X_eval)} / Type y_eval {type(y_eval)}")
        logger.debug(f"Conversion en np.array")
        X_eval = np.array(X_eval)
        y_eval = np.array(y_eval)
        logger.debug(
            f"Type X_eval {type(X_eval)} / Type y_eval {type(y_eval)}")
        eval_metrics = utils_models.get_prediction_metrics(
            trained_model, X_eval, y_eval)
        logger.debug(f"Eval metrics {eval_metrics}")

        logger.info("Enregistrement des métriques")
        for key, value in eval_metrics.items():
            logger.debug(f"eval_{key}: {value}")
            mlflow.log_param(key, value)

        # Enregistrement des données utilisées
        logger.info("Enregistrement des données utilisées")
        mlflow.log_param("Dataset", dataset_infos["Dataset Name"])
        mlflow.log_artifact(dataset_path, artifact_path="Dataset")

        # Enregistrement du modèle
        logger.info("Log du modèle")
        mlflow.keras.log_model(trained_model, "model")
        mlflow.log_param("model", trained_model)
        artifacts_uri = run.info.artifact_uri
        model_path = f"{artifacts_uri}/model"
        logger.debug(f"artifacts_uri {artifacts_uri}")
        logger.debug(f"Model path {model_path}")

        # Récupérer ou créer un model registry
        try:
            registry = client.get_registered_model(model_name)
            logger.debug(f"Model registry {model_name} existe")

        except MlflowException as e:
            logger.debug(
                f"Model registry {model_name} n'existe pas - Création d'un nouveau registre")
            client.create_registered_model(
                name=model_name, description=model_info["model_desc"])

        # Mise à jour d'information sur le modèle registry

        tags_v = {"Auteur": model_info["auteur"], "Run_id": f"{run_id}"}
        # Créer une nouvelle version du modèle et ajouter des infos spéciques à la version
        client.create_model_version(
            model_name, model_path, run_id, tags_v, description=model_info["model_desc"])
        logger.debug(
            f"Model version {client.get_latest_versions(model_name)[0].version}")
        # Ajout
        logger.debug(f"Enregistrement terminé")
        return run_id, client.get_latest_versions(model_name)[0].version, experiment_id


def main(retrain=False,
         model_name=None,
         model_version=None,
         include_prod_data=False,
         balance=True,
         dataset_version=None,
         max_epochs=model_hp["max_epochs"],
         num_trials=model_hp["num_trials"]):
    logger.debug("model_tracking.main()")
    logger.debug(f"model_hp {model_hp}")
    logger.debug(f"retrain={retrain}, model_name={model_name}, model_version={model_version},include_prod_data={include_prod_data}, balance= {balance}, dataset_version={dataset_version},max_epochs={max_epochs},num_trials={num_trials}")

    logger.debug(f"Initialisation du client MLFlow")

    logger.debug(
        f"Identification du scénario en fonction des valeurs renseignées")
    if retrain:
        logger.debug(f"Réentrainement du modèle")
        if model_name is None:
            logger.debug(
                f"Modèle non renseigné - Réentrainement du modèle de production")
            model, model_name, model_version = utils_models.get_mlflow_prod_model()
            if model is None:
                logger.error(f"Modèle de production non trouvé")
            else:
                logger.debug(
                    f"Modèle à réentrainer {model_name}-{model_version} sur les données de production")
        # SCENARIO 1 ici - Réentrainement modèle
        logger.debug(
            f"Début du réentrainement du Modèle {model_name}-{model_version} (Dernier modèle ou modèle renseigné)")
        dataset_prod_infos = utils_data.get_latest_dataset_info("PROD")
        logger.debug(f"dataset_prod_infos = {dataset_prod_infos}")
        if dataset_prod_infos is None:
            logger.error(f"Aucun Dataset de production trouvé")
        run_id, model_version, experiment_id = model_retrain(model, dataset_prod_infos,
                                                             model_name,
                                                             model_version,
                                                             max_epochs=model_hp["max_epochs"],
                                                             num_trials=model_hp["num_trials"],
                                                             experiment_name="Model Retraining"
                                                             )
    else:
        if dataset_version is None:
            logger.debug(
                f"Dataset non renseigné - Utilisation du dernier Dataset de Reference")
            dataset_infos = utils_data.get_latest_dataset_info("REF")
            if dataset_infos is None:
                logger.error(f"Aucun Dataset trouvé")
            else:
                logger.debug(f"dataset_infos = {dataset_infos}")
                dataset_version = dataset_infos["Dataset Name"]
                logger.debug(f"Dataset à utiliser {dataset_version}")
        else:
            logger.debug(
                f"Dataset renseigné {dataset_version}")
            dataset_infos = utils_data.get_dataset_info_by_dataset_name(
                dataset_version)
            if dataset_info is None:
                logger.error(f"Dataset {dataset_version} non trouvé")

        logger.debug(
            f"Entrainement d'un nouveau modèle (Dataset renseigné ou dernier dataset)")
        if include_prod_data:
            logger.debug(
                f"Inclure data de PROD - Mettre d'abord à jour le dataset et récupérer la nouvelle version")
            dataset_infos = update_dataset.update_dataset_ref(
                dataset_infos["Chemin du Dataset"], "PROD")
            dataset_version = dataset_infos["Dataset Version"]
        else:
            logger.debug(f"Ne pas inclure data de PROD")
        # SCENARIO 2 ici - Entrainement modèle from scratch / Avec ou sans données de PROD, sur la version définie

        run_id, model_name, model_version, experiment_id = model_tracking(
            dataset_infos, balance, max_epochs, num_trials)
    logger.info("Fin du processus")
    logger.debug(
        f"run_id {run_id} / model_name {model_name} / model_version {model_version}")
    experiment_link = utils_models.get_mlflow_link(experiment_id, run_id)

    return run_id, model_name, model_version, experiment_link
