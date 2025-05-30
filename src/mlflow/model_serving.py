'''
Créé le 08/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: MLFlow Model Serving
-- Déploiement du modèle sur MLFlow
'''

# Imports
import os
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.keras
from sklearn.metrics import f1_score
from mlflow.tracking import MlflowClient

# Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import model_info, mlflow_info
from src.utils import utils_models
# Import de modules internes: ici les modules liés à la construction /entrainement du modele
from src.config.log_config import setup_logging

logger = setup_logging("MLFLOW")
mlflow_uri = mlflow_info["mlflow_tracking_uri"]
client = MlflowClient(
    registry_uri=mlflow_uri, tracking_uri=mlflow_uri)

mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)


def model_version_serving(model_name=model_info["model_name_prefix"], num_version=2):
    logger.debug(
        f"---------model_version_serving(model_name={model_name},num_version={num_version})-----------")
    # Archiver le modèle de production à l'état Archived avant de mettre un nouveau modèle en production
    model_prod_version = utils_models.get_mlflow_prod_version()
    logger.debug(f"model_prod_version {model_prod_version}")
    if model_prod_version is not None:
        logger.debug(
            f"Archivage du modele actuel {model_name}-{model_prod_version}")

        client.transition_model_version_stage(
            name=model_name,
            version=model_prod_version,
            stage="Archived"
        )
        utils_models.archive_model_pred_drift_logs(
            f"{model_name}_{model_prod_version}")

    else:
        logger.debug(
            f"Aucun modèle en production pour {model_name}, pas d'archivage nécessaire.")

    # Déployer le nouveau modèle
    logger.debug(f"Déploiement du nouveau modèle {model_name}-{num_version}")
    client.transition_model_version_stage(
        name=model_name,
        version=num_version,
        stage="Production"
    )

    # Récupérer tous les modèles enregistrés
    logger.debug(
        "Suppression du tag de tous les autres modèles - Un modèle avec le tag ready_prod uniquement")
    registered_models = client.search_registered_models()
    logger.debug(f"Registered models {registered_models}")
    for model in registered_models:
        if model.name.lower() == model_name.lower():
            logger.debug(
                f"Suppression du tag deploy_prod de toutes les versions du modèle {model.name}")
            for version in model.latest_versions:
                logger.debug(f"Version {version.version}")
                client.delete_model_version_tag(
                    name=model_name, version=version.version, key='deploy_flag')
                logger.debug(
                    f"Model Name: {model_name}, Version: {version.version}")


def auto_model_serving(model_name=model_info["model_name_prefix"]):
    """
    Déploiement automatique en Production de modèle ayant le tag deploy_flag = production_ready
    Utilisé en batch ou par l'admin pour un déploiement automatique
    """
    logger.debug(
        f"----------auto_model_serving(model_name={model_name})----------")
    # Récupérer tous les modèles enregistrés
    model_versions = client.search_model_versions(f"name='{model_name}'")
    logger.debug(f"model_versions {model_versions}")
    # Parcourir chaque modèle enregistré
    for version in model_versions:
        logger.debug(f"Version {version.version}")
        # Récupérer les détails de la version du modèle
        tags = version.tags
        logger.debug(f"Tags: {tags}")
        # Vérifier si le tag `deploy_flag` est présent et a la valeur `production_ready`
        logger.debug(
            "verif si deploy_flag prodcution ready est trouvé")
        logger.debug(f"Tag deploy_flag {tags.get('deploy_flag')}")
        if 'deploy_flag' in tags.keys() and tags.get('deploy_flag') == 'production_ready':
            logger.debug(f"Tag {tags.get('deploy_flag')} trouvé")
            model_version_serving(model_name, version.version)
            logger.debug(
                f"Version déployée en Prod et désactivation des FLAGS")
            logger.debug(
                f"Modele traité {model_name}-{version.version}")
            return model_name, version.version
        else:
            logger.debug(f"Skip car Tag est None")
            # Si aucun modèle n'a été déployé, retourner None
        logger.debug(
            f"Aucun modèle avec le tag 'deploy_flag' à 'production_ready' n'a été trouvé.")
    return None, None


def check_flags():
    # Initialiser le client MLflow
    registered_models = client.search_registered_models()

    # Parcourir chaque modèle enregistré
    for model in registered_models:
        logger.debug(f"Model Name: {model.name}")

        # Parcourir les différentes versions du modèle
        for version in model.latest_versions:
            logger.debug(f"  Version: {version.version}")

            # Récupérer les informations sur la version du modèle, y compris les tags
            model_version_details = client.get_model_version(
                name=model.name, version=version.version)

            # Accéder aux tags de la version
            tags = model_version_details.tags

            # Lister tous les tags pour chaque version
            for tag_key, tag_value in tags.items():
                logger.debug(f"    Tag: {tag_key}, Value: {tag_value}")


def make_prod_ready(model_name=model_info["model_name_prefix"], num_version=2):
    logger.debug(
        f"-----------make_prod_ready(model_name={model_name},num_version={num_version})-------")
    client.set_model_version_tag(
        model_name, num_version, "deploy_flag", "production_ready")
    logger.info(f"Modèle {model_name}-{num_version} mis à jour")


def main(num_version=2):
    model_version_serving(model_info["model_name"], num_version)


if __name__ == "__main__":
    main()
