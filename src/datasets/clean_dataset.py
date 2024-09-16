'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fichier de définition des images à utiliser pour un modèle depuis un dataset
Stratégie: 
- Identifier la classe minimale
- Extraire aléatoirement le nombre de cette classe minimale pour toutes les autres classes
- Loguer la liste des ficheirs sélectionnés dans un fichier metadata propre au modèle et au dataset
- Cela permet de réutiliser le même dataset sur plusieurs modèles sans en recréer des versions spécifiques à chaque modèle

'''
# IMPORTS
# Imports externes
import os
import json
import csv
import os
# Imports internes
from src.utils import utils_data, utils_models
from src.datasets import image_preprocessing
from src.config.run_config import init_paths
from src.config.log_config import setup_logging
logger = setup_logging("datasets")

# FONCTIONS


def prepare_dataset_for_model(dataset_infos, model_name):
    """
    Prépare le dataset pour un modèle spécifique
    1 - Lit les infos du dataset dans le log data
    2 - Récupère le nombre d'image par classe par rapport à la valeur de la classe minimale
    3 - Sélectionne aléatoirement les données en se basant sur le fichier metadata.csv du dataset
    4 - Alimente un fichier metadata spécifique au modèle pour indiquer les images qui seront utilisées
    Nom du fichier : VERSION-DATAESET-NOM_MODELE/VERSION_metadata.csv
    Répertoire  : data/processed/dataset_name/
    5 - Ne copie pas les images, ne dédie pas un répertoire à chaque modèle
    """
    logger.debug(
        f"-----prepare_dataset_for_model(dataset_infos={dataset_infos}, model_name={model_name})----")
    # Initialisation des valeurs / chemins
    dataset_name = dataset_infos["Dataset Name"]
    logger.debug(f"dataset_name {dataset_name}")
    nb_image_par_classe = dataset_infos["Nombre d'images par classe"]
    logger.debug(f"nb_image_par_classe {nb_image_par_classe}")
    nb_image_par_classe = nb_image_par_classe.replace("'", '"')
    nb_image_par_classe = json.loads(nb_image_par_classe)
    logger.debug(f"nb_image_par_classe {nb_image_par_classe}")
    # Nombre d'image à récupérer
    min_nb_image_par_classe = min(nb_image_par_classe.values())
    dataset_metadata_path = os.path.join(
        dataset_infos["Chemin du Dataset"], "metadata.csv")
    target_metadata_path = os.path.join(
        init_paths["main_path"], init_paths["models_data_path"], model_name, f"{dataset_name}_{model_name}_metadata.csv")
    logger.debug(f"nb_image_par_classe {nb_image_par_classe}")
    logger.debug(f"min_nb_image_par_classe {min_nb_image_par_classe}")
    logger.debug(f"dataset_metadata_path {dataset_metadata_path}")
    logger.debug(f"target_metadata_path {target_metadata_path}")

    # Création du répertoire destination
    os.makedirs(os.path.dirname(target_metadata_path), exist_ok=True)
    # Lecture du fichier metadata.csv du dataset
    with open(dataset_metadata_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        # Initialisation du fichier csv de destination
        with open(target_metadata_path, 'w', newline='') as csvfile_out:
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
            writer.writeheader()
            # Initialisation des compteurs
            compteur_classe = {classe.strip(
                "'"): 0 for classe in nb_image_par_classe.keys()}

            logger.debug(f"compteur_classe {compteur_classe}")
            # Parcours du fichier metadata.csv
            for row in reader:
                # Récupération de la classe
                logger.debug(f"row {row}")
                classe = row["Classe"]
                logger.debug(f"classe {classe}")
                # Si le nombre d'image pour la classe est inférieur au nombre d'image minimal
                if compteur_classe[classe] < min_nb_image_par_classe:
                    # Incrémentation du compteur
                    compteur_classe[classe] += 1
                    # Ecriture de la ligne dans le fichier csv de destination
                    writer.writerow(row)
    # Affichage des compteurs
    logger.debug(f"compteur_classe {compteur_classe}")
    return dataset_infos, target_metadata_path


def main_prepare():
    logger.debug(
        f"---------image_preprocessing.py----Fonction  main_prepare()")
    try:
        dataset_infos = utils_data.get_latest_dataset_info("REF")
        dataset_path = dataset_infos["Chemin du Dataset"]
        logger.debug(
            f"Trainement du dataset courant : dataset_path= {dataset_path}")
        _, model_name, model_version = utils_models.get_mlflow_prod_model()
        logger.debug(
            f"Trainement du modèle : model_name/Version= {model_name}-{model_version}")
        dataset_infos, target_metadata_path = prepare_dataset_for_model(
            dataset_infos, f"{model_name}-{model_version}")

        data = image_preprocessing.preprocess_data(
            dataset_path, target_metadata_path)
        i = 0
        for (_, label) in data:
            i += 1
            logger.debug(f" {i} - Label {label}")
        logger.info("Fin de la fonction main_dataset_model()")
        return "OK"

    except Exception as e:
        logger.error(f"Une erreur s'est produite : {e}")
        return e


if __name__ == "__main__":
    main_prepare()
