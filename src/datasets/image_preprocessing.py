'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Image Preprocessing pour EfficientNetB0

'''

# IMPORTS
# Imports externes
from tensorflow.keras.applications.efficientnet import preprocess_input as pp_effnet
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import csv
# Imports internes
from src.utils import utils_data, utils_models
from src.datasets import clean_dataset
from src.config.log_config import setup_logging
# Redirection vers le fichier de log radio-mlops_datasets.log
logger = setup_logging("datasets")

# FONCTIONS PRINCIPALES
# 1 - PREPROCESSING des données


def preprocess_data(dataset_path, model_name_metadata_path=None, size=224, dim=3):
    logger.debug(
        f"---------image_preprocessing.py----Fonction  preprocess_data(dataset_path={dataset_path},model_name_metadata_path={model_name_metadata_path},size={size},dim={dim})")
    """
    Fonction qui prétraite les données d'image pour les modèles de réseau de neurones.
    Le fichier metadata.csv présent dans le dataset permet d'identifier le label qui peut être différent du nom du sous-répertoire.
    Cette fonction peut être précédée par une fonction d'équilibrage du nombre de données par classe, traitée et logués dans un ficheir metadata.csv
    Fonction : clean_dataset.prepare_dataset_for_model
    :param dataset_path: Chemin du dataset
    :param model_name_metadata_path : Chemin du fichier metadata contenant les images qui seront utilisées par le modèle
    :param size: Entier, la taille cible à laquelle chaque image est redimensionnée. Pour efficientNetB0, ce sera 224
    :param dim: Entier, le nombre de dimensions de couleur requis (1 pour les niveaux de gris, 3 pour RGB). Pour efficientNetB0, le dim sera 1

    Applique la fonction de preprocessing de l'architecture EfficientNetB0.
    On isole cette partie dans une fonction séparé pour le rendre scalable en cas de changement de choix d'architecture (resnet etc.)
    :return les objets images et labels associés
    :raises Exception: Si :
        - FileNotFoundError Le fichier metadata.csv est erroné
        - Le fichier cherché n'est pas présent
        - ou toute autre erreur
    """

    # Initialisation des listes pour stocker les images et les étiquettes
    data = []
    # Parcours du dataset et lecture du fichier metadata.csv : Celui du dataset si metadata du modele n'est pas renseigné
    metadata_path = os.path.join(
        dataset_path, 'metadata.csv') if model_name_metadata_path is None else model_name_metadata_path
    try:
        logger.debug(f"Ouverture du fichier {metadata_path}")
        with open(metadata_path, 'r') as f:
            # Ignore la première ligne (en-tête)
            reader = csv.DictReader(f)
            # Récupérer le label de la colonne "Classe" et le chemin de l'image dans "Sous-répertoire CIBLE" et "Nom de fichier"
            logger.debug(
                "Récupérer le label de la colonne 'Classe' et le chemin de l'image dans 'Sous-répertoire CIBLE' et 'Nom de fichier'")
            for row in reader:
                logger.debug(f"row {row}")
                label = row["Classe"]
                rep_src = row["Sous-répertoire"]
                img_name = row["Nom de fichier"]
                img_path = os.path.join(dataset_path, rep_src, img_name)
                logger.debug(f"label={label}")
                logger.debug(f"rep_src={rep_src}")
                logger.debug(f"img_name={img_name}")
                logger.debug(f"img_path={img_path}")
                try:
                    logger.debug(
                        f"Preprocessing d'une image via la fonction preprocess_one_image()")
                    if label.lower() == "unlabeled":
                        logger.debug(f"Ignorer UNLABELED")
                    else:
                        img_array = preprocess_one_image(img_path, size, dim)
                        logger.debug(
                            "Ajout de l'image et son label dans la liste des données à retourner")
                        data.append((img_array, label))
                except Exception as e:
                    logger.error(
                        f"Erreur de processing sur l'image {img_path}: {str(e)}")
                    raise Exception(
                        f"Erreur de processing sur l'image {img_path}: {str(e)}")
            logger.info(
                "Images preprocessés et retournés avec leur labels correspondant")
            logger.debug(f"Taille data {len(data)}")
            return data
    except FileNotFoundError:
        logger.error(
            f"Le fichier metadata.csv ou model_name_metadata_path n'ont pas été trouvés dans le répertoire {dataset_path} ou {model_name_metadata_path}")
        raise FileNotFoundError()

# 2 - PREPROCESSING de données non labellisées


def preprocess_unlabeled_data(dataset_path, size, dim):
    logger.debug(
        f"---------image_preprocessing.py----Fonction  preprocess_unlabeled_data(dataset_path={dataset_path},size={size},dim={dim})")
    """
    Fonction qui prétraite les données d'image non labellisées pour les modèles de réseau de neurones.
    Le fichier metadata.csv présent dans le dataset permet d'identifier le label qui peut être différent du nom du sous-répertoire.
    :param dataset_path: Chemin vers le répertoire contenant les images non labellisées.
    :param size: Entier, la taille cible à laquelle chaque image est redimensionnée. Pour efficientNetB0, ce sera 224
    :param dim: Entier, le nombre de dimensions de couleur requis (1 pour les niveaux de gris, 3 pour RGB). Pour efficientNetB0, le dim sera 1

    Applique la fonction de preprocessing de l'architecture EfficientNetB0.
    On isole cette partie dans une fonction séparé pour le rendre scalable en cas de changement de choix d'architecture (resnet etc.)
    :return les objets images
    :raises Exception: Si :
        - FileNotFoundError Le fichier metadata.csv est erroné
        - Le fichier cherché n'est pas présent
        - ou toute autre erreur
    """
    # Initialisation des listes pour stocker les images et les étiquettes
    data = []

    # Parcourt du dataset et lecture du fichier metadata.csv
    try:
        logger.debug("Ouverture du fichier metadata.csv")
        with open(os.path.join(dataset_path, 'metadata.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                logger.debug(f"row {row}")
                rep_src = row["Sous-répertoire"]
                img_name = row["Nom de fichier"]
                img_path = os.path.join(dataset_path, rep_src, img_name)
                logger.debug(f"rep_src={rep_src}")
                logger.debug(f"img_name={img_name}")
                logger.debug(f"img_path={img_path}")
                try:
                    img_array = preprocess_one_image(img_path, size, dim)
                    data.append(img_array)
                except Exception as e:
                    logger.error(
                        f"Erreur de processing sur l'image {img_path}: {str(e)}")
                    raise Exception(
                        f"Erreur de processing sur l'image {img_path}: {str(e)}")
            logger.info(
                "Images preprocessés et retournés avec leur labels correspondant")
            logger.debug(f"Taille data {len(data)}")
            return data
    except FileNotFoundError:
        logger.error(
            f"Le fichier metadata.csv n'a pas été trouvé dans le répertoire {dataset_path}")
        raise FileNotFoundError()

# 3 - Preprocessing d'une imge


def preprocess_one_image(img_path, size=224, dim=3):
    """
    Fonction qui prétraite une image
    :param size: Entier, la taille cible à laquelle chaque image est redimensionnée. Pour efficientNetB0, ce sera 224
    :param dim: Entier, le nombre de dimensions de couleur requis (1 pour les niveaux de gris, 3 pour RGB). Pour efficientNetB0, le dim sera 1

    Applique la fonction de preprocessing de l'architecture EfficientNetB0.
    On isole cette partie dans une fonction séparé pour le rendre scalable en cas de changement de choix d'architecture (resnet etc.)
    :return les objets images
    :raises Exception: En cas d'erreur de preprocessing de l'image
    """
    logger.debug(
        f"---------image_preprocessing.py----Fonction  preprocess_one_image(img_path={img_path},size={size},dim={dim})-----------")
    try:
        img = load_img(img_path, target_size=(size, size),
                       color_mode='grayscale' if dim == 1 else 'rgb')
        img_array = img_to_array(img)
        img_array = pp_effnet(img_array)
        return img_array
    except Exception as e:
        logger.error(f"Erreur de processing sur l'image {img_path}: {str(e)}")
        raise Exception(
            f"Erreur de processing sur l'image {img_path}: {str(e)}")

# Fonction MAIN


def main_dataset():
    logger.debug(f"---------image_preprocessing.py----Fonction  main()")
    try:
        dataset_infos = utils_data.get_latest_dataset_info("REF")
        dataset_path = dataset_infos["Chemin du Dataset"]
        logger.debug(
            "Trainement du dataset courant : dataset_path= {dataset_path}")
        data = preprocess_data(dataset_path)
        i = 0
        for (img, label) in data:
            i += 1
            logger.debug(f" {i} - Label {label}")
        logger.info("Fin de la fonction main()")
        return "OK"

    except Exception as e:
        logger.error(f"Une erreur s'est produite : {e}")
        return e


def main_dataset_model():
    logger.debug(
        f"---------image_preprocessing.py----Fonction  main_dataset_model()")
    try:
        dataset_infos = utils_data.get_latest_dataset_info("REF")
        dataset_path = dataset_infos["Chemin du Dataset"]
        logger.debug(
            f"Trainement du dataset courant : dataset_path= {dataset_path}")
        _, model_name, model_version = utils_models.get_mlflow_prod_model()
        logger.debug(
            f"Trainement du modèle : model_name/Version= {model_name}-{model_version}")
        dataset_infos, target_metadata_path = clean_dataset.prepare_dataset_for_model(
            dataset_infos, f"{model_name}-{model_version}")

        data = preprocess_data(dataset_path, target_metadata_path)
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
    # main_dataset()
    main_dataset_model()
