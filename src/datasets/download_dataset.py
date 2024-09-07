'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Téléchargement de la base de données
Appel de l'API Kaggle et stockage dans un répertoire local
'''

# IMPORTS
# Imports externes
from datetime import datetime
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import uuid
import shutil
import traceback
# Imports internes
from src.utils import utils_data
from src.config.run_config import init_paths, dataset_info
from src.config.log_config import setup_logging
# Redirection vers le fichier de log radio-mlops_datasets.log
logger = setup_logging("datasets")

# FONCTIONS PRINCIPALES
# 1 - Téléchargement du Dataset Kaggle


def dl_dataset_kaggle_api(url, destination):
    """
    Télécharge et extrait un dataset depuis Kaggle vers un dossier de destination.

    :param url: L'URL du dataset Kaggle à télécharger.
    :type url: str

    :param destination: Le chemin du dossier où extraire le contenu du dataset.
    :type destination: str

    :raises Exception: En cas d'erreur non gérée.
    """

    logger.debug(
        f"---------donwload_dataset.py----Fonction  get_dataset_kaggle_api(url={url}, destination={destination}---------------")

    try:
        # Initialiser l'API Kaggle
        logger.info("Initialisation de l'API Kaggle")
        api = KaggleApi()
        api.authenticate()

        # On crée le dossier du dataset s'il n'existe pas
        logger.debug("Création du Dossier de destination s'il n'existe pas ")
        logger.debug(f"destination = {destination}")
        if not os.path.exists(destination):
            os.makedirs(destination)

        logger.info(f"Téléchargement du dataset {url} dans {destination}")
        api.dataset_download_files(url, path=destination, unzip=True)

        logger.info(f"Dataset copié dans le répertoire {destination}")

        return "OK"

    except Exception as e:
        logger.error(f"Une erreur s'est produite : {e}")
        return e


# 2 - Génération du fichier metadata.csv

def build_dataset_kaggle_metadata(dataset_path):
    """
    Construit le fichier `metadata.csv` à partir du dossier de dataset.

    Stocke les informations de tous les fichiers présents dans le dataset brut.
    Définit avec un flag "Ignored" les fichiers à ignorer dans le périmètre projet 
    (masks, lung opacity, fichiers xlsx et readme à la racine).

    :param dataset_path: Le chemin du dataset en local qui contiendra le fichier `metadata.csv` .
    :type dataset_path: str

    :raises Exception: En cas d'erreur non gérée.
    """

    logger.debug(
        f"---------build_dataset_kaggle_metadata(dataset_path={dataset_path})---------")

    if not os.path.exists(dataset_path):
        logger.error(f"Le dataset {dataset_path} n'existe pas")
        return None

    # Placer le fichier metadata.csv dans le répertoire parent du dataset
    metadata_path = os.path.join(dataset_path, "metadata.csv")

    # Récupérer le chemin des logs de dataset KAGGLE
    dataset_logging_path = utils_data.get_logging_path("KAGGLE")

    # Déplacer les sous-dossiers du répertoire intermédiaire vers le répertoire parent
    intermediate_directory = os.path.join(
        dataset_path, dataset_info["KAGGLE_dataset_prefix"])
    if os.path.exists(intermediate_directory):
        for subdir in os.listdir(intermediate_directory):
            src = os.path.join(intermediate_directory, subdir)
            dest = os.path.join(dataset_path, subdir)
            if os.path.isdir(src):
                shutil.move(src, dest)
                logger.debug(f"Déplacement de {src} vers {dest}")

        # Supprimer le répertoire intermédiaire
        os.rmdir(intermediate_directory)
        logger.debug(
            f"Répertoire intermédiaire {intermediate_directory} supprimé.")

    # Définition du header du fichier metadata.csv
    df, metadata_path = utils_data.initialize_metadata_file(
        dataset_path, "kaggle")
    date_ajout = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_source = "KAGGLE"
    dataset_uid = str(uuid.uuid4())

    # Récupération des informations du dernier dataset pour comparaison
    if os.path.exists(dataset_logging_path):
        latest_dataset_info = utils_data.get_latest_dataset_info("KAGGLE")
        logger.debug(f"Lecture des informations sur le dernier dataset")
        df_latest_dataset = pd.read_csv(os.path.join(
            latest_dataset_info["Chemin du Dataset"], "metadata.csv"))
    else:
        latest_dataset_info = None
        # utils_data.initialize_logging_file("KAGGLE")
        # dataset_logging_path = None
        df_latest_dataset = pd.DataFrame()

    # Parcours des fichiers dans le dataset
    data = process_directory(
        dataset_path, ".", date_ajout, dataset_source, dataset_uid, df_latest_dataset)

    # Ajout des données au DataFrame
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    # Sauvegarde du DataFrame dans le fichier metadata.csv
    df.to_csv(metadata_path, index=False)

    # Log des informations du dataset
    log_dataset_info(df, dataset_path, dataset_logging_path,
                     dataset_uid, dataset_source, date_ajout)

# FONCTIONS UTILES


def process_directory(path, subdir, date_ajout, dataset_source, dataset_uid, df_latest_dataset):
    """
    Parcours un répertoire donné et traite les fichiers qu'il contient.

    :param path: Le chemin actuel du répertoire.
    :param subdir: Le sous-répertoire courant.
    :param date_ajout: La date d'ajout des fichiers.
    :param dataset_source: La source du dataset.
    :param dataset_uid: L'UID du dataset.
    :param df_latest_dataset: Le dataframe du dernier dataset pour comparaison.

    :return: Une liste de dictionnaires représentant les fichiers traités.
    """
    data = []
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            logger.debug(f"Traitement du sous-répertoire {item}")
            data += process_directory(full_path, os.path.join(subdir, item),
                                      date_ajout, dataset_source, dataset_uid, df_latest_dataset)
        else:
            if item.lower() == ".ds_store":
                continue
            ligne = create_metadata_entry(
                full_path, subdir, item, date_ajout, dataset_source, dataset_uid, df_latest_dataset)
            data.append(ligne)
    return data


def create_metadata_entry(file_path, subdir, filename, date_ajout, dataset_source, dataset_uid, df_latest_dataset):
    """
    Crée une entrée de metadata pour un fichier donné.

    :param file_path: Le chemin complet du fichier.
    :param subdir: Le sous-répertoire contenant le fichier.
    :param filename: Le nom du fichier.
    :param date_ajout: La date d'ajout du fichier.
    :param dataset_source: La source du dataset.
    :param dataset_uid: L'UID du dataset.
    :param df_latest_dataset: Le dataframe du dernier dataset pour comparaison.

    :return: Un dictionnaire représentant les metadata du fichier.
    """
    if subdir == ".":
        label = "N/A"
    else:
        label = os.path.basename(os.path.dirname(subdir))
    ligne = {
        "UID": str(uuid.uuid4()),
        "Dataset SOURCE": dataset_source,
        "Dataset SOURCE UID": dataset_uid,
        "Sous-répertoire SOURCE": subdir,
        "Classe": label,
        "Sous-répertoire CIBLE": utils_data.remove_space_from_foldername(label),
        "Nom de fichier": filename,
        "Date d'ajout": date_ajout,
        "md5": utils_data.calcul_md5(file_path),
        "Taille": utils_data.get_file_size(file_path),
        "Taille formattée": utils_data.convert_size(utils_data.get_file_size(file_path)),
        "Type": determine_type(subdir),
        "Format": utils_data.get_file_extension(filename),
        "Ignored": determine_ignore_status(subdir, filename)
    }

    if df_latest_dataset.empty:
        ligne["Status"] = "ADDED"
    else:
        md5_value = ligne["md5"]
        if md5_value in df_latest_dataset["md5"].values:
            dataset_source_uid_value = df_latest_dataset.loc[df_latest_dataset["md5"]
                                                             == md5_value, "Dataset SOURCE UID"].values[0]
            ligne["Dataset SOURCE UID"] = dataset_source_uid_value
            ligne["Status"] = "UNCHANGED"
        else:
            ligne["Status"] = "ADDED"

    return ligne


def determine_type(subdir):
    """
    Détermine le type de fichier en fonction du sous-répertoire.

    :param subdir: Le sous-répertoire contenant le fichier.

    :return: "Image" si le sous-répertoire est 'images', "Mask" s'il est 'masks', "Autre" sinon.
    """
    subdir_lower = subdir.lower()
    if 'images' in subdir_lower:
        return "Image"
    elif 'masks' in subdir_lower:
        return "Mask"
    else:
        return "Autre"


def determine_ignore_status(subdir, filename):
    """
    Détermine si un fichier doit être ignoré.

    :param subdir: Le sous-répertoire contenant le fichier.
    :param filename: Le nom du fichier.

    :return: True si le fichier doit être ignoré, sinon False.
    """
    if "lung" in subdir.lower() or subdir.lower().endswith("masks"):
        return True
    return subdir == "." or filename.lower().endswith(".unknown")


def log_dataset_info(df, dataset_path, dataset_logging_path, dataset_uid, dataset_source, date_ajout):
    """
    Log les informations générales du dataset.

    :param df: Le DataFrame contenant les metadata.
    :param dataset_path: Le chemin du dataset.
    :param dataset_logging_path: Le chemin du fichier de log du dataset.
    :param dataset_uid: L'UID du dataset.
    :param dataset_source: La source du dataset.
    :param date_ajout: La date d'ajout du dataset.
    """

    df_image_utilisable = df.loc[(df['Ignored'] == False)
                                 & (df['Type'] == 'Image')]

    nb_image_par_classe = df_image_utilisable.groupby(
        'Classe').size().to_dict()

    if os.path.exists(dataset_logging_path):
        df_data_logging = pd.read_csv(dataset_logging_path)
        latest_dataset_info = utils_data.get_latest_dataset_info(
            "KAGGLE")
        latest_dataset_version = latest_dataset_info["Dataset Version"]
        latest_dateset_uid = latest_dataset_info["UID"]
        dataset_version = utils_data.increment_version(
            str(latest_dataset_version))
    else:
        latest_dateset_uid = 0
        df_data_logging, dataset_logging_path = utils_data.initialize_logging_file(
            "KAGGLE")
        dataset_version = utils_data.increment_version(
            "0.0", increment_type="majeur")

    dataset_path_versioned = f"{dataset_path}-{dataset_version}"
    os.rename(dataset_path, dataset_path_versioned)

    info_dataset = {
        "UID": dataset_uid,
        "Dataset SOURCE": dataset_source,
        "Dataset Name": os.path.basename(dataset_path_versioned),
        "Dataset Version": dataset_version,
        "Date d'ajout": date_ajout,
        "Description": "Dataset KAGGLE",
        "Chemin du Dataset": dataset_path_versioned,
        "Nombre total de fichier": df.shape[0],
        "Nombre total d'images": df[df['Type'] == "Image"].shape[0],
        "Nombre d'image utilisable": df_image_utilisable.shape[0],
        "Nombre d'image utilisable par classe": str(nb_image_par_classe),
        "Taille totale": df["Taille"].sum(),
        "Taille totale formattée": utils_data.convert_size(df["Taille"].sum()),
        "Taille utilisable": df_image_utilisable["Taille"].sum(),
        "Taille utilisable formattée": utils_data.convert_size(df_image_utilisable["Taille"].sum()),
        "UID Copie du Dataset": latest_dateset_uid,
        "Disponible": True
    }

    df_data_logging = pd.concat(
        [df_data_logging, pd.DataFrame([info_dataset])], ignore_index=True)
    df_data_logging.to_csv(dataset_logging_path, index=False)


# Fonction MAIN qui exécute le téléchargement, le versionning et le logging

def download_dataset_main():
    """
    Fonction principale pour le téléchargement du dataset qui effectue trois actions principales :
    - Téléchargement du dataset à partir de l'API Kaggle.
    - Construction d'un fichier metadata.csv pour le dataset téléchargé.
    - Logging des informations du dataset brut dans un fichier de log.
    """
    logger.debug("---------download_dataset.py----Fonction main()---------")

    # Récupération des informations de configuration pour l'URL du dataset et le chemin de destination
    logger.info(
        "Récupération des variables de configuration pour l'URL du dataset et le chemin de dépôt des données brutes")
    url = dataset_info["KAGGLE_dataset_url"]
    destination = os.path.join(
        init_paths["main_path"],
        init_paths["KAGGLE_datasets_folder"]
    )

    logger.debug(f"URL du dataset : {url}")
    logger.debug(f"Chemin de destination : {destination}")

    # Téléchargement du dataset
    logger.debug(
        "Appel de la fonction dl_dataset_kaggle_api pour le téléchargement du dataset")
    dl_dataset_kaggle_api(url, destination)

    # Construction du fichier metadata.csv
    logger.info("Construction du fichier metadata.csv")
    build_dataset_kaggle_metadata(os.path.join(
        destination, dataset_info["KAGGLE_dataset_prefix"]))

    logger.info(
        "Téléchargement et traitement du dataset terminés avec succès")


if __name__ == "__main__":
    download_dataset_main()
