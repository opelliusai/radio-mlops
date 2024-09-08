'''
Créé le 06/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fonctions utiles pour les données:
-- Correspondance des labels avec des valeurs numériques
-- chargement d'un modèle
-- sauvegarde/chargement des résultats d'un modèle (historique, training plots)

'''
from collections import defaultdict
import math
import os
import shutil
import random
from datetime import datetime
import os
import hashlib
import uuid
from src.config.run_config import init_paths, dataset_info, model_info, current_dataset_label_correspondance
import csv
from src.config.log_config import setup_logging
import pandas as pd
logger = setup_logging("UTILS_DATA")


def label_to_numeric(labels, correspondance):
    '''
    Convertit les labels en valeurs numériques en utilisant une correspondance donnée.
    '''
    logger.debug(
        f"---------label_to_numeric(labels={labels}, correspondance={correspondance})")
    res = [correspondance[label] for label in labels]
    logger.debug(f"res={res}")
    return res


def numeric_to_label(numbers, correspondance):
    '''
    Convertit les valeurs numériques en labels en utilisant une correspondance donnée.
    '''
    logger.debug(
        f"---------numeric_to_label(numbers={numbers}, correspondance={correspondance})")
    res = [correspondance[number] for number in numbers]
    logger.debug(f"res={res}")
    return res


def generate_numeric_correspondance(labels):
    '''
    Génère une correspondance entre les labels et des valeurs numériques uniques.
    '''
    logger.debug(f"---------generate_numeric_correspondance(labels={labels})")
    unique_labels = list(set(labels))
    logger.debug(f"Unique labels {unique_labels}")
    correspondance = {label: i for i, label in enumerate(unique_labels)}
    logger.debug(f"Correspondance {correspondance}")
    return correspondance


def invert_dict(d):
    return {v: k for k, v in d.items()}


def generer_metadata_json(dataset_path, dataset_name, description, version):
    '''
    Parcourt un dataset et calcule le nombre d'images par répertoire pour créer le dataset
    :param: dataset_path
    :return json_dict à écrire dans un fichier type metadata.json
    '''
    current_date = datetime.now()
    # Format the date as YYYY-MM-DD
    creation_date = current_date.strftime("%Y-%m-%d")
    json_dict = {}
    json_dict["dataset_name"] = dataset_name
    json_dict["description"] = description
    json_dict["version"] = version
    json_dict["creation_date"] = creation_date
    json_dict["size"] = get_size(dataset_path)

    # Calcul du nombre de classes basé sur le nombre de sous-répertoires
    # Calcul du nombre de sample par classe basé sur le

    num_classes = 0
    nb_sample_per_class = {}
    class_types = {}

    for subfolder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder)
        if os.path.isdir(subfolder_path):
            num_classes += 1
            nb_sample_per_class[subfolder] = len(os.listdir(subfolder_path))
    class_types = list(nb_sample_per_class.keys())
    # Mise à jour
    json_dict["num_classes"] = num_classes
    json_dict["nb_samples_per_class"] = nb_sample_per_class
    json_dict["classes_types"] = class_types
    json_dict["last_modification"] = creation_date
    logger.debug(f"json dict {json_dict}")

    return json_dict


def get_size(start_path='.'):
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return convert_size(total_size)


def get_file_size(file_path):
    try:
        return os.path.getsize(file_path)
    except FileNotFoundError as e:
        logger.error(f"Fichier n'existe pas {file_path}")
        return None


def get_file_extension(file_name):
    return file_name.split(".")[-1]


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def remove_space_from_foldername(folder_name):
    """
    Fonction qui harmonise les noms des répertoires en supprimant les espaces remplacés par _
    A terme, on peut aussi mettre en minuscule les noms de répertoires
    Le nouveau nom de répertoire servira aussi de nom de classe
    """
    return folder_name.replace(" ", "_")


def calcul_md5(chemin_fichier):
    """
    Calcule le hash MD5 d'un fichier donné.
    """
    hash_md5 = hashlib.md5()

    with open(chemin_fichier, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    res = hash_md5.hexdigest()
    logger.debug(f"Hash md5 pour le fichier {chemin_fichier}: {res}")

    return res


def increment_version(version, increment_type="mineur"):
    """
    Incrémente une version au format 'x.y'.

    :param version: Version actuelle sous forme de chaîne de caractères, ex: '0.0'.
    :param increment_type: Type d'incrément, 'mineur' ou 'majeur'.
    :return: Nouvelle version sous forme de chaîne de caractères.
    """
    # Séparer les parties majeure et mineure
    major, minor = map(int, version.split('.'))

    if increment_type.lower() == "mineur":
        minor += 1
    elif increment_type.lower() == "majeur":
        major += 1
        minor = 0
    else:
        raise ValueError(
            "Type d'incrément invalide. Utilisez 'mineur' ou 'majeur'.")
    new_version = f"{major}.{minor}"
    logger.debug(
        f"Version {increment_type} pour la version actuelle {version} : {new_version}")

    return new_version


def get_classes():
    logger.debug("-------------get_classes()--------------")
    logger.debug(f"Return {list(current_dataset_label_correspondance.keys())}")
    return list(current_dataset_label_correspondance.keys())


def get_datasets():
    logger.debug("-------------get_datasets()--------------")
    datasets_folder = os.path.join(
        init_paths["main_path"], init_paths["REF_datasets_folder"])
    logger.debug("Parcours des sous-répertoires du répertoire des dataset et prise en compte uniquement des répertoires ayant comme préfixe celui indiqué dans dataset_info")
    nb_datasets = 0
    liste_dataset_names = []
    for subfolder in os.listdir(datasets_folder):
        if subfolder.startswith(dataset_info["REF_dataset_prefix"]):
            nb_datasets += 1
            dataset = {}
            dataset["dataset_name"] = subfolder
            dataset["dataset_full_path"] = os.path.join(
                datasets_folder, subfolder)
            liste_dataset_names.append(dataset)
        else:
            logger.debug(
                f"Le sous-répertoire {subfolder} ne commence pas par {dataset_info['dataset_prefix']} et n'est pas pris en compte")
    logger.debug(f"Nombre de datasets trouvés: {nb_datasets}")
    return liste_dataset_names


def get_prod_datasets():
    logger.debug("-------------get_prod_datasets()--------------")
    datasets_folder = os.path.join(
        init_paths["main_path"], init_paths["prod_datasets_folder"])
    logger.debug("Parcours des sous-répertoires du répertoire des dataset et prise en compte uniquement des répertoires ayant comme préfixe celui indiqué dans dataset_info")
    nb_datasets = 0
    liste_dataset_names = []
    for subfolder in os.listdir(datasets_folder):
        if subfolder.startswith(dataset_info["prod_dataset_prefix"]):
            nb_datasets += 1
            dataset = {}
            dataset["dataset_name"] = subfolder
            dataset["dataset_full_path"] = os.path.join(
                datasets_folder, subfolder)
            liste_dataset_names.append(dataset)
        else:
            logger.debug(
                f"Le sous-répertoire {subfolder} ne commence pas par {dataset_info['dataset_prefix']} et n'est pas pris en compte")
    logger.debug(f"Nombre de datasets trouvés: {nb_datasets}")
    return liste_dataset_names


def initialize_dataset(dataset_path, type="REF"):
    if type.lower() == "ref":
        if not os.path.exists(dataset_path):
            logger.debug(f"Création du répertoire {dataset_path}")
            os.makedirs(dataset_path)
        logger.debug(
            f"Création des sous-répertoires relatifs aux labels actuels")
        classes = get_classes()
        logger.debug(f"Classes : {classes}")
        for classe in classes:
            rep = remove_space_from_foldername(classe)
            logger.debug(
                f"Création du répertoire {os.path.join(dataset_path,rep)}")
            os.makedirs(os.path.join(dataset_path, rep), exist_ok=True)
        logger.debug(
            f"Création du répertoire {os.path.join(dataset_path, 'UNLABELED')}")
        os.makedirs(os.path.join(dataset_path, "UNLABELED"), exist_ok=True)
        # Initialisation du metadata.csv
        initialize_metadata_file(dataset_path, "REF")
        return dataset_path

    elif type.lower() == "prod":
        logger.debug(
            f"-----------init_prod_dataset(dataset_path={dataset_path})-----------")
        # Création du répertoire du dataset de production
        if not os.path.exists(dataset_path):
            logger.debug(f"Création du répertoire {dataset_path}")
            os.makedirs(dataset_path)
        logger.debug(
            f"Création des sous-répertoires relatifs aux labels actuels")
        classes = get_classes()

        logger.debug(f"Classes : {classes}")
        for classe in classes:
            rep = remove_space_from_foldername(classe)
            logger.debug(
                f"Création du répertoire {os.path.join(dataset_path,rep)}")
            os.makedirs(os.path.join(dataset_path, rep), exist_ok=True)
        logger.debug(
            f"Création du répertoire {os.path.join(dataset_path, 'UNLABELED')}")
        os.makedirs(os.path.join(dataset_path, "UNLABELED"), exist_ok=True)

        # Initialisation du metadata.csv
        initialize_metadata_file(dataset_path, "PROD")
        return dataset_path
    else:
        logger.debug(f"Type de DATASET {type} non reconnu")
        return None


def update_metadata_prod(dataset_path, dataset_uid, pred_id, filename, taille_image, label):
    logger.debug(
        f"-----------update_metadata_prod(dataset_path={dataset_path},dataset_uid={dataset_uid},label={label},filename={filename},taille_image={taille_image},pred_id={pred_id})-----------")
    # Ajout d'une ligne au fichier metadata.csv
    rep = remove_space_from_foldername(label)
    md5 = calcul_md5(os.path.join(dataset_path, rep, filename))
    logger.debug(
        f"Ajout d'une ligne au fichier metadata.csv pour le fichier {filename} avec le md5 {md5}")
    with open(os.path.join(dataset_path, "metadata.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        UID = uuid.uuid4()
        writer.writerow(
            [UID, dataset_uid, rep, rep, label, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), md5, taille_image, convert_size(taille_image), pred_id])
    logger.debug("Ajout terminé")


def copy_directory(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copy_directory(s, d)
        else:
            shutil.copy2(s, d)


def move_directory(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            move_directory(s, d)
        else:
            shutil.move(s, d)


def move_file(src, dst):
    # Create the destination directory if it doesn't exist
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Move the file
    shutil.move(src, dst)
# Dataset history management


def log_dataset_info(dataset_name, dataset_version, desc, nb_classes, nb_images_per_class,
                     nb_total_images, dataset_path, metadata_csv_path, metadata_json_path,
                     creation_date, last_update, is_current_dataset=False, type="REF"):
    """
    # information logging:
    # - UID
    # - Nom du Dataset
    # - Version du Dataset
    # - Description
    # - Nombre de classes
    # - Nombres d'images par classe
    # - Nombre d'image total
    # - Chemin de Dataset
    # - Chemin metadata_csv
    # - Chemin metadata_json
    """
    logger.debug(f"log_dataset_info(type={type})")
    # logger.debug(f"log_dataset_info(dataset_name={dataset_name}, dataset_version={str(dataset_version)}, desc={desc}, nb_classes={str(nb_classes)}, nb_images_per_class={str(nb_images_per_class)},
    #                 nb_total_images={str(nb_total_images)}, dataset_path={dataset_path}, metadata_csv_path={metadata_csv_path}, metadata_json_path={metadata_json_path},
    #                 creation_date={creation_date}, last_update={last_update}, is_current_dataset={str(is_current_dataset)}, type={type}")
    dataset_logging_path = get_logging_path(type)
    unique_id = str(uuid.uuid4())
    dataset_infos = {
        "UID": unique_id,
        "Dataset Name": dataset_name,
        "Dataset Version": dataset_version,
        "Description": desc,
        "Nombre de classes": nb_classes,
        "Images par classe": nb_images_per_class,
        "Nombre total d'images": nb_total_images,
        "Chemin du Dataset": dataset_path,
        "Chemin Metadata CSV": metadata_csv_path,
        "Chemin Metadata JSON": metadata_json_path,
        "Date de création": creation_date,
        "Dernière mise à jour": last_update,
        "Dataset courant": is_current_dataset
    }
    # 1 - Créer un fichier de log avec les colonnes s'il n'existe pas
    file_exists = os.path.isfile(dataset_logging_path)
    with open(dataset_logging_path, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            logger.debug(f"Fichier de logging n'existe pas")
            logger.debug(f"Création du fichier de log {dataset_logging_path}")
            logger.debug(f"Ecriture de l'entête {dataset_infos.keys()}")
            writer.writerow(dataset_infos.keys())
        else:
            logger.debug(
                "Le fichier de logging existe, stockage des résultats en fin de fichier")
        logger.debug(f"Ecriture des résultats {dataset_infos.values()}")
        # Ecriture des informations sur le dataset à la fin du fichier
        writer.writerow(dataset_infos.values())
    if is_current_dataset:
        logger.debug(f"Set Dataset courant : {dataset_infos['Dataset Name']}")
        set_new_current_dataset(unique_id)
    return unique_id


def get_current_dataset(type="REF"):
    """
    Récupére le dataset tagué comme "Dataset courant" = True
    """
    logger.debug(f"---get_current_dataset(type={type})--")
    if type.lower() == "ref":
        dataset_logging_path = os.path.join(
            init_paths["main_path"], init_paths["dataset_logging_folder"], dataset_info["dataset_logging_filename"])
        logger.debug(f"ref dataset_logging_path : {dataset_logging_path}")
    elif type.lower() == "prod":
        dataset_logging_path = os.path.join(
            init_paths["main_path"], init_paths["prod_dataset_logging_folder"], dataset_info["prod_dataset_logging_filename"])
        logger.debug(f"prod dataset_logging_path : {dataset_logging_path}")
    else:
        logger.debug(f"Type de dataset {type} non reconnu")
        return None
    logger.debug(f"dataset_logging_path : {dataset_logging_path}")
    file_exists = os.path.isfile(dataset_logging_path)
    if not file_exists:
        logger.debug(f"Le fichier de logging n'existe pas")
        return None
    else:
        with open(dataset_logging_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                logger.debug(f"row {row} - {len(row)}")
                logger.debug(f"row 12 - {bool(row[12])}")
                if row[12].lower() == "true":  # "Dataset courant" = True
                    logger.debug(f"Dataset courant trouvé")
                    logger.debug(f"row du dataset trouvé - {row}")
                    return row
    return None


def get_current_dataset_version(type="REF"):
    """
    Récupére le dataset tagué comme "Dataset courant" = True
    """
    logger.debug(f"---get_current_dataset_version(type={type})--")
    if type.lower() == "ref":
        dataset_logging_path = os.path.join(
            init_paths["main_path"], init_paths["dataset_logging_folder"], dataset_info["dataset_logging_filename"])
        logger.debug(f"ref dataset_logging_path : {dataset_logging_path}")
    elif type.lower() == "prod":
        dataset_logging_path = os.path.join(
            init_paths["main_path"], init_paths["prod_dataset_logging_folder"], dataset_info["prod_dataset_logging_filename"])
        logger.debug(f"prod dataset_logging_path : {dataset_logging_path}")
    else:
        logger.debug(f"Type de dataset {type} non reconnu")
        return None
    logger.debug(f"dataset_logging_path : {dataset_logging_path}")
    file_exists = os.path.isfile(dataset_logging_path)
    if not file_exists:
        logger.debug(f"Le fichier de logging n'existe pas")
        return None
    else:
        with open(dataset_logging_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                logger.debug(f"row {row} - {len(row)}")
                logger.debug(f"row 12 - {bool(row[12])}")
                if row[12].lower() == "true":  # "Dataset courant" = True
                    logger.debug(f"Dataset courant trouvé")
                    logger.debug(f"row du dataset trouvé - {row}")
                    return row[2]
    return None


def set_new_current_dataset(new_current_dataset_ID, type="REF"):
    """
    Met la valeur du dataset courant à FALSE pour prépar
    """
    # Définition du chemin en fonction du type de données
    if type.lower() == "ref":
        dataset_logging_path = os.path.join(
            init_paths["main_path"], init_paths["dataset_logging_folder"], dataset_info["dataset_logging_filename"])
        logger.debug(f"dataset_logging_path : {dataset_logging_path}")
    elif type.lower() == "prod":
        dataset_logging_path = os.path.join(
            init_paths["main_path"], init_paths["prod_dataset_logging_folder"], dataset_info["prod_dataset_logging_filename"])
        logger.debug(f"dataset_logging_path : {dataset_logging_path}")
    else:
        logger.debug(f"Type de dataset {type} non reconnu")
        return None

    # Parcours du fichier pour unsetter le dataset courant et setter le nouveau dataset.
    file_exists = os.path.isfile(dataset_logging_path)
    if not file_exists:
        logger.debug("Le fichier dataset n'existe pas")
        return None
    else:
        df = pd.read_csv(dataset_logging_path)
        # Mise à jour du dataset de UID en True
        df.loc[df["UID"] == new_current_dataset_ID, "Dataset courant"] = True
        # Mise à jour du dataset courant en False
        df.loc[(df["UID"] != new_current_dataset_ID) & (
            df["Dataset courant"] == True), "Dataset courant"] = False
        # Enregistrement du dataset
        df.to_csv(dataset_logging_path, index=False)
        return new_current_dataset_ID


def check_logging_exists(type="REF"):
    logger.debug(f"----check_logging_exists(type={type}------")
    logging_path = get_logging_path(type)
    return os.path.exists(logging_path), logging_path


def get_latest_dataset_info(type="REF"):
    exist, filepath = check_logging_exists(type)
    if not exist:
        logger.debug("Le fichier dataset logging n'existe pas")
        return None
    logger.debug(f"get_latest_dataset_info(type= {type})")
    # Charger le fichier csv
    data = pd.read_csv(filepath)
    # Conversion en numérique
    data['version'] = pd.to_numeric(data['Dataset Version'], errors='coerce')
    # Chercher la version maximum
    latest_dataset_info = data.loc[data["Date d'ajout"].idxmax()]
    logger.debug(f"latest_dataset_info {latest_dataset_info}")
    logger.debug(
        f"latest_dataset_info CHEMIN {latest_dataset_info['Chemin du Dataset']}")
    return latest_dataset_info


def get_all_dataset_info(type="REF"):
    exist, filepath = check_logging_exists(type)
    if not exist:
        logger.debug("Le fichier dataset logging n'existe pas")
        return None
    logger.debug(f"get_all_dataset_info(type= {type})")
    # Charger le fichier csv
    all_dataset_info = pd.read_csv(filepath)

    return all_dataset_info


def get_dataset_info_by_version(version, type="REF"):
    exist, filepath = check_logging_exists(type)
    if not exist:
        logger.debug("Le fichier dataset logging n'existe pas")
        return None
    logger.debug(f"get_latest_dataset(type= {type})")
    # Charger le fichier csv
    data = pd.read_csv(filepath)
    # Conversion en numérique
    data['version'] = pd.to_numeric(data['Dataset Version'], errors='coerce')
    # Chercher la version maximum
    dataset_info = data.loc[data['Dataset Version'] == version]
    dataset_info = dataset_info.reset_index(drop=True)
    if dataset_info.empty:
        logger.debug(f"Aucun dataset trouvé avec la version {version}")
        return None
    return dataset_info.iloc[0]


def get_dataset_info_by_UID(dataset_uid, type="REF"):
    exist, filepath = check_logging_exists(type)
    if not exist:
        logger.debug("Le fichier dataset logging n'existe pas")
        return None
    logger.debug(
        f"get_dataset_info_by_dataset_name(name= {dataset_uid}, type= {type})")
    # Charger le fichier csv
    data = pd.read_csv(filepath)
    # Chercher la ligne correspondant au nom du dataset
    dataset_info = data.loc[data["UID"] == dataset_uid]
    # Supprimer l'index de la ligne de résultat
    dataset_info = dataset_info.reset_index(drop=True)
    if dataset_info.empty:
        logger.debug(f"Aucun dataset trouvé avec le nom {dataset_uid}")
        return None
    return dataset_info.iloc[0]


def get_dataset_info_by_dataset_name(dataset_name, type="REF"):
    exist, filepath = check_logging_exists(type)
    if not exist:
        logger.debug("Le fichier dataset logging n'existe pas")
        return None
    logger.debug(
        f"get_dataset_info_by_dataset_name(name= {dataset_name}, type= {type})")
    # Charger le fichier csv
    data = pd.read_csv(filepath)
    # Chercher la ligne correspondant au nom du dataset
    dataset_info = data.loc[data["Dataset Name"] == dataset_name]
    # Supprimer l'index de la ligne de résultat
    dataset_info = dataset_info.reset_index(drop=True)
    if dataset_info.empty:
        logger.debug(f"Aucun dataset trouvé avec le nom {dataset_name}")
        return None
    return dataset_info.iloc[0]


def DELETE_get_dataset_info_by_dataset_name(dataset_name, type="REF"):
    exist, filepath = check_logging_exists(type)
    if not exist:
        logger.debug("Le fichier dataset logging n'existe pas")
        return None
    logger.debug(
        f"get_dataset_info_by_dataset_name(dataset_name={dataset_name}, type={type})")
    # Charger le fichier csv
    data = pd.read_csv(filepath)
    logger.debug(f"Data {data['Dataset Name']}")
    # Trouver la ligne correspondante
    dataset_info = data.loc[data['Dataset Name'] == dataset_name]
    # Réinitialiser l'index du résultat
    dataset_info = dataset_info.reset_index(drop=True)
    logger.debug(f"dataset_info {dataset_info.to_string(index=False)}")
    return dataset_info


def get_logging_path(type="REF"):
    logger.debug(f"----get_logging_path(type={type})")
    if type.lower() == "ref":
        logging_path = os.path.join(
            init_paths["main_path"], init_paths["dataset_logging_folder"], dataset_info["REF_dataset_logging_filename"])
        logger.debug(f"logging_path : {logging_path}")
    elif type.lower() == "prod":
        logging_path = os.path.join(
            init_paths["main_path"], init_paths["dataset_logging_folder"], dataset_info["PROD_dataset_logging_filename"])
        logger.debug(f"logging_path : {logging_path}")
    elif type.lower() == "kaggle":
        logging_path = os.path.join(
            init_paths["main_path"], init_paths["dataset_logging_folder"], dataset_info["KAGGLE_dataset_logging_filename"])
        logger.debug(f"logging_path : {logging_path}")
    elif type.lower() == "pred":
        logging_path = os.path.join(
            init_paths["main_path"], init_paths["PRED_logging_folder"], model_info["PRED_logging_filename"])
        logger.debug(f"logging_path : {logging_path}")
    elif type.lower() == "drift":
        logging_path = os.path.join(
            init_paths["main_path"], init_paths["model_drift_folder"], model_info["MODEL_DRIFT_logging_filename"])
        logger.debug(f"logging_path : {logging_path}")
    else:
        logger.debug(f"Type de LOGGING {type} non reconnu")
        return None
    return logging_path


def initialize_logging_file(type="REF"):
    logger.debug(f"----get_logging_path(type={type})")
    logging_path = get_logging_path(type)
    if type.lower() == "ref":
        log_columns = ["UID",
                       "Dataset Update SOURCE",
                       "Dataset Name",
                       "Dataset Version",
                       "Dataset Base UID",
                       "Date d'ajout",
                       "Description",
                       "Chemin du Dataset",
                       "Nombre total de fichiers",
                       "Nombre d'images par classe",
                       "Taille totale",
                       "Taille totale formattée",
                       "Disponible"]
    elif type.lower() == "prod":
        log_columns = ["UID",
                       "Dataset Name",
                       "Dataset Version",
                       "Date d'ajout",
                       "Description",
                       "Chemin du Dataset",
                       "Nombre total de fichiers",
                       "Nombre d'images par classe",
                       "Taille totale",
                       "Taille totale formattée",
                       "Disponible"
                       ]
    elif type.lower() == "kaggle":
        log_columns = ["UID",
                       "Dataset SOURCE",
                       "Dataset Name",
                       "Dataset Version",
                       "Date d'ajout",
                       "Description",
                       "Chemin du Dataset",
                       "Nombre total de fichier",
                       "Nombre total d'images",
                       "Nombre d'image utilisable",
                       "Nombre d'image utilisable par classe",
                       "Taille totale",
                       "Taille totale formattée",
                       "Taille utilisable",
                       "Taille utilisable formattée",
                       "Disponible"]
    elif type.lower() == "pred":
        log_columns = ["UID",
                       "Nom du modèle",
                       "Chemin de l\'image",
                       "Taille",
                       "'Taille formattée",
                       "md5",
                       "Prédiction",
                       "Indice de confiance",
                       "Temps de prédiction",
                       "Date de prédiction",
                       "Prédiction validée",
                       "Perf Prédiction",
                       "Username"]
    elif type.lower() == "drift":
        log_columns = ["UID",
                       "Nom du modèle",
                       "New Mean",
                       "Original Mean",
                       "Original",
                       "New STD",
                       "Original STD",
                       "Mean Diff",
                       "STD Diff",
                       "Drift",
                       "Date de calcul",
                       "Temps de calcul"]
    else:
        logger.debug(f"Type de LOGGING {type} non reconnu")
        return None
    df_logging = pd.DataFrame(columns=log_columns)
    df_logging.to_csv(logging_path, index=False)
    return df_logging, logging_path


def initialize_metadata_file(dataset_path, type="REF"):
    logger.debug(
        f"----initialize_metadata_file(dataset_path={dataset_path}, type={type})")
    # Construction du chemin du metadata.csv, situé à la racine du dataset
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    if type.lower() == "ref":
        meta_columns = ["UID",
                        "Dataset SOURCE",
                        "Dataset SOURCE UID",
                        "Sous-répertoire",
                        "Classe",
                        "Nom de fichier",
                        "Date d'ajout",
                        "md5",
                        "Taille",
                        "Taille formattée",
                        "Status"]
    elif type.lower() == "prod":
        meta_columns = ["UID",
                        "Dataset UID",
                        "Sous-répertoire SOURCE",
                        "Sous-répertoire CIBLE",
                        "Classe",
                        "Nom de fichier",
                        "Date d'ajout",
                        "md5",
                        "Taille",
                        "Taille formattée",
                        "Pred ID"
                        ]
    elif type.lower() == "kaggle":
        meta_columns = ["UID",
                        "Dataset SOURCE",
                        "Dataset SOURCE UID",
                        "Sous-répertoire SOURCE",
                        "Classe",
                        "Sous-répertoire CIBLE",
                        "Nom de fichier",
                        "Date d'ajout",
                        "md5",
                        "Taille",
                        "Taille formattée",
                        "Type",
                        "Format",
                        "Ignored",
                        "Status"]
    else:
        logger.debug(f"Type de LOGGING {type} non reconnu")
        return None
    df_dataset = pd.DataFrame(columns=meta_columns)
    df_dataset.to_csv(metadata_path, index=False)

    return df_dataset, metadata_path


def update_or_add_dataset_info_prod(csv_path, dataset_info):
    """
    Met à jour ou ajoute des informations de dataset dans le fichier CSV.

    Args:
    - csv_path (str): Chemin vers le fichier CSV de log des datasets.
    - dataset_info (dict): Informations sur le dataset à mettre à jour ou ajouter.

    dataset_info doit contenir les clés suivantes :
    UID, Dataset Name, Dataset Version, Date d'ajout, Description, Chemin du Dataset, 
    Nombre total de fichiers, Nombre d'images par classe, Taille totale, Taille totale formattée, Disponible
    """

    # Lire le CSV et stocker les lignes
    updated = False
    csv_lines = []

    if os.path.exists(csv_path):
        with open(csv_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["UID"] == dataset_info["UID"]:
                    # Mise à jour des informations pour le dataset existant
                    row["Nombre total de fichiers"] = dataset_info["Nombre total de fichiers"]
                    row["Nombre d'images par classe"] = dataset_info["Nombre d'images par classe"]
                    row["Taille totale"] = dataset_info["Taille totale"]
                    row["Taille totale formattée"] = dataset_info["Taille totale formattée"]
                    updated = True
                csv_lines.append(row)

    if not updated:
        # Si le dataset n'existe pas, on ajoute une nouvelle ligne
        csv_lines.append(dataset_info)

    # Écrire les données mises à jour dans le CSV
    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = ["UID", "Dataset Name", "Dataset Version", "Date d'ajout", "Description", "Chemin du Dataset",
                      "Nombre total de fichiers", "Nombre d'images par classe", "Taille totale", "Taille totale formattée", "Disponible"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(csv_lines)


def get_total_file_count(dataset_path):
    """
    Retourne le nombre total de fichiers dans le répertoire du dataset.

    Args:
    - dataset_path (str): Chemin vers le répertoire du dataset.

    Returns:
    - int: Nombre total de fichiers dans le dataset.
    """
    total_files = 0
    for root, dirs, files in os.walk(dataset_path):
        total_files += len(files)
    return total_files


def get_image_count_per_class(dataset_path):
    """
    Retourne un dictionnaire avec le nombre d'images par classe (par dossier).

    Args:
    - dataset_path (str): Chemin vers le répertoire du dataset.

    Returns:
    - dict: Dictionnaire où les clés sont les noms des classes (dossiers) 
            et les valeurs sont le nombre d'images dans chaque classe.
    """
    image_count = defaultdict(int)

    for root, dirs, files in os.walk(dataset_path):
        class_name = os.path.basename(root)
        image_count[class_name] += len([file for file in files if file.lower(
        ).endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])

    return dict(image_count)


def get_total_size(dataset_path):
    """
    Retourne la taille totale de tous les fichiers dans le répertoire du dataset.

    Args:
    - dataset_path (str): Chemin vers le répertoire du dataset.

    Returns:
    - int: Taille totale des fichiers en octets.
    """
    total_size = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    return total_size


def get_unlabeled_image():
    """
    Fournir les informations sur une image non labellisée pour contribution
    """
    current_prod_dataset = get_latest_dataset_info("PROD")
    if current_prod_dataset is None:
        logger.error("Aucun Dataset de PROD trouvé.")
        return "Aucun Dataset de PROD trouvé.", None, None, None, None
    else:
        dataset_path = current_prod_dataset["Chemin du Dataset"]
        logger.debug(f"dataset_path {dataset_path}")
        metadata_path = os.path.join(dataset_path, "metadata.csv")
        logger.debug(f"metadata_path {metadata_path}")
        # Lecture du dataset pour trouver le chemin de l'image
        df = pd.read_csv(metadata_path)
        # Filtrer les images non labellisées
        df = df[df["Classe"] == "UNLABELED"]
        # Sélectionner une image au hasard
        if df is None:
            return "Aucune image non labellisée trouvée.", None, None, None, None
        else:
            # Sélectionner une image au hasard
            random_image = df.sample(1)
            # Récupérer son UID
            image_uid = random_image["UID"].values[0]
            # Récupérer le chemin de l'image
            image_path = random_image["Sous-répertoire CIBLE"].values[0]
            # Récupérer le nom de l'image
            image_name = random_image["Nom de fichier"].values[0]
            pred_id = random_image["Pred ID"].values[0]
            # Construire le chemin complet
            logger.debug(f"image_uid {image_uid}")
            logger.debug(f"image_name {image_name}")
            logger.debug(f"image_path   {image_path}")
            logger.debug(f"pred_id   {pred_id}")

            full_image_path = os.path.join(
                dataset_path, image_path, image_name)

            logger.debug(f"full_image_path {full_image_path}")
            return "OK", image_uid, image_name, full_image_path, pred_id
