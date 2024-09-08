'''
Créé le 07/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Stratégie de mise à jour des datasets:
    - Mise à jour des données de référence: Avec des données KAGGLE ou PROD (toujours basé sur le md5)
    - Ajout d'une image dans le Dataset de PROD (Image labellisée Après prédiction)

'''

# IMPORTS
# Imports externes
from datetime import datetime
import os
import pandas as pd
import uuid
import shutil
# Imports internes
from src.utils import utils_data, utils_models
from src.config.run_config import init_paths, dataset_info
from src.config.log_config import setup_logging
# Redirection vers le fichier de log radio-mlops_datasets.log
logger = setup_logging("datasets")

# FONCTIONS PRINCIPALES
# 1 - Mise à jour du Dataset de REFERENCE (principal)


def update_dataset_ref(update_dataset_path, source_type="KAGGLE", base_dataset_id=None):
    """
    Met à jour la référence d'un dataset en fonction du type de source et incrémente la version des données
    :param update_dataset_path: Le chemin du dataset à mettre à jour. Si non fourni, le chemin est déterminé en fonction de la dernière version loguée.
    :type update_dataset_path: str

    :param source_type: Le type de source du dataset (par défaut "KAGGLE"). Peut être "KAGGLE" ou "PROD".
    :type source_type: str

    :param base_dataset_id: L'UID du dataset de base à partir duquel la nouvelle version sera construite. Peut être None si une nouvelle référence doit être créée.
    :type base_dataset_id: str

    :raises Exception: En cas d'erreur non gérée.
    """
    logger.debug(
        f"Début de la mise à jour du dataset: {update_dataset_path}, source: {source_type}, base UID: {base_dataset_id}")

    if not update_dataset_path:
        logger.debug("update_dataset_path = None")
        # Construction du chemin KAGGLE basé sur la dernière version logguée
        latest_dataset_info = utils_data.get_latest_dataset_info(source_type)
        update_dataset_path = latest_dataset_info["Chemin du Dataset"]
        update_dataset_id = latest_dataset_info["UID"]
    else:
        update_dataset_name = os.path.basename(update_dataset_path)
        if source_type.lower() == "kaggle":
            prefix = dataset_info["KAGGLE_dataset_prefix"]
        elif source_type.lower() == "prod":
            prefix = dataset_info["PROD_dataset_prefix"]
        update_dataset_id = update_dataset_name.split(prefix)[-1]
    logger.debug(f"update_dataset_path = {update_dataset_path}")

    # Vérification des paramètres
    validate_input_parameters(update_dataset_path, source_type)

    new_dataset_uid = str(uuid.uuid4())
    ref_dataset_logging_exists, ref_dataset_logging_path = utils_data.check_logging_exists(
        "REF")

    if not ref_dataset_logging_exists:
        new_dataset_version = "1.0"
        new_ref_dataset_path = os.path.join(
            init_paths["main_path"], init_paths["REF_datasets_folder"], dataset_info["REF_dataset_prefix"]+new_dataset_version)
        utils_data.initialize_dataset(new_ref_dataset_path, "REF")
        base_dataset_id = "0.0"
        utils_data.initialize_logging_file("REF")

    else:
        base_dataset_id, base_dataset_path, new_dataset_version = prepare_base_dataset(
            base_dataset_id)
        new_ref_dataset_path = os.path.join(
            init_paths["main_path"], init_paths["REF_datasets_folder"], dataset_info["REF_dataset_prefix"]+new_dataset_version)
        utils_data.initialize_dataset(new_ref_dataset_path, "REF")
        # Copie du contenu du dernier dataset dans le dataset courant
        utils_data.copy_directory(base_dataset_path, new_ref_dataset_path)

    # Mise à jour du metadata.csv pour mettre tous les status a UNCHANGED maintenant qu'ils ont été copié du dataset précédent
    # Ne fera rien si le fichier est vide
    logger.debug(f"new_ref_dataset_path {new_ref_dataset_path}")
    df_new_metadata_csv = update_metadata_csv(new_ref_dataset_path)
    logger.debug(f"df_new_metadata_csv {df_new_metadata_csv}")
    df_update_dataset_metadata = load_and_filter_update_metadata(
        update_dataset_path, source_type)

    logger.debug(f"new_ref_dataset_path {new_ref_dataset_path}")
    copy_new_data_and_update_metadata(
        df_new_metadata_csv, df_update_dataset_metadata, new_ref_dataset_path, update_dataset_path, source_type)

    # Mise à jour du df_new_metadata_csv avec la nouvelle version du fichier metadata.csv avant logging
    df_new_metadata_csv = pd.read_csv(os.path.join(
        new_ref_dataset_path, "metadata.csv"))
    log_new_dataset(new_dataset_uid, source_type, new_ref_dataset_path, new_dataset_version,
                    base_dataset_id, update_dataset_id, df_new_metadata_csv, ref_dataset_logging_path)

    new_dataset_infos = utils_data.get_dataset_info_by_UID(
        new_dataset_uid, "REF")
    return new_dataset_infos
# 2 - Ajout d'une image labellisée dans le Dataset de PROD (Image labellisée Après prédiction)


def add_one_image(image_path, pred_id, label="UNLABELED"):
    """
    Ajoute une nouvelle image à un ensemble de données existant et met à jour les métadonnées associées.
    Ces données sont stockées dans le DATASET de PROD
    :param image_path: Le chemin de l'image à ajouter.
    :type image_path: str

    :param label: Le label associé à l'image. Par défaut "UNLABELED".
    :type label: str

    :param pred_id: Id de prediction associé à l'ajout,

    :raises Exception: En cas d'erreur non gérée.

    :return: "OK" si l'opération est réussie.
    :rtype: str
    """
    logger.debug(
        f"add_one_image(image_path={image_path},pred_id = {pred_id},label={label})")

    current_prod_dataset = utils_data.get_latest_dataset_info("PROD")
    # Créer un nouveau répertoire s'il n'existe pas
    # On fait appel à une fonction qui va initialiser les répertoires et le fichier csv
    prod_logging_path = utils_data.get_logging_path("PROD")
    if current_prod_dataset is None:
        new_version = dataset_info["PROD_dataset_prefix"]+"1.0"
        utils_data.initialize_logging_file("PROD")
        dataset_path = os.path.join(
            init_paths["main_path"], init_paths["PROD_datasets_folder"], new_version)
        utils_data.initialize_dataset(dataset_path, "PROD")
        logger.debug("Le répertoire dataset prod n'existe pas et sera créé")
        dataset_uid = uuid.uuid4()
        dataset_name = new_version
        dataset_version = "1.0"
        dataset_description = "Dataset PROD"

    else:
        dataset_path = current_prod_dataset["Chemin du Dataset"]
        logger.debug("Le répertoire dataset existe")
        dataset_uid = current_prod_dataset["UID"]
        dataset_name = current_prod_dataset["Dataset Name"]
        dataset_version = current_prod_dataset["Dataset Version"]
        dataset_description = current_prod_dataset["Description"]

    # Copie du fichier selon le label indiqué
    # Copie de l'image dans un répertoire correspondant à son label
    rep = utils_data.remove_space_from_foldername(label)
    shutil.copy(image_path, os.path.join(dataset_path, rep))
    logger.debug(
        f"{image_path} copiée dans {os.path.join(dataset_path, rep)}")
    # Mise à jour du fichier metadata.csv
    taille_image = utils_data.get_file_size(image_path)
    logger.debug(f"Mise à jour du fichier metadata.csv")
    logger.debug(
        f"dataset_path={dataset_path}, dataset_uid={dataset_uid}, label={label}, os.path.basename(image_path)={os.path.basename(image_path)}, taille_image={taille_image}, pred_id={pred_id})")
    utils_data.update_metadata_prod(
        dataset_path, dataset_uid, pred_id, os.path.basename(image_path), taille_image, label)

    dataset_infos = {
        "UID": dataset_uid,
        "Dataset Name": dataset_name,
        "Dataset Version": dataset_version,
        "Date d'ajout": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Description": dataset_description,
        "Chemin du Dataset": dataset_path,
        "Nombre total de fichiers": utils_data.get_total_file_count(dataset_path),
        "Nombre d'images par classe": utils_data.get_image_count_per_class(dataset_path),
        "Taille totale": utils_data.get_total_size(dataset_path),
        "Taille totale formattée": utils_data.convert_size(utils_data.get_total_size(dataset_path)),
        "Disponible": True  # ou un autre statut selon votre besoin
    }

    # Chemin vers le fichier CSV de log

    # Mise à jour ou ajout du dataset dans le fichier CSV
    utils_data.update_or_add_dataset_info_prod(
        prod_logging_path, dataset_infos)

    return "OK"

# Mise à jour du label d'une image


def update_image(image_uid, pred_id, label):
    """
    Mise à jour du label et met à jour les métadonnées associées.
    Ces données sont stockées dans le DATASET de PROD
    :param image_uid: Le chemin de l'image à ajouter.
    :type image_uid: str


    :param pred_id: L'ID de la prédiction'
    :type pred_id: str

    :param label: Le label proposé à l'image.
    :type label: str

    :raises Exception: En cas d'erreur non gérée.

    :return: "OK" si l'opération est réussie.
    :rtype: str
    """
    logger.debug(
        f"update_image(image_path={update_image},pred_id={pred_id},label={label})")

    current_prod_dataset = utils_data.get_latest_dataset_info("PROD")
    if current_prod_dataset is None:
        logger.error("Aucun Dataset de PROD trouvé.")
    else:
        dataset_path = current_prod_dataset["Chemin du Dataset"]
        metadata_path = os.path.join(dataset_path, "metadata.csv")
        # Lecture du dataset pour trouver le chemin de l'image
        df = pd.read_csv(metadata_path)
        image_row = df[df["UID"] == image_uid]
        if image_row.empty:
            logger.error(f"Aucune image trouvée avec l'UID {image_uid}.")
        else:
            rep = image_row["Sous-répertoire CIBLE"].values[0]
            new_rep = utils_data.remove_space_from_foldername(label)
            nom_image = image_row["Nom de fichier"].values[0]
            logger.debug(f"rep {rep}")
            logger.debug(f"new_rep {new_rep}")
            logger.debug(f"nom_image {nom_image}")
            chemin_image = os.path.join(dataset_path, rep, nom_image)
            chemin_cible = os.path.join(dataset_path, new_rep, nom_image)
            logger.debug(
                "Déplacement de l'image du répertoire {rep} à {new_rep}")
            utils_data.move_file(chemin_image, chemin_cible)
            # Mise à jour de la classe dans le fichier metadata.csv
            old_label = df.loc[df["UID"] == image_uid, "Classe"]
            logger.debug(f"Ancien Label {old_label}")
            logger.debug(f"Ancien répertoire {rep}")
            df.loc[df["UID"] == image_uid, "Classe"] = label
            df.loc[df["UID"] == image_uid, "Sous-répertoire SOURCE"] = new_rep
            df.loc[df["UID"] == image_uid, "Sous-répertoire CIBLE"] = new_rep
            new_label = df.loc[df["UID"] == image_uid, "Classe"]
            new_rep = df.loc[df["UID"] == image_uid, "Sous-répertoire SOURCE"]
            logger.debug(f"Nouveau Label {new_label}")
            logger.debug(f"Nouveau répertoire {new_rep}")
            logger.debug(f"Nouveau DataFrame {df}")
            logger.debug(
                f"Mise à jour du metadata.csv du dataset (Sans créer une nouvelle version)")
            df.to_csv(metadata_path, index=False)
            logger.debug(f"Mise à jour des logs de prediction")
            utils_models.update_log_prediction(pred_id, label)
    return "OK"

# FONCTIONS UTILES


def validate_input_parameters(update_dataset_path, source_type):
    if not isinstance(update_dataset_path, (str, bytes, os.PathLike)):
        raise TypeError(
            f"Le chemin du dataset doit être une chaîne de caractères valide. Valeur reçue: {update_dataset_path}")

    if not os.path.exists(update_dataset_path):
        raise FileNotFoundError(
            f"Le dataset à mettre à jour n'existe pas: {update_dataset_path}")

    if source_type not in ["KAGGLE", "PROD"]:
        raise ValueError(
            f"Le type de source doit être 'KAGGLE' ou 'PROD': {source_type}")


def prepare_base_dataset(base_dataset_id):
    if base_dataset_id is None:
        base_dataset_row = utils_data.get_latest_dataset_info("REF")
        base_dataset_id = base_dataset_row["UID"]
    else:
        base_dataset_row = utils_data.get_dataset_info_by_UID(
            base_dataset_id, "REF")

    base_dataset_path = base_dataset_row["Chemin du Dataset"]
    new_dataset_version = utils_data.increment_version(
        str(base_dataset_row["Dataset Version"]))
    return base_dataset_id, base_dataset_path, new_dataset_version


def create_new_dataset_directory(new_dataset_version):
    new_ref_dataset_path = os.path.join(
        init_paths["main_path"], init_paths["REF_datasets_folder"], dataset_info["REF_dataset_prefix"] + new_dataset_version)
    os.makedirs(new_ref_dataset_path, exist_ok=True)
    return new_ref_dataset_path


def update_metadata_csv(new_ref_dataset_path):
    new_metadata_csv_path = os.path.join(new_ref_dataset_path, "metadata.csv")
    df_new_metadata_csv = pd.read_csv(new_metadata_csv_path)
    df_new_metadata_csv["Status"] = "UNCHANGED"
    return df_new_metadata_csv


def load_and_filter_update_metadata(update_dataset_path, source_type):
    df_update_dataset_metadata = pd.read_csv(
        os.path.join(update_dataset_path, "metadata.csv"))
    if source_type.lower() == "kaggle":
        df_update_dataset_metadata = df_update_dataset_metadata[
            df_update_dataset_metadata["Ignored"] == False]

    classes = utils_data.get_classes()
    df_update_dataset_metadata = df_update_dataset_metadata[df_update_dataset_metadata["Classe"].isin(
        classes)]

    return df_update_dataset_metadata


def copy_new_data_and_update_metadata(df_new_metadata_csv, df_update_dataset_metadata, new_ref_dataset_path, update_dataset_path, source_type):

    logger.debug(f"update_dataset_path {update_dataset_path}")
    new_data = []
    date_ajout = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    filtered_df_new_data = df_update_dataset_metadata[~df_update_dataset_metadata["md5"].isin(
        df_new_metadata_csv["md5"])]

    logger.debug(f"filtered_df_new_data {filtered_df_new_data}")
    for _, row in filtered_df_new_data.iterrows():
        logger.debug(
            f"row['Sous-répertoire SOURCE'] {row['Sous-répertoire SOURCE']}")
        logger.debug(
            f"row['Sous-répertoire CIBLE'] {row['Sous-répertoire CIBLE']}")
        logger.debug(
            f"row['Nom de fichier'] {row['Nom de fichier']}")
        filepath = os.path.join(
            update_dataset_path, row["Sous-répertoire SOURCE"], row["Nom de fichier"])
        file_destpath = os.path.join(
            new_ref_dataset_path, row["Sous-répertoire CIBLE"], row["Nom de fichier"])
        filepath_strip = filepath.strip("'")
        file_destpath_strip = file_destpath.strip("'")
        shutil.copy(filepath.strip("'"), file_destpath.strip("'"))

        if source_type.lower() == "kaggle":
            data_source_uid = row["Dataset SOURCE UID"]
        elif source_type.lower() == "prod":
            logger.debug(f"filepath {filepath}")
            logger.debug(f"file_destpath {file_destpath}")
            logger.debug(f"filepath_strip {filepath_strip}")
            logger.debug(f"file_destpath_strip {file_destpath_strip}")

            data_source_uid = row["Dataset UID"]
        ligne = {
            "UID": str(uuid.uuid4()),
            "Dataset SOURCE": source_type,
            "Dataset SOURCE UID": data_source_uid,
            "Sous-répertoire": row["Sous-répertoire CIBLE"],
            "Classe": row["Classe"],
            "Nom de fichier": row["Nom de fichier"],
            "Date d'ajout": date_ajout,
            "md5": row["md5"],
            "Taille": row["Taille"],
            "Taille formattée": row["Taille formattée"],
            "Status": "ADDED"
        }
        new_data.append(ligne)

    df_new_metadata_csv = pd.concat(
        [df_new_metadata_csv, pd.DataFrame(new_data)], ignore_index=True)
    df_new_metadata_csv.to_csv(os.path.join(
        new_ref_dataset_path, "metadata.csv"), index=False)


def log_new_dataset(new_dataset_uid, source_type, new_ref_dataset_path, new_dataset_version, base_dataset_id, update_dataset_id, df_new_metadata_csv, ref_dataset_logging_path):
    nb_image_par_classe = df_new_metadata_csv.groupby(
        'Classe').size().to_dict()

    info_dataset = {
        "UID": new_dataset_uid,
        "Dataset Update SOURCE": source_type,
        "Dataset Update UID": update_dataset_id,
        "Dataset Name": os.path.basename(new_ref_dataset_path),
        "Dataset Version": new_dataset_version,
        "Dataset Base UID": base_dataset_id,
        "Date d'ajout": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Description": "Dataset REF",
        "Chemin du Dataset": new_ref_dataset_path,
        "Nombre total de fichiers": df_new_metadata_csv.shape[0],
        "Nombre d'images par classe": str(nb_image_par_classe),
        "Taille totale": df_new_metadata_csv["Taille"].sum(),
        "Taille totale formattée": utils_data.convert_size(df_new_metadata_csv["Taille"].sum()),
        "Disponible": True
    }

    df_ref_dataset = pd.read_csv(ref_dataset_logging_path)
    df_ref_dataset = pd.concat(
        [df_ref_dataset, pd.DataFrame([info_dataset])], ignore_index=True)
    df_ref_dataset.to_csv(ref_dataset_logging_path, index=False)

    logger.debug(f"Nouveau dataset loggé avec succès: {info_dataset}")


# FONCTION MAIN pour une mise à jour d'un dataset de REF depuis KAGGLE ou PROD
if __name__ == "__main__":
    """

    # Version KAGGLE
    update_dataset_ref(None,
                       source_type="KAGGLE", base_dataset_id=None)
    # FIN VERSION KAGGLE
    """

    # Version PROD
    # update_dataset_path = os.path.join(
    update_dataset_ref(None,
                       source_type="PROD", base_dataset_id=None)
