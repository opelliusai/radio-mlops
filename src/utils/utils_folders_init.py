'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Initialisation des répertoires 
Pour s'assurer qu'aucun répertoire principal ne manque pour la bonne exécution du code

'''
import os
from src.config.run_config import init_paths
from src.config.log_config import setup_logging

logger = setup_logging("UTILS_FOLDER_INIT")


def create_directories(paths):
    """
    Cette fonction crée des dossiers basés sur les chemins fournis.

    :param paths: Un dictionnaire où les clés sont les noms des dossiers et les valeurs sont les chemins.
    :type paths: dict

    :return: None
    None

    :raises OSError: Si les dossiers ne peuvent pas être créés.
    """

    logger.debug(f"-------------create_directories(paths={paths})")

    for _, path in paths.items():
        # Joindre le chemin principal avec le chemin spécifique
        full_path = os.path.join(paths["main_path"], path)

        try:
            # Créer le dossier s'il n'existe pas
            os.makedirs(full_path, exist_ok=True)
            logger.debug(f"Dossier créé: {full_path}")
        except OSError as error:
            logger.error(
                f"Le dossier {full_path} ne peut pas être créé. Erreur: {error}")


if __name__ == "__main__":
    create_directories(init_paths)
