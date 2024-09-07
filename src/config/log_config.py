'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fichier de configuration des logs
Actuellement toutes les logs seront stockées dans le même fichier projet
A terme sera divisé par périmètre en déclarant le logger au niveau du fichier

'''
# IMPORTS
# Imports externes
import os
import logging.handlers
# Imports internes
from src.config.run_config import init_paths, infolog

# FONCTION PRINCIPALE


def setup_logging(logfile_label):
    '''
    Logging setup, using run_config infolog information
    '''
    main_path = init_paths["main_path"]
    log_folder = init_paths["logs_folder"]
    logfile_prefix = infolog["logfile_prefix"]
    logfile_name = f"{logfile_prefix}{logfile_label}.log"
    logfile_path = os.path.join(main_path, log_folder, logfile_name)

    # Ensure the log folder exists
    os.makedirs(os.path.join(main_path, log_folder), exist_ok=True)

    # Create a formatter and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        logfile_path, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Create a logger and add handlers to it
    logger = logging.getLogger(infolog["project_name"])
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
