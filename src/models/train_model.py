'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Entrainement (ou réentrainement) du modèle sélectionné

'''

#####
# Imports
#####
import os
from sklearn.model_selection import train_test_split
import time
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler  # Callbacks

from src.config.run_config import init_paths, dataset_info, infolog, model_hp

# Import des modules internes
from src.models import build_model
from src.datasets import update_dataset, image_preprocessing
from src.utils import utils_data
from src.utils import utils_models
from src.config.log_config import setup_logging

logger = setup_logging("MODELS")

# FUNCTIONS


def train_model(model, ml_hp, X, y, full_run_folder, model_training_history_csv):
    logger.debug(
        f"train_model(model={model}, ml_hp={ml_hp}, X, y={y}, full_run_folder={full_run_folder}, model_training_history_csv={model_training_history_csv})")
    """
    Trains the model using provided data, MLFlow hyperparameters, and run ID for file naming (CSVLogger).
    It could be an initial model training, or a new training based on new production data
    :param model: The model to be trained.
    :param ml_hp: MLFlow hyperparameters.
    :param X: Training data features.
    :param y: Training data labels.
    :param run_id: Used for naming callback files (CSVLogger).
    
    :return: A tuple containing the trained model, basic metrics (other metrics to be calculated in MLFlow), training history, and execution time.
    
    :raises KeyError: If 'max_epochs' is missing from ml_hp.
    :raises ValueError: If data splitting results in empty training or validation sets.
    """
    logger.debug("--------- train_model ---------")
    try:
        # 1 - Retrieving MLFlow hyperparameters for model training
        max_epochs = ml_hp.get('max_epochs')
        if max_epochs is None:
            raise KeyError("'max_epochs' must be specified in ml_hp.")

        # 2 - Splitting data for training
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=1234)
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(
                "Data splitting resulted in empty training or validation sets.")
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        logger.debug("Data Split Info")
        logger.debug(f"Size X {len(X)}")
        logger.debug(f"Size y {len(y)}")
        logger.debug(f"Train X size {len(X_train)}")
        logger.debug(f"Train y size {len(y_train)}")
        logger.debug(f"Validation X size {len(X_val)}")
        logger.debug(f"Validation y size {len(y_val)}")

        # Callbacks: Early Stopping, CSV Logger, LearningRateScheduler with patience of 10 epochs
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

        # CSV Logger
        # csv_logger_filename=f"model_training_logs_{run_id}_{ml_hp['archi']}.csv"
        logger.debug(
            f" Model Training CSV logger file name  {model_training_history_csv}")
        full_csv_logger_path = os.path.join(
            full_run_folder, model_training_history_csv)
        csv_logger = CSVLogger(full_csv_logger_path,
                               append=True, separator=';')
        logger.debug(
            f" Model Training history file name stored in  (/!\ may be duplicate with the json file) {full_csv_logger_path}")

        def scheduler(epoch, lr):
            return lr * np.exp(-0.1) if epoch >= 10 else lr
        lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

        # Timing model training
        start_time = time.time()
        logger.debug(f"Type X {type(X)} / Len X {len(X)}")
        logger.debug(
            f"Type X_train {type(X_train)} / Len X_train {len(X_train)}")
        logger.debug(f"Type X_val {type(X_val)} / Len X_val {len(X_val)}")

        logger.debug(f"Type y {type(y)} / Len y {len(y)}")
        logger.debug(
            f"Type y_train {type(y_train)} / Len y_train {len(y_train)}")
        logger.debug(f"Type y_val {type(y_val)} / Len y_val {len(y_val)}")
        logger.debug(f"Type model {type(model)}")
        history = model.fit(X_train, y_train, epochs=max_epochs, validation_data=(
            X_val, y_val), callbacks=[early_stopping, lr_scheduler, csv_logger])
        end_time = time.time()

        execution_time = round(end_time - start_time, 2) / 60
        logger.debug(f"Model training time {execution_time} min")

        # Gathering basic metrics
        metrics = {
            'accuracy': max(history.history['accuracy']),
            'val_accuracy': max(history.history['val_accuracy']),
            'loss': max(history.history['loss']),
            'val_loss': max(history.history['val_loss']),
        }

        logger.debug(f"Execution time {execution_time}")
        logger.debug(f"history {history.history}")
        return model, metrics, history.history

    except KeyError as e:
        logger.error(f"KeyError: {e}")
        raise
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise


'''
def drift_detection_action():
    """
    Activé lorsqu'un drift est détecté. 
    # Effectue l'entrainement du modèle sur les données de reference + prod
    # Effectuer un réentrainement du modèle de production sur les données de production
    """
    logger.debug(f"-----drift_detection_action()----")
    ref_dataset=os.path.join(init_paths["main_path"],init_paths["processed_datasets_folder"],dataset_info["dataset_prefix"]+dataset_info["dataset_version"])
    prod_dataset=os.path.join(init_paths["main_path"],init_paths["prod_datasets_folder"],dataset_info["prod_dataset_prefix"]+dataset_info["prod_dataset_version"])
    logger.debug(f"ref_dataset {ref_dataset}")
    logger.debug(f"prod_dataset {prod_dataset}")
    
    logger.debug("1- Réentrainement du modèle actuelle sur les données de production")
    logger.debug("-- Chargement du modèle de production")
    current_model = utils_models.get_mlflow_prod_model()
    #retrain_model_main(current_model,prod_dataset)
    logger.debug("2- Fusion des deux datasets ref + prod et Entrainement d'un nouveau modèle")
    new_dataset_path=update_dataset.merge_datasets()
    logger.debug(f"new_dataset_path {new_dataset_path}")
    model_tracking.mlflow_train_model(new_dataset_path,
                    max_epochs=1,
                    num_trials=1)
'''


def retrain_model_main(model, dataset_path, max_epochs=18, num_trials=5):
    ml_hp = {}
    ml_hp['max_epochs'] = max_epochs
    ml_hp['num_trials'] = num_trials
    # Paramètres additionels pour l'entrainement
    ml_hp["img_size"] = 224
    ml_hp["img_dim"] = 3
    full_run_folder = os.path.join(
        init_paths["main_path"], init_paths["run_folder"])  # ./data/processed/mflow
    data = image_preprocessing.preprocess_data(dataset_path, 224, 3)
    X, y = map(list, zip(*data))
    for i in y:
        logger.debug(f"Label {i}")
    # récupérer une liste unique des labels
    unique_labels = list(set(y))
    logger.debug(f"Unique labels {unique_labels}")
    # Créer un dictionnaire pour mapper les labels à des entiers
    labels_dic = utils_data.generate_numeric_correspondance(unique_labels)
    logger.debug(f"Labels dic {labels_dic}")
    labels_num = utils_data.label_to_numeric(y, labels_dic)
    logger.debug(f"Labels num {labels_num}")
    model = build_model.main()
    model_training_history_csv = "Covid19_3C_EffnetB0_model_history_run_id.csv"
    final_model, metrics, history = train_model(
        model, ml_hp, X, labels_num, full_run_folder, model_training_history_csv)
    logger.debug(f"metrics {metrics}")
    logger.debug(f"history {history}")
    version = "1.1"
    model_save_path_keras = os.path.join(
        init_paths["main_path"], init_paths["models_path"], f"COVID19_Effnetb0_Model_{version}.keras")
    model_save_path_h5 = os.path.join(
        init_paths["main_path"], init_paths["models_path"], f"COVID19_Effnetb0_Model_{version}.h5")
    utils_models.save_model(final_model, model_save_path_keras)
    utils_models.save_model(final_model, model_save_path_h5)


def train_model_main(dataset_path, max_epochs=18, num_trials=5):
    # A SUPPRIMER car on passe désormais par MLFlow
    ml_hp = {}
    ml_hp['max_epochs'] = max_epochs
    ml_hp['num_trials'] = num_trials
    # Paramètres additionels pour l'entrainement
    ml_hp["img_size"] = 224
    ml_hp["img_dim"] = 3
    full_run_folder = os.path.join(
        init_paths["main_path"], init_paths["run_folder"])  # ./data/processed/mflow
    data = image_preprocessing.preprocess_data(dataset_path, 224, 3)
    X, y = map(list, zip(*data))
    for i in y:
        logger.debug(f"Label {i}")
    # récupérer une liste unique des labels
    unique_labels = list(set(y))
    logger.debug(f"Unique labels {unique_labels}")
    # Créer un dictionnaire pour mapper les labels à des entiers
    labels_dic = utils_data.generate_numeric_correspondance(unique_labels)
    logger.debug(f"Labels dic {labels_dic}")
    labels_num = utils_data.label_to_numeric(y, labels_dic)
    logger.debug(f"Labels num {labels_num}")
    model = build_model.main()
    model_training_history_csv = "Covid19_3C_EffnetB0_model_history_run_id.csv"
    final_model, metrics, history = train_model(
        model, ml_hp, X, labels_num, full_run_folder, model_training_history_csv)
    logger.debug(f"metrics {metrics}")
    logger.debug(f"history {history}")
    version = "1.1"
    model_save_path_keras = os.path.join(
        init_paths["main_path"], init_paths["models_path"], f"COVID19_Effnetb0_Model_{version}.keras")
    model_save_path_h5 = os.path.join(
        init_paths["main_path"], init_paths["models_path"], f"COVID19_Effnetb0_Model_{version}.h5")
    utils_models.save_model(final_model, model_save_path_keras)
    utils_models.save_model(final_model, model_save_path_h5)


if __name__ == "__main__":
    drift_detection_action()
