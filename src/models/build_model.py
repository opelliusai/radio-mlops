'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Construction du modèle et utilisation de kerastuner pour optimiser la sélection du modèle

'''
# Imports
# EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB0

# Imports related to the Neural Networks construction
# Layers etc.
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Conv2D
from tensorflow.keras.models import Model  # Models objects
from tensorflow.keras.optimizers import Adam  # Optimizers
from tensorflow.keras.regularizers import l2  # Regularizers
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler  # Callbacks
# Import keras tuner
from kerastuner.tuners import RandomSearch

import numpy as np
import os
from sklearn.model_selection import train_test_split
import time

# Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths, infolog
from src.utils import utils_data
# Import de modules internes: ici preprocessing
from src.datasets import image_preprocessing
from src.config.log_config import setup_logging

logger = setup_logging("MODELS")

######
# Recherche des hyperparamètres - Keras Tuning Random Search
######


def tuner_randomsearch(ml_hp, full_run_folder, full_kt_folder, keras_tuning_history_filename, X, y, num_classes):
    """
    Performs hyperparameter tuning using RandomSearch with specified machine learning hyperparameters, 
    logging the process, and utilizing callbacks like EarlyStopping and LearningRateScheduler.

    uses the local build_model function.
    :param ml_hp: Dictionary containing machine learning hyperparameters including 'max_epochs', 
                  'num_trials'
    :param full_run_folder: The directory where training artifacts should be stored.
    :param full_kt_folder: The directory for Keras Tuner artifacts.
    :param keras_tuning_history_filename 
    :param X: Features dataset, expected to be in a format suitable for model training (e.g., numpy array).
    :param y: Labels dataset, expected to be in a format suitable for model training (e.g., numpy array).
    :param num_classes: Number of classes is dynamic therefore it is provided after the subfolders processing

    Tuning training logs are stored in the main_model_logs folder with the naming pattern: rs_tuning_training_log_{archi}_{run_id}.csv using ';' separator
    :return: The best model found during the RandomSearch tuning process.

    :raises KeyError: If required keys are missing in the `ml_hp` dictionary.
    :raises FileNotFoundError: If the specified directory does not exist.
    :raises Exception: For other exceptions not explicitly caught related to model building and tuning.
    """
    try:
        logger.debug("---------tuner_randomsearch------------")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=1234)

        # Logging split information
        logger.debug(
            f"Sizes X: {len(X)}, y: {len(y)}, X_train: {len(X_train)}, y_train: {len(y_train)}, X_val: {len(X_val)}, y_val: {len(y_val)}")
        # logger.debug(f"Shapes X: {X.shape}, y: {y.shape}, X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")

        max_epochs = ml_hp['max_epochs']
        num_trials = ml_hp['num_trials']
        # Callbacks configuration
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

        logger.debug(
            f" KT search history file name  {keras_tuning_history_filename}")
        full_csv_logger_path = os.path.join(
            full_run_folder, keras_tuning_history_filename)
        csv_logger = CSVLogger(full_csv_logger_path,
                               append=True, separator=';')
        logger.debug(
            f" KT search history file name stored in  {full_csv_logger_path}")

        # Learning rate adjustment
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * np.exp(-0.1)
        lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

        # Keras Tuner configuration
        tuner = RandomSearch(
            hypermodel=lambda hp: build_model_efficientnetb0(
                hp=hp, ml_hp=ml_hp, num_classes=num_classes),
            objective='val_accuracy',
            max_trials=num_trials,
            directory=full_kt_folder,
            project_name="trials"
        )

        # Tuning execution
        start_time = time.time()
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
        tuner.search(X_train, y_train, validation_data=(
            X_val, y_val), epochs=max_epochs, callbacks=[early_stopping, lr_scheduler, csv_logger])
        end_time = time.time()
        tuning_time = round(end_time - start_time, 2) / 60
        logger.debug(f"Keras Tuning time {tuning_time} min")

        # Best model retrieval
        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.hypermodel.build(best_hp)

        logger.debug(f"Best Hyperparameters {best_hp}")
        return best_model, best_hp

    except KeyError as e:
        logger.error(f"Missing key in ml_hp: {e}")
        raise KeyError(f"Missing key in ml_hp: {e}") from None
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        raise FileNotFoundError(f"Directory not found: {e}") from None
    except Exception as e:
        logger.error(f"An error occurred during tuning: {e}")
        raise Exception(f"An error occurred during tuning: {e}") from None


######
# Model : EfficientNetB0
######
def build_model_efficientnetb0(hp, ml_hp, num_classes):
    """
    Constructs an EfficientNetB0 architecture model for image classification, compatible with binary and multi-class scenarios.
    Utilizes both Keras Tuner optimization hyperparameters and general configuration parameters from a configuration object.
    This configuration includes:
    - 'archi': defining the architecture to be built,
    - 'img_size': defining the image size for model input,
    - 'img_dim': color dimensions (RGB 3 or Grayscale 1),
    - etc.

    :param hp: HyperParameters object from Keras Tuner for model hyperparameter optimization.
    :param ml_hp: MLFlow HyperParameters object containing configuration details.
    :param num_classes: Number of classes is dynamic therefore it is provided after the subfolders processing

    :return: A compiled Keras model optimized with hyperparameters.

    :raises KeyError: If essential keys are missing from the ml_hp dictionary.
    :raises ValueError: If configuration values in ml_hp are invalid or unsupported, such as an unsupported number of classes.
    """
    logger.debug("---------build_model_efficientnetb0------------")

    try:
        # 1a - Retrieving MLFlow hyperparameters
        img_size = ml_hp["img_size"]
        img_dim = ml_hp["img_dim"]
        # num_classes = ml_hp["num_classes"]

        # 1b - Optional parameters will have a default value
        hidden_layers_activation = ml_hp.get("hl_activation", "relu")

        # 1c - Initialization based on mlflow_archive hyperparameters
        shape = (img_size, img_size, img_dim)

        # 2 - Classification specifics: loss function
        if num_classes == 1:
            logger.debug("--- BINARY CLASSIFICATION ------")
            loss_function = 'binary_crossentropy'
            output_activation = "sigmoid"
        else:  # 0 refers to dynamic multiple classification , therefore any num_classes other than 1 will be non binary
            logger.debug("--- MULTICLASS CLASSIFICATION ------")
            loss_function = 'sparse_categorical_crossentropy'
            output_activation = "softmax"

        logger.debug("--- Hyperparameters ---")

        # 4 - Defining hyperparameters to be optimized with Keras Tuner
        learning_rate = hp.Float(
            'learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        dropout_rate = hp.Float(
            'dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])
        num_dropout_layers = hp.Int(
            'num_dropout_layers', min_value=1, max_value=5)
        dropout_connect_rate = hp.Float(
            'dropout_connect_rate', min_value=0.2, max_value=0.4, step=0.1)

        logger.debug("--- Architecture-specific details ---")
        logger.debug(f"dropout_connect_rate = {dropout_connect_rate}")

        # 5 - Loading the base model
        base_model = EfficientNetB0(
            weights='imagenet', include_top=False, input_shape=shape)

        # 6 - Layer adjustments for fine-tuning
        # Fine-tuning strategy: Unfreezing convolutional layers while keeping batch normalization layers frozen
        for layer in base_model.layers:
            layer.trainable = isinstance(layer, Conv2D)

        # 7 - Adding Fully Connected Layers
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(units, activation=hidden_layers_activation,
                  kernel_regularizer=l2(l2_lambda))(x)
        x = BatchNormalization()(x)

        # 8 - Adding Dropout Layers
        for _ in range(num_dropout_layers):
            x = Dropout(dropout_rate)(x)

        # 9 - Output Layer
        output = Dense(num_classes, activation=output_activation)(x)

        # 10 - Finalizing model construction
        model = Model(inputs=base_model.input, outputs=output)

        # 11 - Compiling the model with hyperparameters and classification-specific loss functions
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=loss_function, metrics=['accuracy'])
        return model
    except KeyError as e:
        logger.error(f"KeyError: Missing key in ml_hp: {e}")
        raise KeyError(f"Missing key in ml_hp: {e}") from None
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise ValueError(e) from None


def build_model(ml_hp, X, y, num_classes):
    full_run_folder = os.path.join(
        init_paths["main_path"], init_paths["run_folder"])  # ./data/processed/mflow
    full_kt_folder = os.path.join(
        init_paths["main_path"], init_paths["keras_tuner_folder"])  # ./data/processed/kt
    keras_tuning_history_filename = "Covid19_3C_EffnetB0_model_history_run_id.csv"
    model = tuner_randomsearch(ml_hp, full_run_folder, full_kt_folder,
                               keras_tuning_history_filename, X, y, num_classes)
    return model


def main():
    logger.debug("Préparation des paramètres")
    ml_hp = {}
    ml_hp['max_epochs'] = 18
    ml_hp['num_trials'] = 5
    ml_hp["img_size"] = 224
    ml_hp["img_dim"] = 3
    dataset_path = os.path.join(
        init_paths["main_path"], init_paths["processed_datasets_folder"], "COVID-19_MC_1.4")  # ./data/raw/datasets
    logger.debug("main - dataset_path{dataset_path}")
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
    num_classes = 3
    full_run_folder = os.path.join(
        init_paths["main_path"], init_paths["run_folder"])  # ./data/processed/mflow
    full_kt_folder = os.path.join(
        init_paths["main_path"], init_paths["keras_tuner_folder"])  # ./data/processed/kt
    keras_tuning_history_filename = "Covid19_3C_EffnetB0_model_history_run_id.csv"
    model = tuner_randomsearch(ml_hp, full_run_folder, full_kt_folder,
                               keras_tuning_history_filename, X, labels_num, num_classes)
    return model


if __name__ == "__main__":
    main()
