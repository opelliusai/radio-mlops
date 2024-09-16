'''
Créé le 05/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Fonctions utiles pour les modèles:
-- sauvegarde d'un modèle
-- chargement d'un modèle
-- sauvegarde/chargement des résultats d'un modèle (historique, training plots)

'''

# IMPORTS
import os
from tensorflow.keras.models import load_model
import pickle
import json
# Usual functions for calculation, dataframes and plots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
import csv
import datetime
import uuid
from mlflow.tracking import MlflowClient
import mlflow
# metrics
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.models import Model
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
import keras

# Internal imports
from src.datasets import image_preprocessing
from src.utils import utils_data
# Import des fichiers de configuration et des informations sur les logs
from src.config.run_config import init_paths, model_info, mlflow_info, infolog
from src.config.log_config import setup_logging

logger = setup_logging("UTILS_MODELS")
mlflow_uri = mlflow_info["mlflow_tracking_uri"]
client = MlflowClient(
    registry_uri=mlflow_uri, tracking_uri=mlflow_uri)

mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)
# Fonctions

# Sauvegarder les predictions


def save_prediction(model_name, image_path, prediction, confiance, temps_prediction, date_prediction, username):
    '''
        Mise à jour du fichier de prediction avec les informations recueillies
        Et sauvegarde de l'image dans le dossier des images prédites
    '''
    logger.debug(
        "-------save_prediction(model_name={model_name}, image_path={img_path}, prediction={prediction}, confiance={confiance}, temps_prediction={temps_prediction}, date_prediction={date_prediction},username={username}")
    unique_id = str(uuid.uuid4())
    # Créer un dictionnaire avec les informations
    log_exist, prediction_logging_filepath = utils_data.check_logging_exists(
        "PRED")
    if not log_exist:
        logger.debug(f"Initialisation du fichier de log des predictions")
        utils_data.initialize_logging_file("PRED")

    pred_data = {
        'UID': unique_id,
        'Nom du modèle': model_name,
        'Chemin de l\'image': image_path,
        'Taille': utils_data.get_file_size(image_path),
        'Taille formattée': utils_data.convert_size(utils_data.get_file_size(image_path)),
        'md5': utils_data.calcul_md5(image_path),
        'Prédiction': prediction,
        'Indice de confiance': confiance,
        'Temps de prédiction': temps_prediction,
        'Date de prédiction': date_prediction,
        'Prédiction validée': "N/A",
        'Perf Prédiction': "N/A",
        "Username": username
    }
    with open(prediction_logging_filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        # Si le fichier n'existe pas, on ajoute l'entête
        writer.writerow(pred_data.values())
    return unique_id

# FUNCTIONS
# Keras Models save and load


def save_model(model, save_path):
    """
    Saves a Keras model to the specified path.

    :param model: The model to save.
    :param save_path: The file FULL path where to save the model.
    :raises PermissionError: If there has been a write permision issue
    :raises IOError: if there has been an I/O error
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------save_model--------------------")
    try:  # Saving the model
        # Extract the directory from the complete file path
        directory = os.path.dirname(save_path)

        # Check if the directory already exists
        if not os.path.exists(directory):
            # Create the directory if it does not exist
            os.makedirs(directory)

        # Save the model to the specified path
        model.save(save_path)

    except PermissionError:
        # Exception if there is no permission to write in the directory
        logger.error(
            "Error: No permission to write to the specified directory.")
        raise
    except IOError as e:
        # Handle general I/O errors
        logger.error(f"An I/O error occurred: {e}")
        raise
    except Exception as e:
        # Handle other possible exceptions
        logger.error(f"An unexpected error occurred: {e}")
        raise

    return save_path


def save_weights(model, save_path):
    """
    Saves a Keras model to the specified path.

    :param model: The model to save.
    :param save_path: The file FULL path where to save the model.
    :raises PermissionError: If there has been a write permision issue
    :raises IOError: if there has been an I/O error
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------save_model--------------------")
    try:  # Saving the model
        # Extract the directory from the complete file path
        directory = os.path.dirname(save_path)

        # Check if the directory already exists
        if not os.path.exists(directory):
            # Create the directory if it does not exist
            os.makedirs(directory)

        # Save the model to the specified path
        model.save_weights(save_path)

    except PermissionError:
        # Exception if there is no permission to write in the directory
        logger.error(
            "Error: No permission to write to the specified directory.")
        raise
    except IOError as e:
        # Handle general I/O errors
        logger.error(f"An I/O error occurred: {e}")
        raise
    except Exception as e:
        # Handle other possible exceptions
        logger.error(f"An unexpected error occurred: {e}")
        raise

    return save_path


# Models loading. Names load_models to distinguish the function from the keras 'load_model' function

def load_current_model():
    """
    Charge le modèle courant
    Construit le chemin vers le modèle courant et le charge
    :return: Le modèle chargé.
    :raises IOError: if the file to load is not accessible
    :raises ValueError: If there has been an issue with the model interpretation (not a model or corrupted file)
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------load_current_model--------------------")
    try:
        # Load the model from the specified file path
        logger.debug(f"Construction du chemin du modèle")
        load_path = os.path.join(
            init_paths["main_path"], init_paths["models_path"], model_info["selected_model_name"])
        logger.debug(f"Chemin du modèle courant - {load_path}")
        model = load_model(load_path)
        logger.info("Modèle chargé")
        return model
    except IOError as e:
        # Handle errors related to file access issues
        logger.error(f"Error: Could not access file to load model. {e}")
        raise
    except ValueError as e:
        # Handle errors related to the model file being invalid or corrupted
        logger.error(
            f"Error: The file might not be a Keras model file or it is corrupted. {e}")
        raise
    except Exception as e:
        # Handle other possible exceptions
        logger.error(
            f"An unexpected error occurred while loading the model: {e}")
        raise
# Models training history save and load (pickle or json format)


def save_history(history, save_path):
    """
    Saves a training history to a file, supporting both Pickle, JSON and CSV formats.

    uses pickle.dump or json.dump depending on the specified file path extension
    :param history: The training history to save.
    :param save_path: The FULL file path where the training history should be saved.

    :raises If there has been an issue with the model interpretation (not a model or corrupted file)
    :raises PermissionError: If the write permission is not granted to the specified path
    :raises IOError: if there is an I/O issue
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------save_history--------------------")

    try:
        # Ensure the directory exists
        logger.debug(f"Saving file in {save_path}")
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Determine the file format from the extension and save accordingly
        _, ext = os.path.splitext(save_path)
        if ext == '.pkl':
            with open(save_path, 'wb') as f:
                pickle.dump(history, f)
        elif ext == '.json':
            with open(save_path, 'w') as f:
                json.dump(history, f)
        elif ext == '.csv':
            # We need to ensure that the history object is a dataframe, if not, saving might be corrupted and therefore the save request is rejected
            if isinstance(history, pd.DataFrame):
                history.to_csv(save_path, index=False)
            else:
                raise ValueError(
                    "Unsupported object format. Object must be a dataframe to be stored in a csv format")
        else:
            raise ValueError(
                "Unsupported file format. Please use .pkl or .json.")
    except PermissionError:
        logger.error(f"Error: No permission to write to {save_path}.")
        raise
    except IOError as e:
        logger.error(f"An I/O error occurred while saving to {save_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    return save_path

# Models training history load (pickle, json or csv format)


def load_history(load_path):
    """
    Loads a training history from a file, supporting both pickle and JSON formats.
    uses pickle.load or json.load depending on the specified file path extension
    :param load_path: The file path of the training history to load.
    :return: The loaded training history.
    :raises ValueError: If the file format is not suppoeted
    :raises IOError: if there is an I/O issue
    :raises pickle.PickleError: if the pickle file might be corrupted
    :raises json.JSONDecodeError: if the JSON file might be corrupted
    :raises pd.errors.ParserError: if the CSV file to be stored in a dataframe might be corrupted
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------load_history--------------------")
    try:
        # Determine the file format from the extension
        _, file_extension = os.path.splitext(load_path)
        with open(load_path, 'rb' if file_extension.lower() == '.pkl' else 'r') as f:
            if file_extension.lower() == '.pkl':
                history = pickle.load(f)
            elif file_extension.lower() == '.json':
                history = json.load(f)
            elif file_extension.lower() == '.csv':
                history = pd.read_csv(f)
            else:
                raise ValueError(
                    "Unsupported file format. Please use .pkl, .json or .csv .")
        return history
    except IOError as e:
        logger.error(f"Error: Could not access file at {load_path}. {e}")
        raise
    except (pickle.PickleError, json.JSONDecodeError, pd.errors.ParserError) as e:
        logger.error(
            f"Error: The file might be corrupted or in an incorrect format. {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

# Saving Training visual reports (history) Pickle, JSON or CSV file


def generate_training_plots(history_file, output_filepath, run_info):
    """
    Generates and saves training and validation loss and accuracy plots from a history file.
    The history file can be in PKL, CSV, or JSON format.

    :param history_file: The FULL filepath of the training history file.
    :param output_folder: The output directory to save the plots.
    :param run_info: (currently run id) Information or identifier for the training run to be included in the plot titles.
    the file name format is "{run_info}_training_validation_plots.png"
    :raises Exception: If an error occurs
    """
    logger.debug(
        "--------------------generate_training_plots--------------------")
    # Determine file extension and load history accordingly
    logger.debug(f"history_file {history_file}")
    logger.debug(f"output_folder {output_filepath}")
    logger.debug(f"run_id {run_info}")
    _, file_extension = os.path.splitext(history_file)

    # first step is to load the history file
    try:
        if file_extension.lower() == '.pkl':
            # Load history from PKL file
            logger.debug("Managing Pickle file")
            history = pd.read_pickle(history_file)
        elif file_extension.lower() == '.csv':
            # Load history from CSV file
            logger.debug("Managing CSV file")
            history = pd.read_csv(history_file)
        elif file_extension.lower() == '.json':
            # Load history from JSON file
            logger.debug("Managing JSON file")
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            raise ValueError(
                "Unsupported file format. Please use .pkl, .csv, or .json")

        # Create plot
        plt.figure(figsize=(12, 4))

        # Loss subplot
        plt.subplot(121)
        plt.plot(history['loss'], label='train')
        plt.plot(history['val_loss'], label='test')
        plt.title(f'Run {run_info} - Model loss by epoch')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')

        # Accuracy subplot
        plt.subplot(122)
        plt.plot(history['accuracy'], label='train')
        plt.plot(history['val_accuracy'], label='test')
        plt.title(f'Run {run_info} - Model accuracy by epoch')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')

        # Save the plot
        plt.savefig(output_filepath)
        plt.close()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

# Save confusion matrix and classification report from dataframes


def save_dataframe_plot(df, output_filepath, plot_type, labels_dict=None):
    """
    Generates and saves a plot based on the specified type, appending the plot type as a suffix to the run ID for the filename.

    :param df: DataFrame to be plotted, can be a confusion matrix or a classification report.
    :param output_folder: The output directory to save the plot.
    :param run_id: The identifier for the run or experiment. Used as the base name for the output file.
    :param plot_type: Type of the plot to generate - 'confusion_matrix' or 'classification_report'. Also used as a suffix for the filename.
    :param labels_dict: Optional; needed if plot_type is 'confusion_matrix'. It maps class numbers to class names.
    """
    logger.debug("--------------------save_dataframe_plot--------------------")
    # Ensure plot_type is valid and prepare the plot accordingly
    if plot_type == 'confusion_matrix' and labels_dict is not None:
        # Generate labels for confusion matrix
        axis_labels = [labels_dict[i] for i in labels_dict]

        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=axis_labels, yticklabels=axis_labels)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')

    elif plot_type == 'classification_report':
        plt.figure(figsize=(10, len(df) * 0.5))
        sns.heatmap(data=df, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.title('Classification Report')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
    else:
        raise ValueError(
            "Invalid plot type specified. Use 'confusion_matrix' or 'classification_report'.")

    # Save the plot
    plt.savefig(output_filepath)
    plt.close()

    logger.debug(f"Plot saved to {output_filepath}")


def get_prediction_metrics(model, X_eval, y_eval):
    """
    Calculates and logs various metrics for multiclass classification including accuracy, recall, F1-score,
    classification report, and confusion matrix, specifically for models trained with
    sparse categorical cross-entropy loss.

    :param model: The trained model to evaluate.
    :param X_eval: Evaluation data features.
    :param y_eval: True labels for evaluation data as integers (not one-hot encoded).

    :return: A dictionary containing calculated metrics such as accuracy, loss, recall, F1-score,
             confusion matrix, and classification report.

    :raises ValueError: If an error occurs during the prediction or metrics calculation process.
    """
    metrics_dict = {}
    try:
        logger.debug(
            '----------get_prediction_metrics(model,X_eval,y_eval)-------')

        # Model prediction
        y_eval_pred = model.predict(X_eval)
        logger.debug(f"Y EVAL {y_eval}")
        logger.debug(f"Y EVAL PRED {y_eval_pred}")

        # Converting predictions to class indices
        y_eval_pred = np.argmax(y_eval_pred, axis=1)
        logger.debug(f"Y EVAL PRED Classes {y_eval_pred}")

        # Loss calculation using SparseCategoricalCrossentropy
        sce = SparseCategoricalCrossentropy(from_logits=False)
        loss = sce(y_eval, model.predict(X_eval)).numpy()

        logger.debug("MULTICLASS CASE")
        logger.debug(f"NEW Y EVAL PRED {y_eval_pred}")

        # Metrics calculation
        accuracy_s = accuracy_score(y_eval, y_eval_pred)
        recall = recall_score(y_eval, y_eval_pred, average="macro")
        f1 = f1_score(y_eval, y_eval_pred, average="macro")
        class_report = classification_report(
            y_eval, y_eval_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_eval, y_eval_pred)

        # Logging calculated metrics
        logger.debug("Precision/accuracy :", accuracy_s)
        logger.debug("Loss (Sparse Categorical Cross-Entropy) :", loss)
        logger.debug("Recall :", recall)
        logger.debug("F1-Score :", f1)
        logger.debug(f"Classification Report :{class_report}")
        logger.debug(f"Confusion Matrix {conf_matrix}")

        # Metrics aggregation
        metrics_dict = {
            "Accuracy": accuracy_s,
            "Loss": loss,
            "Recall": recall,
            "F1-score": f1,
            "Confusion Matrix": conf_matrix,
            "Classification report": class_report
        }

        return metrics_dict
    except Exception as e:
        logger.error(f"Error during multi-class model evaluation: {e}")
        raise ValueError(f"Error during multi-class model evaluation: {e}")


# Utils MLFlow
def get_mlflow_prod_model():
    model_name = model_info["model_name_prefix"]
    model_version = get_mlflow_prod_version()
    logger.debug(
        f"Chargement du modèle de production {model_name}-{model_version}pour les predictions")
    model_uri = f"models:/{model_name}/Production"
    # model = mlflow.pyfunc.load_model(
    #    model_uri=model_uri)
    model = mlflow.tensorflow.load_model(model_uri=model_uri)
    logger.debug(f"Modele {model_name} chargé")
    return model, model_name, model_version


def get_mlflow_prod_model_tensorflow():
    model_name = model_info["model_name_prefix"]
    model_version = get_mlflow_prod_version()
    logger.debug(
        f"Chargement du modèle de production {model_name}-{model_version}pour les predictions")
    model = mlflow.tensorflow.load_model(
        model_uri=f"models:/{model_name}/Production")
    logger.debug(f"Modele {model_name} chargé")
    return model, model_name, model_version


def get_mlflow_prod_version():
    model_name = model_info["model_name_prefix"]
    logger.debug(f"model_name NAME {model_name}")
    # Search for model versions with a specific name
    filter_string = f"name='{model_info['model_name_prefix']}'"
    model_versions = client.search_model_versions(filter_string=filter_string)

    # Filter the model versions based on the stage
    model_details = [
        version for version in model_versions if version.current_stage == "Production"]

    if model_details:
        production_version = model_details[0].version
        logger.debug(
            f"La version du modèle déployé en Production est : {production_version}")
    else:
        logger.debug("Aucune version en Production trouvée.")
        production_version = None
    return production_version

# Utils Flag pour la mise à jour du modèle
# Externalisation des informations du modèle dans un fichier externe


def get_models(model_name=model_info["model_name_prefix"]):
    filter_string = f"name='{model_name}'"
    model_versions = client.search_model_versions(filter_string=filter_string)
    models_info = []
    i = 1
    runs_info = []
    for model in model_versions:

        run_info = get_run_info(model.run_id)
        experiment_id = run_info.get('EXPERIMENT_ID')
        logger.debug(f"Experiment ID {experiment_id}")
        logger.debug(f"run_info {run_info}")
        logger.debug(f"RUN STATUS {run_info.get('STATUS')}")
        # if run_info.status
        if model.status == "READY" and run_info.get("STATUS") == "active":
            logger.debug(f"Ready Model found {model.name} - {model.version}")
            logger.debug(f"Model RUN ID {model.run_id}")
            # logger.debug(f"run_info {run_info}")
            runs_info.append(run_info)
            model_info = {}
            model_info["Name"] = model.name
            model_info["Version"] = model.version
            timestamp = model.creation_timestamp
            timestamp_seconds = timestamp / 1000
            dt_object = datetime.datetime.fromtimestamp(timestamp_seconds)
            formatted_date = dt_object.strftime('%d/%m/%Y - %H:%M:%S')
            model_info["Date de création"] = formatted_date
            model_info["Etat"] = model.status
            model_info["Phase"] = model.current_stage
            model_info["RUN_ID"] = model.run_id
            model_info["EXPERIMENT_ID"] = experiment_id
            model_info["PARAMS"] = run_info["PARAMS"]
            model_info["Durée"] = run_info["Durée"]
            model_info["Link"] = get_mlflow_link(
                model_info["EXPERIMENT_ID"], model_info["RUN_ID"])
            '''
            model_info["Confusion Matrix"] = get_confusion_matrix_from_run(
                model.run_id)
            
            model_info["Classification Report"] = get_classification_report_from_run(
                model.run_id)
            '''
            models_info.append(model_info)
    logger.debug(f"Model infos {models_info}")
    logger.debug(f"RUN infos {runs_info}")
    logger.debug(f"Type Model infos {type(models_info)}")
    logger.debug(f"Type RUN infos {type(runs_info)}")

    # runs_info = get_runs_info(model_info["RUN"] for model_info in models_info)
    return models_info


def get_mlflow_link(experiment_id, run_id):
    experiment_link = f"{mlflow_uri}/#/experiments/{experiment_id}/runs/{run_id}"
    return experiment_link


def get_run_info(run_id):
    logger.debug(f"-----------get_run_info(run_id={run_id})-------")
    run = client.get_run(run_id=run_id)
    run_info = {}
    run_info["RUN_ID"] = run.info.run_id
    # run_info["Nom"]=run.info.run_name
    run_info["EXPERIMENT_ID"] = run.info.experiment_id
    run_info["PARAMS"] = run.data.params
    # run_info["ARTIFACTS"]=run.info.artifact_uri
    # run_info["metrics"]=run.data.metrics
    logger.debug(f"run.info.end_time {run.info.end_time}")
    logger.debug(f"run.info.start_time {run.info.start_time}")
    if run.info.end_time is not None and run.info.start_time is not None:
        timestamp_duree = run.info.end_time - run.info.start_time
    else:
        timestamp_duree = 0
    timestamp_seconds = timestamp_duree / 1000
    dt_object = datetime.datetime.fromtimestamp(timestamp_seconds)
    formatted_date = dt_object.strftime('%M:%S')
    run_info["Durée"] = formatted_date
    run_info["STATUS"] = run.info.lifecycle_stage

    return run_info


def get_runs_info(run_ids):
    runs_info = []
    for run_id in run_ids:
        run_info = get_run_info(run_id)
        logger.debug(f"Run info for {run_id} / {run_info}")
        if run_info["STATUS"] == 'active':
            runs_info.append(run_info)

    return runs_info


def update_log_prediction(pred_id, label):
    logger.debug(
        f"-----------update_log_prediction(pred_id={pred_id},label={label})-----------")
    pred_filepath = utils_data.get_logging_path("PRED")
    logger.debug(f"Prediction filepath {pred_filepath}")
    # Lecture du fichier des prediction et parcours des lignes jusqu'à trouver la ligne correspondant à la prediction
    df = pd.read_csv(pred_filepath)

    # Mise à jour de la valeur de prédiction validée
    # logger.debug(f'Ancienne valeur {df.loc[df["UID"]==pred_id,"Prédiction validée"]}')
    logger.debug(
        f'Ancienne valeur {df.loc[df["UID"]==pred_id,"Prédiction validée"].fillna("N/A")}')
    df["Prédiction validée"] = df["Prédiction validée"].astype(str)
    df.loc[df["UID"] == pred_id, "Prédiction validée"] = label
    df.loc[(df['Prédiction'] == df['Prédiction validée']),
           'Perf Prédiction'] = 'Correct'
    df.loc[(df['Prédiction'] != df['Prédiction validée']),
           'Perf Prédiction'] = 'Incorrect'
    logger.debug(
        f'Nouvelle valeur {df.loc[df["UID"]==pred_id,"Prédiction validée"]}')
    logger.debug(
        f'Nouvelle valeur {df.loc[df["UID"]==pred_id,"Perf Prédiction"]}')
    df.to_csv(pred_filepath, index=False)

    return "OK"


def save_drift_metrics(model_name, new_mean, original_mean, new_std, original_std, mean_diff, std_diff,
                       drift, drift_calculation_date, temps_calcul):
    ''' Mise à jour du fichier de prediction avec les informations recueillies'''
    logger.debug(
        f"-------save_drift_metrics(model_name={model_name}, new_mean={new_mean}, original_mean={original_mean}, new_std={new_std}, original_std={original_std}, drift={drift}, drift_calculation_date={drift_calculation_date}, temps_calcul={temps_calcul})----")
    # Créer un dictionnaire avec les informations
    unique_id = str(uuid.uuid4())
    data = {
        'UID': unique_id,
        'Nom du modèle': model_name,
        'New Mean': new_mean,
        'Original Mean': original_mean,
        'Original': original_mean,
        'New STD': new_std,
        'Original STD': original_std,
        'Mean Diff': mean_diff,
        'STD Diff': std_diff,
        'Drift': drift,
        'Date de calcul': drift_calculation_date,
        'Temps de calcul': temps_calcul
    }
    drift_filepath = utils_data.get_logging_path("DRIFT_DATA")
    logger.debug(f"Drift filepath {drift_filepath}")
    file_exists = os.path.isfile(drift_filepath)
    with open(drift_filepath, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        # Si le fichier n'existe pas, on ajoute l'entête
        if not file_exists:
            logger.debug(f"Fichier de logging n'existe pas")
            logger.debug(f"Ecriture de l'entête {data.keys()}")
            writer.writerow(data.keys())
        else:
            logger.debug(
                "Le fichier de logging existe, stockage des résultats en fin de fichier")
        logger.debug(f"Ecriture des résultats {data.values()}")
        # Write the new data to the file
        writer.writerow(data.values())
    return unique_id


def save_drift_metrics_model(model_name, recall_diff,
                             drift, drift_calculation_date, temps_calcul):
    ''' Mise à jour du fichier de prediction avec les informations recueillies'''
    logger.debug(
        f"-------save_drift_metrics(model_name={model_name}, recall_diff={recall_diff},drift={drift}, drift_calculation_date={drift_calculation_date}, temps_calcul={temps_calcul})----")
    # Créer un dictionnaire avec les informations
    unique_id = str(uuid.uuid4())
    data = {
        'UID': unique_id,
        'Nom du modèle': model_name,
        'RECALL Diff': recall_diff,
        'Drift': drift,
        'Date de calcul': drift_calculation_date,
        'Temps de calcul': temps_calcul
    }
    drift_filepath = utils_data.get_logging_path("DRIFT_MODEL")
    logger.debug(f"Drift filepath {drift_filepath}")
    file_exists = os.path.isfile(drift_filepath)
    with open(drift_filepath, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        # Si le fichier n'existe pas, on ajoute l'entête
        if not file_exists:
            logger.debug(f"Fichier de logging n'existe pas")
            logger.debug(f"Ecriture de l'entête {data.keys()}")
            writer.writerow(data.keys())
        else:
            logger.debug(
                "Le fichier de logging existe, stockage des résultats en fin de fichier")
        logger.debug(f"Ecriture des résultats {data.values()}")
        # Write the new data to the file
        writer.writerow(data.values())
    return unique_id


def save_drift_metrics_data(model_name, new_mean, original_mean, new_std, original_std, mean_diff, std_diff,
                            drift, drift_calculation_date, temps_calcul):
    ''' Mise à jour du fichier de prediction avec les informations recueillies'''
    logger.debug(
        f"-------save_drift_metrics_data(model_name={model_name}, new_mean={new_mean}, original_mean={original_mean}, new_std={new_std}, original_std={original_std}, drift={drift}, drift_calculation_date={drift_calculation_date}, temps_calcul={temps_calcul})----")
    # Créer un dictionnaire avec les informations
    unique_id = str(uuid.uuid4())
    data = {
        'UID': unique_id,
        'Nom du modèle': model_name,
        'New Mean': new_mean,
        'Original Mean': original_mean,
        'Original': original_mean,
        'New STD': new_std,
        'Original STD': original_std,
        'Mean Diff': mean_diff,
        'STD Diff': std_diff,
        'Drift': drift,
        'Date de calcul': drift_calculation_date,
        'Temps de calcul': temps_calcul
    }
    drift_filepath = utils_data.get_logging_path("DRIFT_DATA")
    logger.debug(f"Drift filepath {drift_filepath}")
    file_exists = os.path.isfile(drift_filepath)
    with open(drift_filepath, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        # Si le fichier n'existe pas, on ajoute l'entête
        if not file_exists:
            logger.debug(f"Fichier de logging n'existe pas")
            logger.debug(f"Ecriture de l'entête {data.keys()}")
            writer.writerow(data.keys())
        else:
            logger.debug(
                "Le fichier de logging existe, stockage des résultats en fin de fichier")
        logger.debug(f"Ecriture des résultats {data.values()}")
        # Write the new data to the file
        writer.writerow(data.values())
    return unique_id


def archive_model_pred_drift_logs(model_name):
    logger.debug(
        f"-------archive_model_pred_drift_logs(model_name={model_name})----")
    prediction_logging_filepath = utils_data.get_logging_path("PRED")
    drift_model_filepath = utils_data.get_logging_path("DRIFT_MODEL")
    drift_data_filepath = utils_data.get_logging_path("DRIFT_DATA")

    logger.debug(f"Prediction logging filepath {prediction_logging_filepath}")
    logger.debug(f"drift model logging filepath {drift_model_filepath}")
    logger.debug(f"drift data logging filepath {drift_data_filepath}")
    # Renommer le fichier de prédiction si le fichier existe
    if os.path.exists(prediction_logging_filepath):
        predict_log_filename = os.path.basename(prediction_logging_filepath)
        os.rename(prediction_logging_filepath,
                  os.path.join(init_paths["PRED_logging_folder"],
                               f"{model_name}_{predict_log_filename}"))

    # Renommer le fichier de Drift si le fichier existe
    if os.path.exists(drift_model_filepath):
        drift_filename = os.path.basename(drift_model_filepath)
        # Renommer le fichier de dérive
        os.rename(drift_model_filepath,
                  os.path.join(init_paths["model_drift_folder"],
                               f"{model_name}_{drift_filename}"))

    if os.path.exists(drift_data_filepath):
        drift_filename = os.path.basename(drift_data_filepath)
        # Renommer le fichier de dérive
        os.rename(drift_data_filepath,
                  os.path.join(init_paths["data_drift_folder"],
                               f"{model_name}_{drift_filename}"))


def get_confusion_matrix_from_run(run_id, artifact_name='confusion_matrix.csv'):
    """
    Récupère la matrice de confusion d'un run MLflow et la restructure en DataFrame.

    :param run_id: Identifiant du run MLflow.
    :param artifact_name: Nom de l'artefact contenant la matrice de confusion (par défaut 'confusion_matrix.csv').
    :return: DataFrame de la matrice de confusion.
    """
    run = client.get_run(run_id)
    artifacts_uri = run.info.artifact_uri

    confusion_matrix_path = os.path.join(artifacts_uri, artifact_name)

    # Télécharger le fichier CSV de la matrice de confusion
    df_confusion_matrix = pd.read_csv(confusion_matrix_path)

    return df_confusion_matrix


def get_classification_report_from_run(run_id, artifact_name='classification_report.json'):
    """
    Récupère le rapport de classification d'un run MLflow et le restructure en DataFrame.

    :param run_id: Identifiant du run MLflow.
    :param artifact_name: Nom de l'artefact contenant le rapport de classification (par défaut 'classification_report.json').
    :return: DataFrame du rapport de classification.
    """
    run = client.get_run(run_id)
    artifacts_uri = run.info.artifact_uri

    classification_report_path = os.path.join(artifacts_uri, artifact_name)
    logger.debug(f"classification_report_path {classification_report_path}")
    if classification_report_path.startswith('file://'):
        classification_report_path = classification_report_path[7:]

    logger.debug(f"classification_report_path {classification_report_path}")

    # Télécharger le fichier JSON du rapport de classification

    with open(classification_report_path, 'r') as f:
        classification_report = json.load(f)

    # Convertir le rapport de classification en DataFrame
    df_classification_report = pd.DataFrame(classification_report).transpose()

    return df_classification_report
