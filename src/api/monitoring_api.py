'''
Créé le 20/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: API Drift Monitoring
-- Envoi des résultats de prédiction à Prometheus
-- Ensuite, on ajoute des seuils d'alerte côté prometheus
'''

# IMPORTS
# Imports externes
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
import time
import uvicorn
# Imports internes
from src.config.log_config import setup_logging
from src.config.run_config import model_hp
from src.models import data_drift_detection, model_drift_detection
from src.mlflow import model_serving, model_tracking
# Redirection vers le fichier de log radio-mlops_monitoring_api.log
logger = setup_logging("monitoring_api")

# Initialisation de l'API FastAPI avec des informations de base
app = FastAPI(
    title="Détection d'anomalies Pulmonaires - API de Gestion d'alertes",
    description="API de Gestion d'alertes",
    version="0.1"
)

# Définition des métriques Prometheus
collector = CollectorRegistry()
MONITORING_API_REQUEST_TIME = Summary(
    'MLOps_monitoring_api_request_processing_seconds', 'Time spent processing request', registry=collector)
MONITORING_API_REQUEST_COUNT = Counter(name='MLOps_monitoring_api_request_count', labelnames=[
                                       'method', 'endpoint'], documentation='Total number of requests', registry=collector)
MONITORING_API_DURATION_OF_REQUESTS = Histogram(name='MLOps_monitoring_api_duration_of_requests', documentation='duration of requests per method or endpoint',
                                                labelnames=['method', 'endpoint'], registry=collector)

MEAN_DIFF_GAUGE = Gauge(
    'mean_diff', 'Difference between new mean and original mean', registry=collector)
STD_DIFF_GAUGE = Gauge(
    'std_diff', 'Difference between new std and original std', registry=collector)

# Dépendance pour mesurer le temps et compter les requêtes


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start_time = time.time()
    method = request.method
    endpoint = request.url.path

    # Incrémente le compteur pour chaque requête
    MONITORING_API_REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    # Exécute la requête réelle
    response = await call_next(request)
    # Mesure la durée de la requête
    duration = time.time() - start_time

    # Observe le temps de traitement de la requête
    MONITORING_API_REQUEST_TIME.observe(duration)
    MONITORING_API_DURATION_OF_REQUESTS.labels(
        method=method, endpoint=endpoint).observe(duration)

    return response

# ENDPOINTS
# Exposition des métriques pour PROMETHEUS


@app.get("/metrics")
async def metrics():
    data = generate_latest(collector)
    print(f"data {data}")
    logger.debug(f"data {data}")
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# Health Check


@app.get("/", summary="Health Check", description="Vérification de l'état de l'application")
async def health_check():
    return JSONResponse(status_code=200, content={"status": "OK"})

# Lancement du calcul des métriques pour le drift detection

# Déploiement d'un modèle prêt pour la PRODUCTION


@app.post("/deploy_ready_model",
          summary="Déployer un modèle prêt pour la PROD",
          description="Déployer un modèle qui a un tag prêt pour la PROD")
async def deploy_ready_model():
    model_name, model_version = model_serving.auto_model_serving()
    return JSONResponse(status_code=200,
                        content={"status": "Déploiement en PROD terminé",
                                 "model_name": model_name,
                                 "model_version": model_version})


@app.post("/drift_metrics", summary="Calcul des Drift metrics", description="Lancement du calcul des métriques pour la détection de drift")
async def drift_metrics(retrain: bool = False):
    # 1 - Lancement du calcul des métriques
    model_name, new_mean, original_mean, new_std, original_std, mean_diff, std_diff, drift = data_drift_detection.drift_detection_main()
    logger.debug(
        f"Data Calcul Drift : {model_name, new_mean, original_mean, new_std, original_std, mean_diff, std_diff, drift}")
    # Mettre à jour les métriques Prometheus
    MEAN_DIFF_GAUGE.set(mean_diff)
    STD_DIFF_GAUGE.set(std_diff)
    content = {"status": "Calcul terminé",
               "model_name": model_name,
               "drift": drift,
               # "new_mean": new_mean.tolist(),
               # "original_mean": original_mean.tolist(),
               # "new_std": new_std.tolist(),
               # "original_std": original_std.tolist(),
               "mean_diff": float(mean_diff),
               "std_diff": float(std_diff),
               # "status_retrain_diff": "N/A",
               # "diff_run_id": None,
               # "diff_model_version": None,
               # "diff_experiment_link": None,
               "status_retrain_comb": "N/A",
               "comb_run_id": None,
               "comb_model_version": None,
               "comb_experiment_link": None
               }
    if retrain == True and drift == True:

        logger.debug(
            f"Détection de drift - Réentrainement du modèle de Production")
        # diff_run_id, model_name, diff_model_version, diff_experiment_link = model_tracking.main(
        #    retrain=True)
        comb_run_id, model_name, comb_model_version, comb_experiment_link = model_tracking.main(
            include_prod_data=True)
        # combiné
        retrain_content = {  # "status_retrain_diff": "Réentrainement du modèle avec les données de production terminé",
            # "diff_run_id": diff_run_id,
            # "diff_model_version": diff_model_version,
            # "diff_experiment_link": diff_experiment_link,
            "status_retrain_comb": "Entrainement du modèle avec les données de référence et production terminé",
            "comb_run_id": comb_run_id,
            "comb_model_version": comb_model_version,
            "comb_experiment_link": comb_experiment_link
        }
        logger.debug(
            f"Data Retrain/Train Model : {retrain_content}")

        content.update(retrain_content)
    logger.debug(f"Contenu final")
    logger.debug(f"{content}")

    return JSONResponse(status_code=200,
                        content=content)


@app.post("/model_drift_metrics", summary="Calcul des Drift metrics", description="Lancement du calcul des métriques pour la détection de drift")
async def model_drift_metrics(retrain: bool = False):
    # 1 - Lancement du calcul des métriques
    recall_mean_diff, drift = model_drift_detection.drift_detection_main()
    logger.debug(
        f"Data Calcul Drift : {recall_mean_diff, drift}")
    # Mettre à jour les métriques Prometheus
    content = {"status": "Calcul terminé",
               "drift": drift,
               "recall_mean_diff": recall_mean_diff,
               "status_retrain_comb": "N/A",
               "comb_run_id": None,
               "comb_model_version": None,
               "comb_experiment_link": None
               }
    if retrain == True and drift == True:

        logger.debug(
            f"Détection de Model drift - Réentrainement du modèle de Production")
        # diff_run_id, model_name, diff_model_version, diff_experiment_link = model_tracking.main(
        #    retrain=True)
        comb_run_id, model_name, comb_model_version, comb_experiment_link = model_tracking.main(
            include_prod_data=True)
        # combiné
        retrain_content = {  # "status_retrain_diff": "Réentrainement du modèle avec les données de production terminé",
            # "diff_run_id": diff_run_id,
            # "diff_model_version": diff_model_version,
            # "diff_experiment_link": diff_experiment_link,
            "status_retrain_comb": "Entrainement du modèle avec les données de référence et production terminé",
            "comb_run_id": comb_run_id,
            "comb_model_version": comb_model_version,
            "comb_experiment_link": comb_experiment_link
        }
        logger.debug(
            f"Data Retrain/Train Model : {retrain_content}")

        content.update(retrain_content)
    logger.debug(f"Contenu final")
    logger.debug(f"{content}")

    return JSONResponse(status_code=200,
                        content=content)


@app.post("/data_drift_metrics", summary="Calcul des Drift metrics", description="Lancement du calcul des métriques pour la détection de drift")
async def data_drift_metrics(retrain: bool = False):
    # 1 - Lancement du calcul des métriques
    model_name, new_mean, original_mean, new_std, original_std, mean_diff, std_diff, drift = data_drift_detection.drift_detection_main()
    logger.debug(
        f"Data Calcul Drift : {model_name, new_mean, original_mean, new_std, original_std, mean_diff, std_diff, drift}")
    # Mettre à jour les métriques Prometheus
    MEAN_DIFF_GAUGE.set(mean_diff)
    STD_DIFF_GAUGE.set(std_diff)
    content = {"status": "Calcul terminé",
               "model_name": model_name,
               "drift": drift,
               # "new_mean": new_mean.tolist(),
               # "original_mean": original_mean.tolist(),
               # "new_std": new_std.tolist(),
               # "original_std": original_std.tolist(),
               "mean_diff": float(mean_diff),
               "std_diff": float(std_diff),
               # "status_retrain_diff": "N/A",
               # "diff_run_id": None,
               # "diff_model_version": None,
               # "diff_experiment_link": None,
               "status_retrain_comb": "N/A",
               "comb_run_id": None,
               "comb_model_version": None,
               "comb_experiment_link": None
               }
    if retrain == True and drift == True:

        logger.debug(
            f"Détection de drift - Réentrainement du modèle de Production")
        # diff_run_id, model_name, diff_model_version, diff_experiment_link = model_tracking.main(
        #    retrain=True)
        comb_run_id, model_name, comb_model_version, comb_experiment_link = model_tracking.main(
            include_prod_data=True)
        # combiné
        retrain_content = {  # "status_retrain_diff": "Réentrainement du modèle avec les données de production terminé",
            # "diff_run_id": diff_run_id,
            # "diff_model_version": diff_model_version,
            # "diff_experiment_link": diff_experiment_link,
            "status_retrain_comb": "Entrainement du modèle avec les données de référence et production terminé",
            "comb_run_id": comb_run_id,
            "comb_model_version": comb_model_version,
            "comb_experiment_link": comb_experiment_link
        }
        logger.debug(
            f"Data Retrain/Train Model : {retrain_content}")

        content.update(retrain_content)
    logger.debug(f"Contenu final")
    logger.debug(f"{content}")

    return JSONResponse(status_code=200,
                        content=content)


@ app.post("/train_model", summary="Entrainement d'un modèle",
           description="Entrainement d'un modèle avec plusieurs options")
async def train_model(retrain: bool = False,
                      model_name: str = None,
                      model_version: str = None,
                      include_prod_data: bool = False,
                      balance: bool = True,
                      dataset_version: str = None,  # Par défaut
                      max_epochs: int = model_hp["max_epochs"],
                      num_trials: int = model_hp["num_trials"]):
    """
    Scénarios possibles:
    1 - Entrainement from scratch / Dataset définir
    Option d'entrainement d'un modèle:
    1. Type d'entrainement:
    - Entrainement from scratch  > retrain = false, model_name = None
        - Dataset de Reference include_prod_data=False
        - Dataset de Reference et données de production include_prod_data=True
    - Réentrainement d'un modèle
        - Avec les données de production

    2. Version du dataset de Référence:
        - Par défaut : Dernière version du Dataset
        - Version du dataset = Nom complet du dataset (basé sur le menu déroulant proposé)

    3. Parametres: Nombre d'epochs et max trials


    # zero: from scracth with current dataset
    # combined : from scracth with New + current data
    # diff : Retrain current model with new data only
    """
    # run_id, model_name, model_version = model_tracking.main(
    # dataset_version, max_epochs, num_trials,)
    run_id, model_name, model_version, experiment_link = model_tracking.main(
        retrain, model_name, model_version, include_prod_data,
        balance, dataset_version, max_epochs, num_trials)
    return JSONResponse(status_code=200,
                        content={"status": "Entrainement terminé",
                                 "run_id": run_id,
                                 "model_name": model_name,
                                 "model_version": model_version,
                                 "experiment_link": experiment_link})


# Lancement de l'API sur le port associé
if __name__ == "__main__":
    # Initialisation des chemins d'accès aux fichiers et dossiers
    uvicorn.run("monitoring_api:app", host="0.0.0.0", port=8083, reload=True)
