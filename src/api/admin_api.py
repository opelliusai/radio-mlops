'''
Créé le 07/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary:API d'administration
-- Etend l'API utilisateur avec des fonctionnalités d'administration
-- Gestion des datasets: téléchargement, nettoyage, mise à jour
-- Modèles: Entrainement, réentrainement,
déploiement ou préparation au déploiement
'''

# IMPORTS
# Imports externes
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, Summary
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import time
import uvicorn
from pydantic import BaseModel
# Imports internes
from src.utils import utils_data, utils_models
from src.datasets import download_dataset, update_dataset
from src.mlflow import model_tracking, model_serving
from src.config.log_config import setup_logging
from src.config.run_config import model_hp
# Redirection vers le fichier de log radio-mlops_admin_api.log
logger = setup_logging("admin_api")

# Initialisation de l'API FastAPI avec des informations de base
app = FastAPI(
    title="Détection d'anomalies Pulmonaires - API Admin",
    description="API d'administration.",
    version="0.1"
)

# Chargement des variables d'environnement
# load_dotenv()

# Définition des métriques Prometheus
collector = CollectorRegistry()
ADMIN_API_REQUEST_TIME = Summary(
    'MLOps_admin_api_request_processing_seconds',
    'Time spent processing request',
    registry=collector)
ADMIN_API_REQUEST_COUNT = Counter(name='MLOps_admin_api_request_count',
                                  labelnames=['method', 'endpoint'],
                                  documentation='Total number of requests',
                                  registry=collector)
ADMIN_API_DURATION_OF_REQUESTS = Histogram(
    name='MLOps_admin_api_duration_of_requests',
    documentation='duration of requests per method or endpoint',
    labelnames=['method', 'endpoint'],
    registry=collector)

# Dépendance pour mesurer le temps et compter les requêtes


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start_time = time.time()
    method = request.method
    endpoint = request.url.path

    # Incrémente le compteur pour chaque requête
    ADMIN_API_REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    # Exécute la requête réelle
    response = await call_next(request)
    # Mesure la durée de la requête
    duration = time.time() - start_time

    # Observe le temps de traitement de la requête
    ADMIN_API_REQUEST_TIME.observe(duration)
    ADMIN_API_DURATION_OF_REQUESTS.labels(
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


@app.get("/",
         summary="Health Check",
         description="Vérification de l'état de l'application")
async def health_check():
    return JSONResponse(status_code=200, content={"status": "OK"})


# Gestion des utilisateurs
# Endpoint pour ajouter un utilisateur (optionnel)
class AddUserRequest(BaseModel):
    username: str
    password: str
    role: str

# Lancer un téléchargement des données


@app.get("/download_dataset",
         summary="Téléchargement du dataset Kaggle",
         description="Téléchargement du dataset Kaggle")
async def download_new_dataset():
    # Télécharger le dataset
    download_dataset.download_dataset_main()
    return JSONResponse(status_code=200,
                        content={"status": "Téléchargement terminé"})

# Mise à jour du Dataset de référence avec des données de PROD ou KAGGLE


@app.post("/update_dataset",
          summary="Mise à jour du dataset de référence",
          description="Données de PROD ou de KAGGLE.")
async def update_dataset_ref(dataset_path: str = None,
                             source_type: str = "KAGGLE",
                             # mise à jour par défaut avec des dataset kaggle
                             base_dataset_id: int = None):
    update_dataset.update_dataset_ref(
        dataset_path, source_type, base_dataset_id)
    return JSONResponse(status_code=200,
                        content={"status": "Mise à jour terminée"})

# Entrainement d'un modèle


@app.post("/train_model", summary="Entrainement d'un modèle",
          description="Entrainement d'un modèle avec plusieurs options")
def train_model(retrain: bool = False,
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


# Mise à jour du tag d'un modèle pour le rendre prêt à être déployé


@app.post("/make_model_prod_ready",
          summary="Rendre un modèle prêt pour la PROD",
          description="Préparation du modèle déploiement au prochain batch")
def make_model_prod_ready(num_version):
    model_serving.make_prod_ready(num_version=num_version)
    return JSONResponse(status_code=200,
                        content={"status": "Mise à jour terminée"})

# Déploiement d'un modèle prêt pour la PRODUCTION


@app.post("/deploy_ready_model",
          summary="Déployer un modèle prêt pour la PROD",
          description="Déployer un modèle qui a un tag prêt pour la PROD")
def deploy_ready_model():
    model_serving.auto_model_serving()
    return JSONResponse(status_code=200,
                        content={"status": "Déploiement en PROD terminé"})

# Forcer le déploiement d'un modèle en PRODUCTION


@app.post("/force_model_serving",
          summary="Déployer un modèle prêt pour la PROD",
          description="Déployer un modèle qui a un tag prêt pour la PROD")
def force_model(num_version: str):
    model_serving.model_version_serving(num_version=num_version)
    return JSONResponse(status_code=200,
                        content={"status": "Déploiement terminé"})

# Liste et informations sur les modèles disponibles


@app.get("/get_models_list",
         summary="Liste tous les modèles disponibles",
         description="Liste tous les modèles disponibles")
def get_models_list():
    list_models = utils_models.get_models()
    logger.debug(f"API ADMIN list_models {list_models}")

    return JSONResponse(status_code=200,
                        content={"status": "OK",
                                 "list_models": list_models})

# Liste et informations sur les runs MLFlow


@ app.post("/get_runs_info",
           summary="Récupérer les informations sur un ou plusieurs run",
           description="Récupérer les informations des RUNs MLFlow")
def get_runs_info(run_ids: list):
    runs_info = utils_models.get_runs_info(run_ids)
    return JSONResponse(status_code=200,
                        content={"status": "OK", "runs_info": runs_info})

# Liste et informations sur les datasets disponibles


@ app.post("/get_datasets_list",
           summary="Lister les datasets disponibles",
           description="Lister les datasets disponibles")
def get_datasets_list(type: str = "REF"):
    list_datasets = utils_data.get_all_dataset_info(type)
    logger.debug(f"list_datasets {list_datasets}")
    data = list_datasets.to_dict(orient='records')

    return JSONResponse(status_code=200,
                        content={"status": "OK", "list_datasets": data})


# Ajout d'un batch d'images et leur label
# Utile pour l'administrateur pour charger un lot de données externes


@ app.post("/add_images", summary="", description="")
async def add_images(images_labels: list):  # dict chemin/label
    ds_old_version, ds_new_version = update_dataset.add_mutliple_images(
        images_labels)
    return JSONResponse(status_code=200,
                        content={"status": "OK",
                                 "old_dataset_version": ds_old_version,
                                 "new_dataset_version": ds_new_version})

if __name__ == "__main__":
    # Lancement de l'API sur le port associé
    uvicorn.run("admin_api:app", host="0.0.0.0", port=8082, reload=True)
