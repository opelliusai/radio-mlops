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
from src.models import model_drift_detection
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


@app.get("/drift_metrics", summary="Calcul des Drift metrics", description="Lancement du calcul des métriques pour la détection de drift")
async def drift_metrics():
    # 1 - Lancement du calcul des métriques
    model_name, new_mean, original_mean, new_std, original_std, mean_diff, std_diff, drift = model_drift_detection.drift_detection_main()
    logger.debug(
        f"Data : {model_name, new_mean, original_mean, new_std, original_std, mean_diff, std_diff, drift}")
    # Mettre à jour les métriques Prometheus
    MEAN_DIFF_GAUGE.set(mean_diff)
    STD_DIFF_GAUGE.set(std_diff)

    # Log des nouvelles metriques

    return JSONResponse(status_code=200, content={"status": "Calcul terminé",
                                                  "model_name": model_name,
                                                  "new_mean": new_mean.tolist(),
                                                  "original_mean": original_mean.tolist(),
                                                  "new_std": new_std.tolist(),
                                                  "original_std": original_std.tolist(),
                                                  "mean_diff": float(mean_diff),
                                                  "std_diff": float(std_diff),
                                                  "drift": drift
                                                  })

# Lancement de l'API sur le port associé
if __name__ == "__main__":
    # Initialisation des chemins d'accès aux fichiers et dossiers
    uvicorn.run("monitoring_api:app", host="0.0.0.0", port=8083, reload=True)
