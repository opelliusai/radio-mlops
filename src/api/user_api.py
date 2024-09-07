'''
Créé le 07/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: API pour la mise à disposition de services aux utilisateurs
-- Gestion du compte
    -- Inscription (à venir) 
    -- Authentification (Avec Profil simple ou admin)
    -- Modification du mot de passe (à venir)
    -- Suppression du compte (à venir)
-- Prédiction
    -- Historique des prédictions de l'utilisateur (à venir)
    -- Exécution d'une prédiction et visualisation du résultat avec indice de confiance
    -- Action: Valider/Invalider/Modifier la prédiction
'''

# IMPORTS
# Imports externes
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status, File, UploadFile, Header
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Summary, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from datetime import datetime
import jwt
import os
import uvicorn
import json
from pydantic import BaseModel
from typing import Dict
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Imports internes
from src.config.run_config import init_paths, model_info, user_mgt
from src.config.log_config import setup_logging
from src.utils import utils_models, utils_data
from src.datasets import update_dataset
from src.models.predict_model import mlflow_predict_and_log
# Redirection vers le fichier de log radio-mlops_user_api.log
logger = setup_logging("user_api")

# Initialisation de l'API FastAPI avec des informations de base
app = FastAPI(
    title="Détection d'anomalies Pulmonaires",
    description="API pour la détection d'anomalies Pulmonaires COVID et Pneumonia Virale.",
    version="0.1"
)

# Gestion des utilisateurs
# Chargement des variables d'environnement
load_dotenv()

# Variables d'environnement
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Charger les utilisateurs depuis le fichier JSON


def load_users() -> Dict[str, Dict]:
    try:
        with open(users_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


users_file = os.path.join(
    init_paths["main_path"], init_paths["streamlit_cache_folder"], user_mgt["user_filename"])
logger.debug(f"Chemin users_file {users_file}")

# Chargement des users
USERS = load_users()
logger.debug(f"Users {USERS}")
# Modèle Pydantic pour le token


class Token(BaseModel):
    access_token: str
    token_type: str

# Fonction pour créer un token JWT


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Fonction pour vérifier le token JWT
def verify_token(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/token"))):
    try:
        logger.info(f"Tentative de décodage du token: {token}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in USERS:
            logger.warning("Le token ne contient pas de 'sub' valide")
            raise HTTPException(
                status_code=401, detail="Could not validate credentials")
        logger.info(f"Token validé pour l'utilisateur: {username}")
        return username
    except jwt.PyJWTError as e:
        logger.error(f"Erreur lors de la validation du token: {str(e)}")
        raise HTTPException(
            status_code=401, detail="Could not validate credentials")

# Fonction pour vérifier la clé API


def verify_api_key(api_key: str = Header(..., alias="api-key")):
    if api_key != API_KEY:
        logger.warning("Tentative d'accès avec une clé API invalide")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    logger.info("Clé API validée")
    return api_key

# Route pour obtenir un token


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username not in USERS or form_data.password != ADMIN_PASSWORD:
        logger.warning(
            f"Tentative de connexion échouée pour l'utilisateur: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    logger.info(f"Connexion réussie pour l'utilisateur: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}


class LoginRequest(BaseModel):
    username: str
    password: str


# Sauvegarder les utilisateurs dans le fichier JSON


def save_users(users: Dict[str, Dict]):
    with open(users_file, "w") as f:
        json.dump(users, f)


###########
# Fin de gestion des utilisateurs


# Création des chemins d'accès aux fichiers et dossiers
# Nom du modèle à utiliser
# model_path=os.path.join(init_paths["main_path"],init_paths["models_path"],model_info["selected_model_name"])
img_storage_path = os.path.join(
    init_paths["main_path"], init_paths["PRED_images_folder"])
prediction_logging_filepath = os.path.join(
    init_paths["main_path"], init_paths["PRED_logging_folder"], model_info["PRED_logging_filename"])

'''
## 3 - Chargement du modèle une fois
model_name=model_info["selected_model_name"]
#model=utils_models.load_model(model_path)
model = utils_models.get_mlflow_prod_model()
## 4 - Charger les variables d'environnement
#load_dotenv()
'''

# Chargement des variables d'environnement
# load_dotenv()

# Définition des métriques Prometheus
collector = CollectorRegistry()
USER_API_REQUEST_TIME = Summary(
    'MLOps_user_api_request_processing_seconds', 'Time spent processing request', registry=collector)
USER_API_REQUEST_COUNT = Counter(name='MLOps_user_api_request_count', labelnames=[
                                 'method', 'endpoint'], documentation='Total number of requests', registry=collector)
USER_API_DURATION_OF_REQUESTS = Histogram(name='MLOps_user_api_duration_of_requests', documentation='duration of requests per method or endpoint',
                                          labelnames=['method', 'endpoint'], registry=collector)

# Dépendance pour mesurer le temps et compter les requêtes


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start_time = time.time()
    method = request.method
    endpoint = request.url.path

    # Incrémente le compteur pour chaque requête
    USER_API_REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    # Exécute la requête réelle
    response = await call_next(request)
    # Mesure la durée de la requête
    duration = time.time() - start_time

    # Observe le temps de traitement de la requête
    USER_API_REQUEST_TIME.observe(duration)
    USER_API_DURATION_OF_REQUESTS.labels(
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


# Login
'''
@app.post("/login", summary="")
async def login(request: LoginRequest):
    user = USERS.get(request.username)
    if user and user["password"] == request.password:
        return {"username": user["username"], "role": user["role"]}
    else:
        raise HTTPException(status_code=401, detail="Identifiants incorrects")
'''


@app.post("/login", summary="")
async def login(login_request: OAuth2PasswordRequestForm = Depends()):
    user = USERS.get(login_request.username)
    logger.debug(f"user {user}")
    if user and user["password"] == login_request.password:
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": login_request.username}, expires_delta=access_token_expires
        )
        logger.info(
            f"Connexion réussie pour l'utilisateur: {login_request.username} / Acces token {access_token} / User name {user['user_name']}")
        return {"status_code": status.HTTP_200_OK, "access_token": access_token, "username": user["username"], "user_name": user["user_name"], "role": user["role"], "token_type": "bearer"}
    else:
        logger.warning(
            f"Tentative de connexion échouée pour l'utilisateur: {login_request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants incorrects",
            headers={"WWW-Authenticate": "Bearer"}
        )
# Modèle Pydantic pour le token


class Token(BaseModel):
    access_token: str
    token_type: str

# Fonction pour créer un token JWT


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Fonction pour vérifier le token JWT


def verify_token(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/token"))):
    try:
        logger.info(f"token {token}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in USERS:
            logger.warning("Le token ne contient pas de 'sub' valide")
            raise HTTPException(
                status_code=401, detail="Impossible de valider l'accès")
        logger.info(f"Token validé pour l'utilisateur: {username}")
        return username
    except jwt.PyJWTError as e:
        logger.error(f"Erreur lors de la validation du token: {str(e)}")
        raise HTTPException(
            status_code=401, detail="Impossible de valider l'accès")

# Prediction


@app.post("/predict", summary="Prédiction sur une image", description="Evaluation de l'état pulmonaire basé sur une image")
async def predict(image: UploadFile = File(...)):
    # Temporairement sans authentification
    logger.debug("---------------user_api: /predict---------------")
    image_original_name = image.filename
    # construction du nom du fichier basé sur la date courante
    logger.debug(f"Image reçue: {image_original_name}")
    logger.debug("Construction du nom du fichier basé sur la date courante ")
    current_time = datetime.now()
    # Formater la date et l'heure pour les inclure dans le nom du fichier
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    # Construire le chemin avec le nouveau nom de fichier
    image_path = os.path.join(
        img_storage_path, f"upload_{formatted_time}_{image_original_name}")
    logger.debug(f"Chemin complet du fichier en local {image_path}")
    # Enregistrer l'image dans le dossier des Upload
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())
        logger.debug(f"Image sauvegardée à: {image_path}")
    # Exécuter la prédiction
    model_name, prediction, confiance, temps_prediction, pred_id = mlflow_predict_and_log(
        image_path)
    return JSONResponse(status_code=200, content={"status": "OK", "model_name": model_name, "prediction": prediction, "confiance": confiance, "temps_prediction": temps_prediction, "image_upload_path": image_path, "pred_id": pred_id})

# Ajout d'une image et de son label


@app.post("/add_image", summary="", description="")
async def add_image(image_path: str, label: str):
    status = update_dataset.add_one_image(image_path, label)
    return JSONResponse(status_code=200, content={"status": status})

# Liste des classes


@app.get("/get_classes")
async def get_classes():
    return {"classes": utils_data.get_classes()}

# Mise à jour des log de prédiction avec le retour utilisateur


@app.post("/update_log_prediction", summary="Met à jour le log d'une prédiction avec le retour utilisateur", description="Met à jour le log d'une prédiction avec le retour utilisateur")
async def update_log_prediction(pred_id: str,
                                label: str):
    logger.debug("API / update_log_prediction")
    logger.debug(f"pred_id: {pred_id} / label:{label}")
    status = utils_models.update_log_prediction(pred_id, label)
    return JSONResponse(status_code=200, content={"status": status})


if __name__ == "__main__":
    # Lancement de l'API sur le port associé
    uvicorn.run("user_api:app", host="0.0.0.0", port=8081, reload=True)
