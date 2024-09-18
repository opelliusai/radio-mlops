# RADIO-MLOps

Ce repository concerne le projet de Radiographie Pulmonaire dans le cadre de la formation MLOps de Datascientest

## Accès rapide
- [Le projet](#le-projet)
- [Architecture applicative](#architecture-applicative)
- [Applications et services](#applications-et-services)
- [Arborescence du projet](https://github.com/opelliusai/test_readme/blob/main/README.md#arborescence-du-projet)
- [Instructions d'utilisation](#instructions)
---
## Le projet
**Contexte**

Début 2020, la propagation rapide du coronavirus (COVID-19) a entravé la capacité des systèmes de santé à réaliser les diagnostics et tests requis dans les délais imposés par la pandémie. Ainsi, une recherche active de solutions alternatives pour le dépistage a été initiée.
En raison des effets significatifs du COVID-19 sur les tissus pulmonaires, l'usage de l'imagerie par radiographie thoracique s'est avéré incontournable pour le dépistage et le suivi de la maladie lors de la crise COVID

**Objectif du client en 2024**
- Disposer d'un outil de diagnostic tout en conservant un taux élevé de détection COVID, mais sans négliger d'autres pathologies, en particulier la Pneumonie Virale
- Mettre à disposition de l'outil aux internes pour établir un premier diagnostic
- Mettre à contribution les Radiologues référents pour la validation du diagnostic

L'outil est donc à la fois de diagnostic mais aussi pédagogique

**Use Case utilisateurs métier**

Les utilisateurs métiers disposent d'une application web pour charger les images et obtenir un diagnostic.<br>
Ils peuvent également contribuer à la validation de diagnostics pour améliorer le modèle

![](/references/MLOps_Utilisateurs_metier_workflow.png)

**Use Case administrateurs et datascientists**

Les utilisateurs dits techniques disposent de plusieurs outils pour administrer et gérer l'application et les modèles
- Une interface de suivi des performances du modèle
- Des outils annexes pour la gestion des opérations: MLFlow, Airflow, Prometheus et Grafana

Ces outils sont proposés aux utilisateurs techniques pour gérer les différentes étapes d'un produit de Machine Learning
![](/references/MLOps_Administrateurs_workflow.png)

---
## Architecture applicative

![](/references/Archi_Docker.png)
Plusieurs conteneurs Dockers sont créés pour le projet avec 3 groupement principaux:
- **APIs**: 3 APIs développés avec FastAPI
- **FrontEnd principal**: 1 application web streamlit qui s'interface exclusivement avec les apis via un module utilitaire (utils_streamlit)
- **1 groupe d'applications annexes**:
  - **MLFlow** :  Gestion du cycle de vie des modèles
  - **Airflow** : Ordonnancement
  - **Prometheus** : Etat de service des APIs et leur système via node_exporter
  - **Grafana**: Possibilité de créer des dashboard (actuellement connecté uniquement à Prometheus et son node_exporter)

**Gestion des données - Metadata et versioning:**

![](/references/Gestion_donnees.png)
3 types de données:
- **Données Kaggle**: Les données source initiales qui servent à entrainer les premières modèles
- **Données de Production**: Les données enrichies par les utilisateurs lors des demandes de diagnostic
- **Données de Référence**: L'agrégation des données et qui sont effectivement utilisées pour entrainer les modèle

**Où sont stockées les données**
- Historique des versions des données: *processed/dataset_logging*
- Données versionnées:
  - **kaggle**: *raw/kaggle_datasets*
  - **reference**: *processed/ref_datasets*
  - **prod**: *processed/prod_dataset*s.

**Ce qu'il faut retenir**
- Pour entrainer les modèles, seules les données de référence sont utilisées. Les données Kaggle et prod alimentent ces données après une mise à jour par l'administrateur ou par un batch (non configuré ici).
- Un équilibrage est réalisé sur la base de la classe minimale. Dans ce cas, un fichier metadata contenant la liste des images réellement utilisée du dataset de référence est créé et associé au nouveau modèle.
  Répertoire contenant les metadata: *data/processed/models_data/MLOps_Radio_Model-$version/RadioPulmonaire_REF-<version_data>_MLOps_Radio_Model-<version_model>_metadata*
  exemple : data/processed/models_data/MLOps_Radio_Model-$version/RadioPulmonaire_REF-1.0_MLOps_Radio_Model-2_metadata.csv
  Ce fichier est accessible dans les Artifacts du Run d'entrainement, en plus des informations du dataset complet

  #### Artefact (avec informations après équilibrage des données)
<img src="/references/exemple_artefact_run.png" alt="" width="400"/>

**Gestion des modèles - MLflow:**
Le cycle de vie des modèles est géré dans MLFlow:
- **Entrainement**: Les hyperparamètres et métriques sont enregistrés. Le versioning des modèles est également géré avec MLFlow
- **Déploiement**: Déclaration d'un modèle en Production en mettant à jour sa phase en 'Production' et celle du modèle courant en 'Archived'. Basé sur un script qui détecte un tag 'deploy_tag' dont la valeur serait 'production_ready'
- **Prédiction**: Chargement du modèle de production (étant en phase 'Production')
- **Réentrainement**: Ajout des informations du modèle initial dans les informations du run associé au nouveau modèle.
  
![](/references/Gestion_modeles.png)

#### 
---
## Applications et services

**APIs :**

3 APIs ont été construites avec FastAPI et un swagger est disponible pour plus d'informations sur les services proposés (/docs)
- **API User _src/api/user_api_**: Disponible sur le port 8081.<br>
Elle est utilisée par l'application web et permet à l'utilisateur de : 
    -  Effectuer une demande de diagnostic.
    -  Faire un retour sur le diagnostic
    -  Contribuer à la labellisation des données dont le diagnostic n'a pas été validé par les utilisateurs 

- **API Admin _src/api/admin_api**: Disponible sur le port 8082.<br>
Elle est utilisée par l'application web et permet aux administrateurs de : 
    -  Lancer un téléchargement du dataset kaggle
    -  Lancer la mise à jour des données de référence avec des données KAGGLE ou de PRODUCTION
    -  Lancer l'entrainement d'un modèle
    -  Consulter les informations sur les modèles du projet et de préparer un modèle candidat au déploiement, ou de forcer son déploiement immédiat (en cas d'urgence par exemple)
 
- **API Monitoring _src/api/monitoring_api_**: Disponible sur le port 8083.<br>
Elle est dédiée aux action de monitoring :
    - Calcul des metriques de drift : Modèle et Data
    - Réentrainement du modèle de production si besoin (en option)

**Application Frontend - Streamlit :**
1 application web Streamlit a été développée pour les utilisateurs et les administrateurs, chacun ayant accès à des fonctionnalités en fonction de son profil/rôle. L'administrateur a accès aux fonctionnalités des utilisateurs métier en plus de ses fonctionnalités.

**Applications annexes :**
Les applications suivantes sont mise à disposition à l'administrateur pour la gestion du cycle de vie du produit.

- **Airflow :** Disponible sur le port 8093
    - **Healthcheck**: Vérification de l'état des services
    - **Deploy_model**: Déploiement de modèle prêts pour la production (Rappel : Tag 'deploy_flag' dont la valeur est 'production_ready')
    - **Data Drift**: Calcul des mesures pour la détection d'un Data drift et réentrainement du modèle si c'est le cas
    - **Model Drift**: Calcul des mesures pour la détection d'un Model drift et réentrainement du modèle si c'est le cas

- **Prometheus et Grafana :**
Des métriques sont envoyées par l'APIs et les node exporter installés dans leurs images docker.
L'administrateur a accès à Prometheus pour consulter ces métriques
Prometheus vérifie également l'état de services des 3 APIs
**Grafana** est connecté à **Prometheus** pour permettre à l'administrateur de créer ses Dashboards.
  
------------
## Applications et ports associés

| Application   | Port   | 
|-------------|-------------|
| User API | 8081 | 
| Admin API | 8082 | 
| Monitoring API | 8083 | 
| MLFlow | 8090 | 
| Prometheus | 8091 | 
| Grafana | 8092 | 
| Airflow | 8093 | 
| Streamlit | 8084 | 

## Arborescence du projet

    ├── LICENSE
    ├── README.md               <- README principal pour le projet
    ├── data                    <- Répertoire des données - local (dans .gitignore)
    │   ├── archives		<- Archivage 
    │   │
    │   ├── models			<- Les modèles 	
    │   │
    │   ├── processed	   	<- Les données après transformation 
    │   │   ├── dataset_logging     <- Fichiers sur l'historique des versions (ref, kaggle, prod)
    │   │   │
    │   │   ├── keras_tuner 	<- Répertoire dédié aux rapports de Keras Random Search
    │   │   │
    │   │   ├── mlflow_runs 	<- Répertoire dédié aux rapports générés lors de l'entrainement (history etc.)
    │   │   │
    │   │   ├── mlruns 		<- Répertoire utilisé au démarrage de MLFlow (Runs et Artifacts)
    │   │   │
    │   │   ├── model_drift 	<- Répertoire dédié aux model drifts
    │   │   │
    │   │   ├── data_drift 	        <- Répertoire dédié aux data drifts
    │   │   │
    │   │   ├── models_data 	<- Métadata des données utilisés par les modèles après équilibrage des classes pour une version donnée de dataset de référence
    │   │   │
    │   │   ├── predictions 	<- Fichiers de log des prédictions
    │   │   │
    │   │   ├── prod_datasets 	<- Datasets de Production avec versions
    │   │   │   
    │   │   ├── ref_datasets 	<- Datatsets de Référence avec versions
    │   │   │
    │   │   └── pytest_reports 	<- Rapports d'exécution des tests unitaires
    │   │
    │   ├── raw                	<- Les données brutes Kaggle 
    │   │   ├── kaggle_datasets  	<- Dataset Kaggle, téléchargé et versionné
    │   │   ├── prediction_images 	<- Répertoire de sauvegarde des images chargées par les utilisateurs (avant d'alimenter la base de données de production avec ou sans labellisation)
    │   │   ├── streamlit_assets 	<- Images, graphiques pour Streamlit
    │   │   └── test_images		<- Images de tests (pour les tests unitaires, depuis l'API ou depuis Streamlit)
    │   │
    │   │     
    ├── docker-compose                   <- Répertoire des fichiers pour la construction des images et démarrage des container dockers
    │   ├── docker-compose.yml           <- Le fichier regroupant toutes les images Docker à créer
    │   ├── docker-compose_<<nom_image>> <- Le fichier docker-compose dédié à chaque image
    │   └── Dockerfile_<<nom_image>>    	<- Le fichier Dockerfile dédié à chaque image
    │
    ├── requirements                     <- Répertoire des fichiers requirements par image
    │   ├── requirements.txt	     <- Fichier des requirements principal
    │   └── requirements_<<nom_image>>.txt <- Fichier des requirements par image, copié via le Dockerfile
    │
    ├── src                     <- Code source du projet
    │   ├── api                 <- APIs FastAPI
    │   │
    │   ├── config              <- Fichiers de configuration des paramètres et des logs
    │   │
    │   ├── datasets            <- Fonctions de traitement des bases de données
    │   │
    │   ├── mlflow              <- Fonctions de tracking et de serving des modèles sur MLFlow
    │   │
    │   ├── models              <- Fonction de construction, d'entrainement du modèle et de prédiction
    │   │
    │   ├── streamlit	    <- Application métier Streamlit
    │   │
    │   ├── streamlit_pres      <- Application projet Streamlit (Soutenance)
    │   │    
    │   ├── tests          	    <- Tests unitaires
    │   │
    │   └── utils               <- Fonctions utiles 

---
## Instructions

Les instructions sont listées par type d'usage souhaité par le développeur
- [Utiliser les images Dockerhub](#-utilisation-des-images-docker-sur-dockerhub)
- [Construire les images et démarrer les conteneurs Docker](#-construire-les-images-et-d%C3%A9marrer-les-conteneurs-docker)
- [Lancer les applications localement (Maintenance du code)](#lancer-les-applications-localement-maintenance-du-code)
- [Exemple de mise en place d'un environnement de machine learning](#exemple-de-mise-en-place-dun-environnement-de-machine-learning-pour-le-projet)

⚠️ : Les instructions Docker supposent que les pré-requis sont maitrisés:
- Installation de Docker sur son poste de développement ou de serveur de déploiement
- Compte Docker créé sur Dockerhub pour récupérer les images

### 🐳 **Utilisation des images Docker sur Dockerhub**
**Repository de référence:** [https://hub.docker.com/repositories/opelliusai](https://hub.docker.com/repositories/opelliusai)

### 1. Liste des images disponibles
| Image   | URL   | 
|-------------|-------------|
| User API | opelliusai/user_api | 
| Admin API | opelliusai/admin_api | 
| Monitoring API | opelliusai/monitoring_api | 
| MLFlow | opelliusai/mlflow | 
| Prometheus | opelliusai/prometheus | 
| Grafana | opelliusai/grafana | 
| Streamlit | opelliusai/streamlit | 

#### 2. Utilisation
Un fichier docker-compose est mis à disposition pour utiliser les dernières images disponibles sur dockerhub
**Localisation dans github**: docker-compose/docker-compose_using_dockerhub.yml

### 🐳 **Construire les images et démarrer les conteneurs Docker**
Le répertoire _docker-compose_ contient trois outils pour la construction et lancement des images:
Se positionner dans le répertoire docker-compose<br>
`cd docker-compose`

#### Construction et lancement de toutes les images avec docker-compose
En utilisant le fichier _docker-compose.yml_
**Construction**
`docker-compose build --no-cache`
**Lancement**
`docker-compose up -d`

#### Construction et lancement de toutes les images avec des scripts sh
**Construction**
`./docker-compose_all_build.sh`
**Lancement**
`./docker-compose_all_start.sh`

#### Construction et lancement image par image
En utilisant le fichier _docker-compose_<nom_image>.yml_
Exemple avec user_api
**Construction**
`docker-compose -f docker-compose_user_api.yml build --no-cache`
**Lancement**
`docker-compose -f docker-compose_user_api.yml up -d`

⚠️Ces deux types de constructions/démarrage n'incluent pas airflow qui doit être démarré séparément

### Installation Airflow
L'image Airflow n'est pas disponible sur dockerhub mais ses fichiers de lancement sont disponible sur github
**Localisation dans github**: docker-compose/docker-compose_airflow.yml
Plus d'informations sur airflow [ici](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
le fichier _docker-compose_airflow.yml_ est configuré pour utiliser le répertoire _../airflow_ (à la racine du projet github) comme base de travail de l'application Airflow 

#### 1. Renseigner les informations pour l'envoi des mails
Il faut également renseigner les informations pour l'envoi des mails (voir la section environment: et les variables AIRFLOW__SMTP__SMTP_*
```text
    AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK: 'true'
    AIRFLOW__SMTP__SMTP_HOST: your_host
    AIRFLOW__SMTP__SMTP_STARTTLS: True
    AIRFLOW__SMTP__SMTP_SSL: False
    AIRFLOW__SMTP__SMTP_USER: youremail@domain.com
    AIRFLOW__SMTP__SMTP_PASSWORD: your_password
    AIRFLOW__SMTP__SMTP_PORT: your_smtp_service_port
    AIRFLOW__SMTP__SMTP_MAIL_FROM: youremail@domain.com
    AIRFLOW__SMTP__SMTP_MAIL_TO: your_dest_email@domain.com
```
    
#### 2. Créer les répertoires et récupérer AIRFLOW_ID pour l'environnement Airflow
```text
cd airflow
mkdir -p ./dags ./logs ./plugins ./config
echo -e "AIRFLOW_UID=$(id -u)" > .env
```
#### 3. Initialisation de l'environnement
```text
docker-compose -f docker-compose_airflow.yml up airflow-init
```
#### 4. Démarrage d'Airflow
```text
docker-compose -f docker-compose_airflow.yml up -d
```

### Lancer les applications localement (Maintenance du code)

**Instructions:**

1. Mettre à jour src/config/run_config.py
Les APIs lancées localement ont des IPs 127.0.0.1 localhost
Par conséquent, remplacer le contenu de run_config.py >> LOCAL_run_config.py

2. Créer un environnement virtuel<br>
```text
python3 -m venv radioEnv
```
3. Activer l’environnement<br>
```text
source radioEnv/bin/activate
```
4. Ajouter le projet dans la variable PYTHONPATH<br>

5. Installer les pré-requis<br>
```text
pip install -r requirements.txt
```

6. Créer tous les répertoires nécessaires<br>
Ce module permet de créer tous les répertoires nécessaires pour le projet (qui ne sont pas dans le repository github
Voir structure du projet pour plus d'information
Techniquement, ce script se base sur le dictionnaire init_paths du fichier `src/config/run_config.py` pour créer les répertoire
Il est également intégré à toutes les images Docker pour s'assurer que l'arborescence est créée.

```text
python src/utils/utils_folders_init.py
```

7. Exécution locale des développements (API) <br>
**Depuis la racine du projet github **
| Image   | Commande   | 
|-------------|-------------|
| User API | `python src/api/user_api.py` | 
| Admin API | `python src/api/admin_api.py` | 
| Monitoring API | `python src/api/monitoring_api` | 

### Exemple de mise en place d'un environnement de Machine Learning pour le projet<br>
1. Téléchargement des données
- API Admin / Endpoint **/download_dataset**
- Les données seront stockées dans le volume docker-compose_shared-data
chemin data/raw/kaggle_dataset/
et versionnées sous le format COVID-19_Radiography_Dataset-X.Y (X.Y étant la dernière version mineure)
- Le log de téléchargement seront stockés dans 
data/processed/dataset_logging/kaggle_dataset_logging.csv

2. Construction des données de référence
-- API Admin / Endpoint **/update_dataset**
source_type="KAGGLE"
- Les données seront stockées dans le volume *docker-compose_shared-data*
chemin: data/processed/ref_dataset/
et versionnées sous le format  RadioPulmonaire-X.Y (X.Y étant la dernière version par incrémentation mineure (ex : 1.0 > 1.1)
- Le log de téléchargement seront stockés dans 
data/processed/dataset_logging/ref_dataset_logging.csv

3. Entrainement de modèles
-- API Admin ou Streamlit Page Entrainement/Reentrainement de modele
Entrainement initiale : 
-- API: Options par défaut
-- Streamlit:
Option "Entrainement complet (Données de référence uniquement)
- Sélection du Dataset
- Hyperparamètre: Max Epochs / Num trials

4. Déploiement d'un modèle initial
-- Streamlit : Information et déploiement de modèles
  - Déploiement immédiat: Bouton "Forcer le déploiement"
  - Déploiement par batch airflow: Bouton "Préparer pour le déploiement"

5. Prédiction 
-- Streamlit: Connexion utilisateur

### Annexe 1 - Github - Gestion des secrets<br>

| Variable secret   | Utilité  | 
|-------------|-------------|
| DOCKER_TOKEN | Dockerhub | 
| DOCKER_USERNAME |  Dockerhub | 
| JWT_SECRET_KEY | Les APIs | 
| KAGGLE_JSON | Admin API | 
| MONITORING_API_KEY | Monitoring API | 
| USERS_JSON | Les APIs | 
| USER_API_KEY | User API | 

### Annexe 2 - Format streamlit_users.json / Variable secret USERS_JSON<br>
{
    "cle_user1": {"username": "identifiant1", "password": "mot_de_passe1", "user_name":"nom_prenom","role": "user"},
    "cle_user2": {"username": "identifiant2", "password": "mot_de_passe2", "user_name":"nom_prenom","role": "admin"},
    ...
}

### Annexe 3 - Github Actions - Workflows<br>
1 workflow par image docker **Docker Build <nom_image>**
1 workflow pour la construction et le push de toutes les images: **Docker Build all containers**

[Retour au menu principal](#acc%C3%A8s-rapide)
[Retour aux instructions](#instructions)

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
