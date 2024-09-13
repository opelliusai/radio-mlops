RADIO-MLOps - Jihane E.
=======================================================================

Ce repository concerne le projet de Radiographie Pulmonaire dans le cadre de la formation MLOps de Datascientest

Structure globale du projet
(en * les répertoires qui ne font pas partie du repository mais qui sont essentiels pour la bonne exécution du projet)
------------

    ├── LICENSE
    ├── README.md               <- README principal pour le projet
    ├── *data                   <- Répertoire des données - local (dans .gitignore)
    │   ├── *archives		<- Archivage 
    │   │
    │   ├── *models			<- Les modèles 	
    │   │
    │   ├── *processed	   	<- Les données après transformation 
    │   │   ├── *dataset_logging <- Fichiers sur l'historique des versions (ref, kaggle, prod)
    │   │   │   ├── kaggle_dataset_logging.csv <- Historique des versions Kaggle
    │   │   │   ├── ref_dataset_logging.csv <- Historique des versions de Référence   
    │   │   │   └── prod_dataset_logging.csv <- Historique des versions de Production       
    │   │   │
    │   │   ├── *keras_tuner 	<- Répertoire dédié aux rapports de Keras Random Search
    │   │   │
    │   │   ├── *mlflow_runs 	<- Répertoire dédié aux rapports générés lors de l'entrainement (history etc.)
    │   │   │
    │   │   ├── *mlruns 		<- Répertoire utilisé au démarrage de MLFlow (Runs et Artifacts)
    │   │   │
    │   │   ├── *model_drift 	<- Répertoire dédié aux model drifts
    │   │   │   ├── model_drift_logging.csv <- Historique de calcul de drift model
    │   │   │   └── <<nom_modele-version>_model_drift_logging.csv <- Historique de calcul de drift après bascule d'une nouvelle version de modèle en production       
    │   │   │
    │   │   ├── *models_data 	<- Métadata des données utilisés par les modèles pour une version donnée de dataset de référence
    │   │   │   └── <<Nom_modele-version>> <- Historique des versions Kaggle
    │   │   │		└──<<Donnees_ref-version>>_<<Nom_modele-version>>_metadata.csv
    │   │   │
    │   │   ├── *predictions 	<- Fichiers de log des prédictions
    │   │   │   └─prediction_logging.csv
    │   │   │
    │   │   ├── *prod_datasets 	<- Datasets de Production avec versions
    │   │   │   
    │   │   ├── *ref_datasets 	<- Datatsets de Référence avec versions
    │   │   │
    │   │   └── *pytest_reports 	<- Rapports d'exécution des tests unitaires
    │   │
    │   ├── *raw                	<- Les données brutes Kaggle 
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
    │   ├── __init__.py         <- Rendre src un module python
    │   │
    │   ├── api                 <- APIs FastAPI
    │   ├── __init__.py         <- Rendre src/api un module python
    │   │   ├── admin_api.py
    │   │   ├── monitoring_api.py
    │   │	└── user_api.py
    │   │
    │   ├── config              <- Fichiers de configuration des paramètres et des logs
    │   │   ├── __init__.py     <- Rendre src/config un module python
    │   │   ├── log_config.py
    │   │	└── run_config.py
    │   │
    │   ├── datasets            <- Fonctions de traitement des bases de données
    │   │   ├── __init__.py     <- Rendre src/datasets un module python
    │   │   ├── clean_dataset.py        <- Préparation/Equilibrage des données pour un modèle
    │   │   ├── download_dataset.py     <- Téléchargement, versioning et génération du metadata.csv 
    │   │   ├── image_preprocessing.py  <- Processing des images pour l'architecture EfficientNet
    │   │	└── update_dataset.py       <- Mise à jour des données de REFERENCE (avec Kaggle ou PROD)
    │   │
    │   ├── mlflow              <- Fonctions de tracking et de serving des modèles sur MLFlow
    │   │   ├── __init__.py     <- Rendre src/mlflow un module python
    │   │   ├── model_serving.py        <- Préparation ou Déploiement d'un modèle sur MLFlow
    │   │	└── model_tracking.py       <- Entrainement / Enregistrement d'un modèle sur MLFlow
    │   │
    │   ├── models              <- Fonction de construction, d'entrainement du modèle et de prédiction
    │   │   ├── __init__.py     <- Rendre src/models un module python
    │   │   ├── build_model.py 		<- Construction de l'architecture EfficientNetB0 et Recherche des meilleurs hyperparamètres
    │   │   ├── model_drift_detection.py 	<- Calcul du drift d'un modèle
    │   │   ├── train_model.py		<- Entrainement et evaluation du modèle (utilisé par srcmlflow/model_tracking.py)
    │   │   └── predict_model.py 	    	<- Exécution de prédictions/Inférence
    │   │
    │   ├── streamlit	    <- Application métier Streamlit
    │   │   ├── __init__.py    
    │   │   ├── Accueil.py
    │   │   └── st_pages        <- Pages de la sidebar streamlit
    │   │       ├── __init__.py    
    │   │       ├── p_datasets.py  		<- Liste des datasets et actions de mise à jour  
    │   │       ├── p_deployment.py		<- Liste des modèles et déploiement
    │   │       ├── p_evaluation.py		<- Historique de calcul de drift et Action d'exécution
    │   │       ├── p_performance.py 	<- Liste des prédictions/temps de prédiction/précision du modèle de production
    │   │       ├── p_predict.py		<- Prédiction, indice de confiance         
    │   │       ├── p_services.py		<- Etat de service des APIs, applications et MLFlow
    │   │       └── p_training.py		<- Entrainement d'un nouveau modèle ou réentrainement d'un modèle existant sur les données de production
    │   │
    │   ├── streamlit_pres     <- Application projet Streamlit (Soutenance)
    │   │   ├── __init__.py    <- Rendre src/streamlit un module python
    │   │   ├── Accueil.py
    │   │   └── pages
    │   │    
    │   ├── tests          	   <- Tests unitaires
    │   │   ├── __init__.py    
    │   │   ├── tests_admin_api.py
    │   │   ├── tests_monitoring_api.py
    │   │   ├── tests_user_api.py
    │   │   └── tests_<<autre>>.py
    │   │
    │   ├── utils                <- Fonctions utiles 
    │   │   ├── __init__.py      <- Rendre src/utils un module python
    │   │   ├── utils_data.py	   <- Fonctions utiles au traitement des données
    │   │   ├── utils_folders_init.py  <- Création des répertoires nécessaires au projet	
    │   │   ├── utils_models.py  	   <- Fonctions utiles à la gestion des modèles	
            └── utils_streamlit.py      <- Fonctions "Interface" entre les APIs et le FrontEnd Streamlit	

Instructions:
1. Créer un environnement virtuel<br>
`python3 -m venv radioEnv`

2. Activer l’environnement<br>
`source radioEnv/bin/activate`

3. Ajouter le projet dans la variable PYTHONPATH<br>

4. Installer les pré-requis<br>
`pip install -r requirements.txt`

5. Créer tous les répertoires nécessaires<br>
`python src/utils/utils_folders_init.py`

6. Exécution locale<br>
>> A VENIR <<

7. Utilisation d'images Docker<br>
Se positionner dans le répertoire docker-compose<br>
`cd docker-compose`

-- Construire toutes les images<br>
`docker-compose build --no-cache`
<br>ou<br>
-- Construction image par image<br>
`docker-compose -f docker-compose_<<nom_image>>.yml build --no-cache`

-- Démarrer les conteneurs<br>
`docker-compose up`
<br>ou<br>
-- Démarrage container par container<br>
`docker-compose -f docker-compose_<<nom_image>>.yml up --no-cache`

Les images disponibles et leurs ports:<br>
- API User: 8081<br>
- API Admin : 8082<br>
- API Monitoring : 8083<br>
- MLflow : 8090<br>
- Prometheus : 8090<br>
- Grafana : 8090<br>
- Airflow : 8090<br>
- Streamlit : 8084<br>
- Streamlit Présentation: 8085<br>

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
