#!/bin/bash

# User API
echo "Démarrage de User API"
docker-compose -f docker-compose_user_api.yml up -d

# Admin API
echo "Démarrage d'Admin API"
docker-compose -f docker-compose_admin_api.yml up -d

# Monitoring API
echo "Démarrage de Monitoring API"
docker-compose -f docker-compose_monitoring_api.yml up -d

# Prometheus
echo "Démarrage de Prometheus"
docker-compose -f docker-compose_prometheus.yml up -d

# Grafana
echo "Démarrage de Grafana"
docker-compose -f docker-compose_grafana.yml up -d

# MLFlow
echo "Démarrage de MLFlow"
docker-compose -f docker-compose_mlflow.yml up -d

# Streamlit
echo "Démarrage de Streamlit"
docker-compose -f docker-compose_streamlit.yml up -d

# Streamlit Pres
echo "Démarrage de Streamlit Pres"
docker-compose -f docker-compose_streamlit_pres.yml up -d

# Airflow INIT
echo "Démarrage de Airflow INIT"
docker-compose -f docker-compose_airflow.yml up airflow-init -d

# Airflow
echo "Démarrage de Airflow"
docker-compose -f docker-compose_airflow.yml up -d
