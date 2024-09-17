#!/bin/bash

# Store the start time
start=$(date +%s)

# User API
docker-compose -f docker-compose_user_api.yml build --no-cache

fin=$(date +%s)
# Calculate the difference in seconds
difference=$(( fin - debug ))

# Print the execution time
echo "Exécution du Buil User API en $difference secondes"

# Admin API
docker-compose -f docker-compose_admin_api.yml build --no-cache

fin=$(date +%s)
# Calculate the difference in seconds
difference=$(( fin - debug ))

# Print the execution time
echo "Exécution du Buil Admin API en $difference secondes"

# Monitoring API
docker-compose -f docker-compose_monitoring_api.yml build --no-cache

fin=$(date +%s)
# Calculate the difference in seconds
difference=$(( fin - debug ))

# Print the execution time
echo "Exécution du Buil Monitoring API en $difference secondes"

# Prometheus
docker-compose -f docker-compose_prometheus.yml build --no-cache

fin=$(date +%s)
# Calculate the difference in seconds
difference=$(( fin - debug ))

# Print the execution time
echo "Exécution du Buil Prometheus en $difference secondes"

# Grafana
docker-compose -f docker-compose_grafana.yml build --no-cache

fin=$(date +%s)
# Calculate the difference in seconds
difference=$(( fin - debug ))

# Print the execution time
echo "Exécution du Buil Grafana en $difference secondes"

# MLFlow
docker-compose -f docker-compose_mlflow.yml build --no-cache

fin=$(date +%s)
# Calculate the difference in seconds
difference=$(( fin - debug ))

# Print the execution time
echo "Exécution du Buil MLFlow en $difference secondes"

# Streamlit
docker-compose -f docker-compose_streamlit.yml build --no-cache

fin=$(date +%s)
# Calculate the difference in seconds
difference=$(( fin - debug ))

# Print the execution time
echo "Exécution du Buil Streamlit en $difference secondes"

# Streamlit pres
docker-compose -f docker-compose_streamlit_pres.yml build --no-cache

fin=$(date +%s)
# Calculate the difference in seconds
difference=$(( fin - debug ))

# Print the execution time
echo "Exécution du Buil Streamlit pres en $difference secondes"


fin=$(date +%s)

# Calculate the difference in seconds
difference=$(( fin - debug ))

# Print the execution time
echo "Exécution des build en $difference secondes"