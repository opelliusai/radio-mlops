# test_cron.py

import datetime

# Obtenir l'heure actuelle
now = datetime.datetime.now()

# Écrire dans un fichier
with open("/app/data/cron.log", "a") as file:
    file.write(f"Cron job exécuté à : {now}\n")
