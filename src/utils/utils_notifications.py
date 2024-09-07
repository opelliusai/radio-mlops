import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from src.config.log_config import setup_logging

logger = setup_logging("UTILS_NOTIFICATIONS")

# Informations de l'expéditeur et du destinataire
sender_email = "mymail@gmail.com"
receiver_email = "mymail@gmail.com"
subject = "Résultats du modèle de machine learning"
body = "Voici les résultats de votre modèle :..."

# Informations du serveur SMTP de Gmail
smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_login = "mymail@gmail.com"
smtp_password = "MDP"

# Création du message
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = subject

# Attachement du corps de l'e-mail
msg.attach(MIMEText(body, 'plain'))

# Connexion au serveur SMTP de Gmail et envoi de l'e-mail
try:
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()  # Démarre la connexion sécurisée
    server.login(smtp_login, smtp_password)
    text = msg.as_string()
    server.sendmail(sender_email, receiver_email, text)
    print("Email envoyé avec succès!")
except Exception as e:
    print(f"Erreur: {e}")
finally:
    server.quit()
