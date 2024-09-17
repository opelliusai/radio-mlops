from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

# Importer votre module personnalisé
from src.scripts import monitoring_drift

# Définit les arguments par défaut pour le DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,  # Envoyer un email en cas d'échec
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': ['jelgasmi@icloud.com'],
}


def data_drift_task(**context):
    status, message_objet, message_corps = monitoring_drift.data_drift()
    context['task_instance'].xcom_push(key='status', value=status)
    context['task_instance'].xcom_push(
        key='message_objet', value=message_objet)
    context['task_instance'].xcom_push(
        key='message_corps', value=message_corps)


def check_status_data(**context):
    task_instance = context['task_instance']
    status = task_instance.xcom_pull(task_ids='task_data_drift', key='status')
    print("--check_status : {status}")
    if status == 'KO':
        return 'send_email'
    else:
        return 'skip_email'


def check_status(**context):
    task_instance = context['task_instance']
    status = task_instance.xcom_pull(task_ids='task_data_drift', key='status')
    print("--check_status : {status}")
    if status == 'KO':
        return 'send_email'
    else:
        return 'skip_email'


with DAG(
    'dag_data_drift',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    task_data_drift = PythonOperator(
        task_id='task_data_drift',
        python_callable=data_drift_task,
        provide_context=True
    )

    # Tâche de décision : envoyer ou non un email
    check_status_task = BranchPythonOperator(
        task_id='check_status',
        python_callable=check_status,
        provide_context=True
    )

    # Tâche d'envoi d'email (si le statut est KO)
    task_email = EmailOperator(
        task_id='send_email',
        to='jelgasmi@gmail.com',  # Adresse email du destinataire
        subject="{{ task_instance.xcom_pull(task_ids='task_healthcheck', key='message_objet') }}",
        html_content="{{ task_instance.xcom_pull(task_ids='task_healthcheck', key='message_corps') }}",
    )

    # Tâche pour ignorer l'envoi d'email (si le statut est OK)
    task_skip_email = DummyOperator(
        task_id='skip_email',
    )

    # Flux d'exécution
    task_data_drift >> check_status_task
    check_status_task >> task_email
    check_status_task >> task_skip_email

    check_status_task.set_upstream(task_data_drift)
    check_status_task.trigger_rule = 'all_done'
