'''
Créé le 07/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Tests - API Admin
'''
import pytest
from httpx import AsyncClient
from httpx import ASGITransport
from src.api.admin_api import app
from src.config.run_config import init_paths, admin_api_info
from fastapi import status

transport = ASGITransport(app=app)


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=transport, base_url=admin_api_info["ADMIN_API_URL"]) as ac:
        response = await ac.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "OK"}


@pytest.mark.asyncio
async def test_metrics():
    async with AsyncClient(transport=transport, base_url=admin_api_info["ADMIN_API_URL"]) as ac:
        response = await ac.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert "MLOps_admin_api_request_processing_seconds" in response.content.decode()


@pytest.mark.asyncio
async def test_download_dataset():
    async with AsyncClient(transport=transport, base_url=admin_api_info["ADMIN_API_URL"]) as ac:
        response = await ac.get("/download_dataset")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "Téléchargement terminé"}


@pytest.mark.asyncio
async def test_update_dataset():
    async with AsyncClient(transport=transport, base_url=admin_api_info["ADMIN_API_URL"]) as ac:
        response = await ac.post("/update_dataset", json={
            "dataset_path": None,
            "source_type": "KAGGLE",
            "base_dataset_id": None
        })
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "Mise à jour terminée"}


@pytest.mark.asyncio
async def test_train_model():
    async with AsyncClient(transport=transport, base_url=admin_api_info["ADMIN_API_URL"]) as ac:
        response = await ac.post("/train_model", params={
            "retrain": False,
            "model_name": None,
            "model_version": None,
            "include_prod_data": False,
            "balance": True,
            "dataset_version": None,
            "max_epochs": 1,
            "num_trials": 1
        })
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "Entrainement terminé"


@pytest.mark.asyncio
async def test_make_model_prod_ready():
    async with AsyncClient(transport=transport, base_url=admin_api_info["ADMIN_API_URL"]) as ac:
        response = await ac.post("/make_model_prod_ready", params={"num_version": 1})
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "Mise à jour terminée"


@pytest.mark.asyncio
async def test_deploy_ready_model():
    async with AsyncClient(transport=transport, base_url=admin_api_info["ADMIN_API_URL"]) as ac:
        response = await ac.post("/deploy_ready_model")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "Déploiement en production terminée"


@pytest.mark.asyncio
async def test_get_models_list():
    async with AsyncClient(transport=transport, base_url=admin_api_info["ADMIN_API_URL"]) as ac:
        response = await ac.get("/get_models_list")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "OK"


@pytest.mark.asyncio
async def test_get_runs_info():
    async with AsyncClient(transport=transport, base_url=admin_api_info["ADMIN_API_URL"]) as ac:
        response = await ac.post("/get_runs_info", json={"run_ids": [1, 2, 3]})
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "OK"


@pytest.mark.asyncio
async def test_get_datasets_list():
    async with AsyncClient(transport=transport, base_url=admin_api_info["ADMIN_API_URL"]) as ac:
        response = await ac.post("/get_datasets_list", json={"type": "REF"})
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "OK"
