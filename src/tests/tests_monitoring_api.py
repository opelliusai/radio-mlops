'''
Créé le 07/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Tests - API Monitoring
'''
import pytest
from httpx import AsyncClient
from httpx import ASGITransport
from fastapi import status
from src.api.monitoring_api import app
import prometheus_client
from src.config.run_config import monitoring_api_info

transport = ASGITransport(app=app)


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=transport, base_url=monitoring_api_info["MONITORING_API_URL"]) as ac:
        response = await ac.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "OK"}


@pytest.mark.asyncio
async def test_metrics():
    async with AsyncClient(transport=transport, base_url=monitoring_api_info["MONITORING_API_URL"]) as ac:
        response = await ac.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert prometheus_client.CONTENT_TYPE_LATEST in response.headers["content-type"]


@pytest.mark.asyncio
async def test_drift_metrics():
    async with AsyncClient(transport=transport, base_url=monitoring_api_info["MONITORING_API_URL"]) as ac:
        response = await ac.get("/drift_metrics")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "Calcul terminé"
    assert "model_name" in data
    assert "new_mean" in data
    assert "original_mean" in data
    assert "mean_diff" in data
    assert "std_diff" in data
    assert "drift" in data
