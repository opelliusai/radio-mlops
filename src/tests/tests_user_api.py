import pytest
from httpx import AsyncClient
from httpx import ASGITransport
from fastapi import status
from src.api.user_api import app
from datetime import timedelta
import os
from src.config.run_config import init_paths, user_api_info

transport = ASGITransport(app=app)


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=transport, base_url=user_api_info["USER_API_URL"]) as ac:
        response = await ac.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "OK"}


@pytest.mark.asyncio
async def test_metrics():
    async with AsyncClient(transport=transport, base_url=user_api_info["USER_API_URL"]) as ac:
        response = await ac.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert "MLOps_user_api_request_processing_seconds" in response.content.decode()


@pytest.mark.asyncio
async def test_login_success():
    async with AsyncClient(transport=transport, base_url=user_api_info["USER_API_URL"]) as ac:
        response = await ac.post("/login", data={"username": "user1", "password": "user123"})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_failure():
    async with AsyncClient(transport=transport, base_url=user_api_info["USER_API_URL"]) as ac:
        response = await ac.post("/login", data={"username": "user1", "password": "toto"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": "Identifiants incorrects"}


@pytest.mark.asyncio
async def test_predict():
    test_image = os.path.join(init_paths["test_images"], "Covid", '094.png')
    with open(test_image, 'rb') as image:
        files = {"image": image}
    async with AsyncClient(transport=transport, base_url=user_api_info["USER_API_URL"]) as ac:
        response = await ac.post("/predict", files=files)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "OK"
    assert "prediction" in data


@pytest.mark.asyncio
async def test_add_image():
    test_add_image = os.path.join(
        init_paths["test_images"], "Normal", '0101.jpeg')
    async with AsyncClient(transport=transport, base_url=user_api_info["USER_API_URL"]) as ac:
        response = await ac.post("/add_image", json={"image_path": test_add_image, "label": "Normal"})
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "OK"}


@pytest.mark.asyncio
async def test_get_classes():
    async with AsyncClient(transport=transport, base_url=user_api_info["USER_API_URL"]) as ac:
        response = await ac.get("/get_classes")
    assert response.status_code == status.HTTP_200_OK
    assert "classes" in response.json()


@pytest.mark.asyncio
async def test_update_log_prediction():
    test_image = os.path.join(init_paths["test_images"], "Covid", '094.png')
    with open(test_image, 'rb') as image:
        files = {"image": image}
    async with AsyncClient(transport=transport, base_url=user_api_info["USER_API_URL"]) as ac:
        response = await ac.post("/predict", files=files)
    pred_id = response.json()["pred_id"]
    async with AsyncClient(transport=transport, base_url=user_api_info["USER_API_URL"]) as ac:
        response = await ac.post("/update_log_prediction", json={"pred_id": pred_id, "label": "COVID"})
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "OK"}
