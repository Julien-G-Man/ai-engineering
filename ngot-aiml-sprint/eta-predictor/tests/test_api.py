from fastapi.testclient import TestClient
from eta_predictor.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_root():
    r = client.get("/")
    assert r.json().get("message")
