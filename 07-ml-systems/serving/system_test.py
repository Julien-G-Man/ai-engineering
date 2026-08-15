"""
System Tests
 - Focus: isolated system operations
 - Purpose: validate system function
 - Scope: endpoint
 - Environment: Python env with app running
"""
import requests
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

endpoint = "http://localhost:8000"

def test_main():
    response = client.get("/")
    print(requests.get(endpoint).json())
    assert response.status_code == 200
    assert response.json() == {"msg": "API"}


def test_delete_nonexistent_item():
    response = client.delete(
        "/items", json={"id": -999})
    assert response.status_code == 404
    assert response.json() == {"detail": "Item not found."}


print(test_main())