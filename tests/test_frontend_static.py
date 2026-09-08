from fastapi.testclient import TestClient

from backend.main import app


def test_spa_history_route_serves_frontend_index_when_dist_exists():
    response = TestClient(app).get("/agent-workflow")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<div id="app">' in response.text


def test_unknown_api_route_is_not_swallowed_by_spa_fallback():
    response = TestClient(app).get("/api/not-a-real-endpoint")

    assert response.status_code == 404
    assert '<div id="app">' not in response.text
