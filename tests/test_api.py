import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.db.database import Base, get_db


# Separate in-memory database used only for tests
TEST_DATABASE_URL = "sqlite://"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


@pytest.fixture(scope="module")
def client():
    # Create tables in the test database
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Make the API use the test database
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    # Clean up
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def test_predict_creates_ticket(client):
    response = client.post(
        "/predict",
        json={
            "ticket_text": "My system is not working properly."
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "ticket_id" in data
    assert "priority" in data
    assert "confidence" in data
    assert "needs_human_review" in data


def test_review_queue_returns_review_tickets(client):
    response = client.post(
        "/predict",
        json={
            "ticket_text": "Something is not working properly with the system."
        },
    )

    assert response.status_code == 200

    ticket_id = response.json()["ticket_id"]

    review_response = client.get("/tickets/review")

    assert review_response.status_code == 200

    review_tickets = review_response.json()

    assert any(ticket["id"] == ticket_id for ticket in review_tickets)


def test_correct_ticket(client):
    response = client.post(
        "/predict",
        json={
            "ticket_text": "Something is not working properly with the system."
        },
    )

    assert response.status_code == 200

    ticket_id = response.json()["ticket_id"]

    correction_response = client.post(
        f"/tickets/{ticket_id}/correct",
        json={
            "confirmed_priority": "Medium",
            "corrected_by": "admin",
        },
    )

    assert correction_response.status_code == 200

    data = correction_response.json()

    assert data["id"] == ticket_id
    assert data["confirmed_priority"] == "Medium"
    assert data["corrected_by"] == "admin"
    assert data["corrected_at"] is not None


def test_corrected_ticket_removed_from_review_queue(client):
    response = client.post(
        "/predict",
        json={
            "ticket_text": "Something is not working properly with the system."
        },
    )

    assert response.status_code == 200

    ticket_id = response.json()["ticket_id"]

    correction_response = client.post(
        f"/tickets/{ticket_id}/correct",
        json={
            "confirmed_priority": "Medium",
            "corrected_by": "admin",
        },
    )

    assert correction_response.status_code == 200

    review_response = client.get("/tickets/review")

    assert review_response.status_code == 200

    review_tickets = review_response.json()

    assert not any(ticket["id"] == ticket_id for ticket in review_tickets)


def test_invalid_confirmed_priority_rejected(client):
    response = client.post(
        "/predict",
        json={
            "ticket_text": "Something is not working properly with the system."
        },
    )

    assert response.status_code == 200

    ticket_id = response.json()["ticket_id"]

    correction_response = client.post(
        f"/tickets/{ticket_id}/correct",
        json={
            "confirmed_priority": "Urgent",
            "corrected_by": "admin",
        },
    )

    assert correction_response.status_code == 422