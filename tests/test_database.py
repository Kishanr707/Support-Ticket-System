"""tests/test_database.py — unit tests for the DB-backed API endpoints

Uses an in-memory SQLite database, overriding the get_db dependency, so
these tests never touch the real tickets.db file.
"""

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.db.database import Base, get_db


# In-memory SQLite, shared across the whole test module.
#
# StaticPool is required because SQLite's ":memory:" database is normally
# created separately for each connection. StaticPool forces all sessions
# to reuse the same connection.
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def fresh_db():
    """Create a fresh database for every test."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    """Use a small fake ML model so tests do not depend on the real model."""
    X = [
        "vpn is down",
        "password reset needed",
        "printer out of paper",
    ]

    y = [
        "High",
        "Medium",
        "Low",
    ]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X, y)

    monkeypatch.setattr(
        "src.model.predict.load_model",
        lambda model_path=None: pipeline,
    )


client = TestClient(app)


def test_predict_logs_a_ticket():
    """Prediction should return priority, confidence, review status and ticket ID."""
    response = client.post(
        "/predict",
        json={"ticket_text": "the vpn is down"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] in {
        "Blocker",
        "Highest",
        "High",
        "Medium",
        "Low",
    }

    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["needs_human_review"], bool)
    assert isinstance(data["ticket_id"], int)

    assert "priority" in data
    assert "ticket_id" in data


def test_predicted_ticket_appears_in_list():
    """A predicted ticket should be stored and returned by GET /tickets."""
    response = client.post(
        "/predict",
        json={"ticket_text": "printer out of paper"},
    )

    assert response.status_code == 200

    response = client.get("/tickets")

    assert response.status_code == 200

    tickets = response.json()

    assert len(tickets) >= 1
    assert tickets[0]["confirmed_priority"] is None


def test_correct_ticket_updates_confirmed_priority():
    """An admin correction should update the confirmed priority."""
    predict_response = client.post(
        "/predict",
        json={"ticket_text": "password reset needed"},
    )

    assert predict_response.status_code == 200

    ticket_id = predict_response.json()["ticket_id"]

    correct_response = client.post(
        f"/tickets/{ticket_id}/correct",
        json={
            "confirmed_priority": "Highest",
            "corrected_by": "test_admin",
        },
    )

    assert correct_response.status_code == 200

    data = correct_response.json()

    assert data["confirmed_priority"] == "Highest"
    assert data["corrected_by"] == "test_admin"
    assert data["corrected_at"] is not None


def test_correct_nonexistent_ticket_returns_404():
    """Correcting a ticket that does not exist should return 404."""
    response = client.post(
        "/tickets/999999/correct",
        json={"confirmed_priority": "Low"},
    )

    assert response.status_code == 404

def test_prediction_metadata_is_stored():
    """Confidence and human-review status should be persisted."""
    response = client.post(
        "/predict",
        json={"ticket_text": "the vpn is down"},
    )

    assert response.status_code == 200

    prediction = response.json()

    response = client.get("/tickets")

    assert response.status_code == 200

    tickets = response.json()

    assert len(tickets) == 1

    ticket = tickets[0]

    assert ticket["id"] == prediction["ticket_id"]
    assert ticket["confidence"] == prediction["confidence"]
    assert ticket["needs_human_review"] == prediction["needs_human_review"]