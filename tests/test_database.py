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

# In-memory SQLite, fresh for the whole test module.
#
# StaticPool is required here: without it, SQLAlchemy opens a NEW
# connection (and therefore a NEW, empty in-memory database) each time
# one is checked out from the pool, since ":memory:" databases aren't
# shared between connections by default. That caused "no such table:
# tickets" — the table got created on one connection, then a request
# used a different, empty one. StaticPool forces every checkout to
# reuse the same single connection, so the table created by
# Base.metadata.create_all() is the one everything else actually sees.
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    X = ["vpn is down", "password reset needed", "printer out of paper"]
    y = ["High", "Medium", "Low"]
    pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X, y)
    monkeypatch.setattr(
        "src.model.predict.load_model", lambda model_path=None: pipeline
    )


client = TestClient(app)


def test_predict_logs_a_ticket():
    response = client.post("/predict", json={"ticket_text": "the vpn is down"})
    assert response.status_code == 200
    data = response.json()
    assert "priority" in data
    assert "ticket_id" in data


def test_predicted_ticket_appears_in_list():
    client.post("/predict", json={"ticket_text": "printer out of paper"})
    response = client.get("/tickets")
    assert response.status_code == 200
    tickets = response.json()
    assert len(tickets) >= 1
    assert tickets[0]["confirmed_priority"] is None  # not corrected yet


def test_correct_ticket_updates_confirmed_priority():
    predict_response = client.post("/predict", json={"ticket_text": "password reset needed"})
    ticket_id = predict_response.json()["ticket_id"]

    correct_response = client.post(
        f"/tickets/{ticket_id}/correct",
        json={"confirmed_priority": "Highest", "corrected_by": "test_admin"},
    )
    assert correct_response.status_code == 200
    data = correct_response.json()
    assert data["confirmed_priority"] == "Highest"
    assert data["corrected_by"] == "test_admin"
    assert data["corrected_at"] is not None


def test_correct_nonexistent_ticket_returns_404():
    response = client.post(
        "/tickets/999999/correct",
        json={"confirmed_priority": "Low"},
    )
    assert response.status_code == 404