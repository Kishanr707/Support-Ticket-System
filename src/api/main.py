"""
src/api/main.py

Stage 6 (~30-35% milestone): FastAPI wrapper around predict().

This is the HTTP contract a frontend talks to. It does not contain any
ML logic itself — it just validates a request, calls predict() from
src/model/predict.py, and returns the result as JSON.

Run locally with:
    uvicorn src.api.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs
(FastAPI generates this automatically).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model.predict import predict

app = FastAPI(
    title="Support Ticket Priority API",
    description="Predicts ticket priority (Blocker/Highest/High/Medium/Low) from ticket text.",
    version="0.1.0",
)

# CORS: without this, a browser blocks requests from a frontend running
# on a different origin (e.g. localhost:3000) even if this server is up.
# allow_origins=["*"] is fine for local development; tighten this to
# your actual frontend's URL before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketRequest(BaseModel):
    ticket_text: str = Field(..., min_length=1, description="Raw ticket text")


class PredictionResponse(BaseModel):
    priority: str


@app.get("/health")
def health():
    """Basic liveness check — useful for confirming the server is up
    before debugging anything else."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_priority(request: TicketRequest):
    """Predict the priority of a single ticket from its text.

    Request body:  {"ticket_text": "the vpn is down"}
    Response body: {"priority": "High"}
    """
    try:
        priority = predict(request.ticket_text)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model file not found. Run `python -m src.model.train` first.",
        )
    return PredictionResponse(priority=priority)