"""
src/api/main.py

FastAPI wrapper around predict(), now with database logging.

Every /predict call is saved to the tickets table (predicted priority,
not yet confirmed by a human). GET /tickets lists them — this is the
data an admin correction dashboard would display. POST
/tickets/{id}/correct is where a human submits the real priority,
which is what any future retraining pipeline should train on.

Run locally with:
    uvicorn src.api.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.database import get_db, init_db
from src.db.models import Ticket
from src.model.predict import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Support Ticket Priority API",
    description="Predicts ticket priority (Blocker/Highest/High/Medium/Low) from ticket text.",
    version="0.2.0",
    lifespan=lifespan,
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
    ticket_id: int


class TicketOut(BaseModel):
    id: int
    ticket_text: str
    predicted_priority: str
    confirmed_priority: Optional[str] = None
    corrected_by: Optional[str] = None
    created_at: datetime
    corrected_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class CorrectionRequest(BaseModel):
    confirmed_priority: str = Field(..., description="The human-confirmed correct priority")
    corrected_by: str = Field(default="admin", description="Who made the correction")


@app.get("/health")
def health():
    """Basic liveness check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_priority(request: TicketRequest, db: Session = Depends(get_db)):
    """Predict the priority of a single ticket from its text, and log
    the prediction to the database.

    Request body:  {"ticket_text": "the vpn is down"}
    Response body: {"priority": "High", "ticket_id": 42}
    """
    try:
        priority = predict(request.ticket_text)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model file not found. Run `python -m src.model.train` first.",
        )

    ticket = Ticket(ticket_text=request.ticket_text, predicted_priority=priority)
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    return PredictionResponse(priority=priority, ticket_id=ticket.id)


@app.get("/tickets", response_model=List[TicketOut])
def list_tickets(db: Session = Depends(get_db)):
    """List all logged tickets, most recent first. This is the data an
    admin correction dashboard would display."""
    return db.query(Ticket).order_by(Ticket.created_at.desc()).all()


@app.post("/tickets/{ticket_id}/correct", response_model=TicketOut)
def correct_ticket(ticket_id: int, correction: CorrectionRequest, db: Session = Depends(get_db)):
    """Submit the human-confirmed correct priority for a ticket.

    This is the feedback-loop endpoint: any future retraining pipeline
    should train on confirmed_priority, never predicted_priority.
    """
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    ticket.confirmed_priority = correction.confirmed_priority
    ticket.corrected_by = correction.corrected_by
    ticket.corrected_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(ticket)

    return ticket