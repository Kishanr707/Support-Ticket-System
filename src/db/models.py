"""
src/db/models.py

One table for now: every prediction made through /predict gets logged
here, with room for a human to later submit the corrected priority via
POST /tickets/{id}/correct.

IMPORTANT for the future retraining pipeline: always train on
confirmed_priority, never on predicted_priority. Training on the
model's own predictions would reinforce whatever it's already getting
wrong instead of correcting it. confirmed_priority being NULL means
"no human has reviewed this ticket yet" — such rows should be excluded
from any retraining dataset.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text

from src.db.database import Base


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    ticket_text = Column(Text, nullable=False)
    predicted_priority = Column(String(20), nullable=False)

    # Filled in later by a human via POST /tickets/{id}/correct.
    # NULL means not yet reviewed.
    confirmed_priority = Column(String(20), nullable=True)
    corrected_by = Column(String(100), nullable=True)
    corrected_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))