"""
src/db/models.py

Database model for support tickets.

Every prediction made through /predict is logged here.
Human-confirmed corrections are stored separately and should be used
for future retraining.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from src.db.database import Base


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    ticket_text = Column(Text, nullable=False)

    # ML prediction
    predicted_priority = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    needs_human_review = Column(Integer, nullable=False, default=0)

    # Human feedback
    # NULL means the ticket has not been reviewed yet.
    confirmed_priority = Column(String(20), nullable=True)
    corrected_by = Column(String(100), nullable=True)
    corrected_at = Column(DateTime, nullable=True)

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )