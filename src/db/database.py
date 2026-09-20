"""
src/db/database.py

Database engine and session setup using SQLAlchemy.

Defaults to a local SQLite file (tickets.db) so no separate database
server needs to be installed or running to develop against this.
Switching to a real database later (e.g. PostgreSQL, per the project's
original architecture plan) is a one-line change: set the DATABASE_URL
environment variable to a PostgreSQL connection string instead of
relying on the SQLite default. No other code needs to change.

Example for later:
    DATABASE_URL=postgresql://user:password@localhost:5432/tickets
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./tickets.db")

# check_same_thread=False is only needed for SQLite (it's overly strict
# by default about multi-threaded access, which FastAPI uses). This
# argument is ignored/not passed for other databases like PostgreSQL.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a database session, closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't already exist. Called once at API startup."""
    Base.metadata.create_all(bind=engine)