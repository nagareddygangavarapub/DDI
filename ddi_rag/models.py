"""
models.py — SQLAlchemy ORM models for DDI-RAG.

Tables:
    users           — registered accounts
    medications     — per-user medication list (most fields optional)
    query_history   — past prescription queries per user
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, Column, Date, DateTime, ForeignKey, String, Text
)
from sqlalchemy.orm import relationship

from database import Base


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


# ── Users ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id         = Column(String(36), primary_key=True, default=_uuid)
    email      = Column(String(255), unique=True, nullable=False, index=True)
    password   = Column(String(255), nullable=False)           # bcrypt hash
    full_name  = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)

    medications   = relationship("Medication",   back_populates="user",
                                 cascade="all, delete-orphan")
    query_history = relationship("QueryHistory", back_populates="user",
                                 cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id":         self.id,
            "email":      self.email,
            "full_name":  self.full_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ── Medications ───────────────────────────────────────────────────────────────

class Medication(Base):
    __tablename__ = "medications"

    id         = Column(String(36), primary_key=True, default=_uuid)
    user_id    = Column(String(36), ForeignKey("users.id"), nullable=False,
                        index=True)

    # Only drug_name is required; everything else is optional
    drug_name  = Column(String(255), nullable=False, index=True)
    dosage     = Column(String(100),  nullable=True)
    frequency  = Column(String(100),  nullable=True)
    start_date = Column(Date,         nullable=True)
    end_date   = Column(Date,         nullable=True)
    is_active  = Column(Boolean,      nullable=True, default=True)
    notes      = Column(Text,         nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)

    user = relationship("User", back_populates="medications")

    def to_dict(self):
        return {
            "id":         self.id,
            "drug_name":  self.drug_name,
            "dosage":     self.dosage,
            "frequency":  self.frequency,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date":   self.end_date.isoformat()   if self.end_date   else None,
            "is_active":  self.is_active,
            "notes":      self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ── Query History ─────────────────────────────────────────────────────────────

class QueryHistory(Base):
    __tablename__ = "query_history"

    id             = Column(String(36), primary_key=True, default=_uuid)
    user_id        = Column(String(36), ForeignKey("users.id"), nullable=False,
                            index=True)
    prescription   = Column(Text, nullable=False)
    detected_drugs = Column(JSON, nullable=True)   # list of drug names
    warnings       = Column(JSON, nullable=True)   # history_warnings list
    results        = Column(JSON, nullable=True)   # RAG results list
    created_at     = Column(DateTime(timezone=True), default=_now, nullable=False)

    user = relationship("User", back_populates="query_history")

    def to_dict(self):
        return {
            "id":             self.id,
            "prescription":   self.prescription,
            "detected_drugs": self.detected_drugs,
            "warnings":       self.warnings,
            "results":        self.results,
            "created_at":     self.created_at.isoformat() if self.created_at else None,
        }
