import uuid
import datetime as dt
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Float, JSON, Boolean
from sqlalchemy.orm import relationship
from .database import Base


def uuid_str():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=uuid_str)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    role = Column(String, default="doctor")  # doctor | patient | admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    predictions_authored = relationship(
        "Prediction", back_populates="doctor",
        foreign_keys="Prediction.doctor_id", cascade="all, delete-orphan"
    )
    predictions_owned = relationship(
        "Prediction", back_populates="patient",
        foreign_keys="Prediction.patient_id"
    )
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(String, primary_key=True, default=uuid_str)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    user = relationship("User", back_populates="refresh_tokens")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=uuid_str)
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False)
    patient_id = Column(String, ForeignKey("users.id"), nullable=True)  # linked patient account, if any
    patient_ref = Column(String, nullable=True)  # free-text label/name if no linked account

    input_payload = Column(JSON, nullable=False)
    stage1_probability = Column(Float, nullable=False)
    stage1_label = Column(String, nullable=False)   # "Not Readmitted" / "Readmitted"
    stage2_probability = Column(Float, nullable=True)
    final_label = Column(String, nullable=False)    # NO / >30 / <30

    top_shap_features = Column(JSON, nullable=True)  # [{feature, value, shap_value}, ...] — doctor view only
    doctor_note = Column(String, nullable=True)       # clinical, technical
    patient_report = Column(String, nullable=True)    # plain-language, no jargon

    created_at = Column(DateTime, default=dt.datetime.utcnow)

    doctor = relationship("User", back_populates="predictions_authored", foreign_keys=[doctor_id])
    patient = relationship("User", back_populates="predictions_owned", foreign_keys=[patient_id])
