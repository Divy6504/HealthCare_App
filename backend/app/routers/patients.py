from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from .. import models, security
from ..database import get_db

router = APIRouter(prefix="/patients", tags=["patients"])


class PatientListItem(BaseModel):
    id: str
    email: str
    full_name: str | None

    class Config:
        from_attributes = True


@router.get("/", response_model=list[PatientListItem])
def list_patients(
    db: Session = Depends(get_db),
    doctor: models.User = Depends(security.require_role("doctor")),
):
    return (
        db.query(models.User)
        .join(models.Prediction, models.Prediction.patient_id == models.User.id)
        .filter(models.Prediction.doctor_id == doctor.id, models.User.role == "patient")
        .distinct()
        .order_by(models.User.full_name)
        .all()
    )
