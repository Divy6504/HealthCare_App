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
        .filter(models.User.role == "patient", models.User.is_active.is_(True))
        .order_by(models.User.full_name)
        .all()
    )
