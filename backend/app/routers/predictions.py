from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .. import models, schemas, security, ml_service, llm_service
from ..database import get_db
from ..limiter import limiter

router = APIRouter(prefix="/predictions", tags=["predictions"])
_llm_pool = ThreadPoolExecutor(max_workers=4)


@router.post("/", response_model=schemas.PredictionOut)
@limiter.limit("30/hour")
def create_prediction(
    request: Request,
    payload: schemas.PatientInput,
    db: Session = Depends(get_db),
    doctor: models.User = Depends(security.require_role("doctor")),
):
    result = ml_service.predict_cascade(payload)

    linked_patient = None
    if payload.patient_email:
        linked_patient = db.query(models.User).filter(
            models.User.email == payload.patient_email, models.User.role == "patient"
        ).first()

    # Run both Gemini calls concurrently instead of back-to-back — halves worst-case latency.
    doctor_future = _llm_pool.submit(
        llm_service.generate_doctor_note,
        result["final_label"], result["stage1_probability"], result["top_shap_features"]
    )
    patient_name = payload.patient_ref or (linked_patient.full_name if linked_patient else None)
    patient_future = _llm_pool.submit(
        llm_service.generate_patient_report,
        result["final_label"], result["stage1_probability"], result["top_shap_features"], patient_name
    )
    doctor_note = doctor_future.result()
    patient_report = patient_future.result()

    record = models.Prediction(
        doctor_id=doctor.id,
        patient_id=linked_patient.id if linked_patient else None,
        patient_ref=payload.patient_ref,
        input_payload=payload.model_dump(by_alias=True),
        stage1_probability=result["stage1_probability"],
        stage1_label=result["stage1_label"],
        stage2_probability=result["stage2_probability"],
        final_label=result["final_label"],
        top_shap_features=result["top_shap_features"],
        doctor_note=doctor_note,
        patient_report=patient_report,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return _to_doctor_view(record)


@router.get("/", response_model=list[schemas.PredictionHistoryItem])
def list_predictions(
    db: Session = Depends(get_db),
    user: models.User = Depends(security.get_current_user),
    limit: int = 50,
):
    q = db.query(models.Prediction)
    if user.role == "doctor":
        q = q.filter(models.Prediction.doctor_id == user.id)
    elif user.role == "patient":
        q = q.filter(models.Prediction.patient_id == user.id)
    else:
        raise HTTPException(status_code=403, detail="Unsupported role")
    return q.order_by(models.Prediction.created_at.desc()).limit(limit).all()


@router.get("/{prediction_id}", response_model=schemas.PredictionOut)
def get_prediction(
    prediction_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(security.get_current_user),
):
    record = db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if user.role == "doctor" and record.doctor_id == user.id:
        return _to_doctor_view(record)
    if user.role == "patient" and record.patient_id == user.id:
        return _to_patient_view(record)
    raise HTTPException(status_code=403, detail="Not authorized to view this record")


def _to_doctor_view(record: models.Prediction) -> schemas.PredictionOut:
    return schemas.PredictionOut(
        id=record.id,
        stage1_probability=record.stage1_probability,
        stage1_label=record.stage1_label,
        stage2_probability=record.stage2_probability,
        final_label=record.final_label,
        top_shap_features=record.top_shap_features,
        doctor_note=record.doctor_note,
        patient_report=None,
        created_at=record.created_at,
    )


def _to_patient_view(record: models.Prediction) -> schemas.PredictionOut:
    return schemas.PredictionOut(
        id=record.id,
        stage1_probability=record.stage1_probability,
        stage1_label=record.stage1_label,
        stage2_probability=record.stage2_probability,
        final_label=record.final_label,
        top_shap_features=None,
        doctor_note=None,
        patient_report=record.patient_report,
        created_at=record.created_at,
    )
