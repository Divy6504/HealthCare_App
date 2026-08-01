from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Any
import datetime as dt


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: Optional[str] = None
    role: str = Field(default="doctor", pattern="^(doctor|patient)$")


class UserLogin(BaseModel):
    email: EmailStr
    password: str
    role: str = Field(pattern="^(doctor|patient)$")


class UserOut(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str]
    role: str

    class Config:
        from_attributes = True


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


# Raw patient/encounter fields expected from the frontend form.
# These map 1:1 to the pre-encoding columns used in the notebook.
class PatientInput(BaseModel):
    race: str
    gender: str                 # "Female" / "Male"
    age: str                    # "[40-50)" style bucket, matches training format
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    medical_specialty: str = "Unknown"
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: str
    diag_2: str
    diag_3: str
    number_diagnoses: int
    max_glu_serum: str = "None"
    A1Cresult: str = "None"
    change: str                 # "No" / "Ch"
    diabetesMed: str            # "No" / "Yes"

    # medication columns, values one of: No / Down / Steady / Up
    metformin: str = "No"
    repaglinide: str = "No"
    nateglinide: str = "No"
    chlorpropamide: str = "No"
    glimepiride: str = "No"
    glipizide: str = "No"
    glyburide: str = "No"
    pioglitazone: str = "No"
    rosiglitazone: str = "No"
    acarbose: str = "No"
    miglitol: str = "No"
    tolazamide: str = "No"
    insulin: str = "No"
    glyburide_metformin: str = Field(default="No", alias="glyburide-metformin")

    patient_ref: Optional[str] = None
    patient_email: Optional[EmailStr] = None  # link to an existing patient account, if known

    class Config:
        populate_by_name = True


class ShapFeature(BaseModel):
    feature: str
    value: Any
    shap_value: float


class PredictionOut(BaseModel):
    id: str
    stage1_probability: float
    stage1_label: str
    stage2_probability: Optional[float]
    final_label: str
    top_shap_features: Optional[list[ShapFeature]] = None  # only populated for doctor role
    doctor_note: Optional[str] = None                       # only populated for doctor role
    patient_report: Optional[str] = None                    # only populated for patient role
    created_at: dt.datetime

    class Config:
        from_attributes = True


class PredictionHistoryItem(BaseModel):
    id: str
    patient_ref: Optional[str]
    final_label: str
    stage1_probability: float
    created_at: dt.datetime

    class Config:
        from_attributes = True
