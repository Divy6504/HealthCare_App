"""
Loads the artifacts produced by the notebook's save cell and reproduces the
exact preprocessing pipeline used at training time, so raw form input from
the frontend can be turned into the three encodings (native / sklearn / logreg)
the cascade models expect.
"""
import os
import joblib
import numpy as np
import pandas as pd

from .config import settings
from .schemas import PatientInput

_ART = settings.artifacts_dir

_model_s1 = None
_model_s2 = None
_bundle = None
_shap_s1 = None
_shap_s2 = None


def load_artifacts():
    global _model_s1, _model_s2, _bundle, _shap_s1, _shap_s2
    _model_s1 = joblib.load(os.path.join(_ART, "model_stage1.joblib"))
    _model_s2 = joblib.load(os.path.join(_ART, "model_stage2.joblib"))
    _bundle = joblib.load(os.path.join(_ART, "preprocessing_bundle.joblib"))

    s1_path = os.path.join(_ART, "shap_explainer_s1.joblib")
    s2_path = os.path.join(_ART, "shap_explainer_s2.joblib")
    _shap_s1 = joblib.load(s1_path) if os.path.exists(s1_path) else None
    _shap_s2 = joblib.load(s2_path) if os.path.exists(s2_path) else None


def _map_icd9(code: str) -> str:
    if code is None or code == "?" or code == "":
        return "Missing"
    code = str(code)
    if code.startswith("250"):
        return "Diabetes"
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        n = float(code)
    except ValueError:
        return "Other"
    if 390 <= n <= 459 or n == 785:
        return "Circulatory"
    if 460 <= n <= 519 or n == 786:
        return "Respiratory"
    if 520 <= n <= 579 or n == 787:
        return "Digestive"
    if 800 <= n <= 999:
        return "Injury"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 580 <= n <= 629 or n == 788:
        return "Genitourinary"
    if 140 <= n <= 239:
        return "Neoplasms"
    return "Other"

_AGE_BUCKET_TO_DECADE = {f"[{i}-{i+10})": i // 10 for i in range(0, 100, 10)}

# Standard UCI Diabetes 130-US-hospitals ID mappings, so raw numeric IDs never
# leak into doctor notes or patient reports as meaningless numbers.
_ADMISSION_TYPE_LABELS = {
    "1": "Emergency", "2": "Urgent", "3": "Elective", "4": "Newborn",
    "5": "Not available", "6": "Not available", "7": "Trauma center", "8": "Not available",
}
_DISCHARGE_DISPOSITION_LABELS = {
    "1": "Discharged to home", "2": "Transferred to another short-term hospital",
    "3": "Transferred to a skilled nursing facility", "4": "Transferred to an intermediate care facility",
    "6": "Discharged to home with home health service", "7": "Left against medical advice",
    "8": "Discharged to home under home IV care", "9": "Admitted as an inpatient",
    "11": "Deceased", "13": "Hospice care at home", "14": "Hospice care at a facility",
    "18": "Not available", "22": "Transferred to a rehabilitation facility",
    "23": "Transferred to a long-term care hospital", "25": "Not available",
}
_ADMISSION_SOURCE_LABELS = {
    "1": "Physician referral", "2": "Clinic referral", "3": "HMO referral",
    "4": "Transferred from another hospital", "5": "Transferred from a skilled nursing facility",
    "6": "Transferred from another healthcare facility", "7": "Emergency room",
    "8": "Court/law enforcement referral", "9": "Not available",
    "17": "Not available", "20": "Not available",
}
_MED_INT_LABELS = {0: "not prescribed", 1: "dosage decreased", 2: "dosage unchanged", 3: "dosage increased"}


def _humanize_value(feature: str, raw_value):
    """Translate a raw stored value into something a human can actually read."""
    raw_str = str(raw_value)
    if feature == "age":
        try:
            decade = int(raw_value)
            return f"{decade*10}-{decade*10+10} years old"
        except (ValueError, TypeError):
            return raw_value
    if feature == "admission_type_id":
        return _ADMISSION_TYPE_LABELS.get(raw_str, raw_str)
    if feature == "discharge_disposition_id":
        return _DISCHARGE_DISPOSITION_LABELS.get(raw_str, raw_str)
    if feature == "admission_source_id":
        return _ADMISSION_SOURCE_LABELS.get(raw_str, raw_str)
    if isinstance(raw_value, (int, float)) and int(raw_value) in _MED_INT_LABELS and feature not in (
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"
    ):
        return _MED_INT_LABELS[int(raw_value)]
    return raw_value

def _build_raw_row(p: PatientInput) -> dict:
    b = _bundle
    med_mapping = b["med_mapping"]
    row = {
        "race": p.race if p.race else "Unknown",
        "gender": 0 if p.gender.lower().startswith("f") else 1,
        "age": _AGE_BUCKET_TO_DECADE.get(p.age, 4),
        "admission_type_id": str(p.admission_type_id),
        "discharge_disposition_id": str(p.discharge_disposition_id),
        "admission_source_id": str(p.admission_source_id),
        "time_in_hospital": p.time_in_hospital,
        "medical_specialty": p.medical_specialty or "Unknown",
        "num_lab_procedures": p.num_lab_procedures,
        "num_procedures": p.num_procedures,
        "num_medications": p.num_medications,
        "number_outpatient": p.number_outpatient,
        "number_emergency": p.number_emergency,
        "number_inpatient": p.number_inpatient,
        "diag_1": _map_icd9(p.diag_1),
        "diag_2": _map_icd9(p.diag_2),
        "diag_3": _map_icd9(p.diag_3),
        "number_diagnoses": p.number_diagnoses,
        "max_glu_serum": p.max_glu_serum or "None",
        "A1Cresult": p.A1Cresult or "None",
        "change": 1 if p.change == "Ch" else 0,
        "diabetesMed": 1 if p.diabetesMed == "Yes" else 0,
    }
    for med in b["remaining_meds"]:
        key = med.replace("-", "_")
        val = getattr(p, key, "No")
        row[med] = med_mapping.get(val, 0)
    return row


def _to_frame(p: PatientInput) -> pd.DataFrame:
    b = _bundle
    row = _build_raw_row(p)
    df = pd.DataFrame([row])
    # ensure every expected feature column exists (fill missing with 0 as a safe default)
    for col in b["feature_columns"]:
        if col not in df.columns:
            df[col] = 0
    df = df[b["feature_columns"]]

    for col in b["cat_cols"]:
        categories = b["category_maps"][col]
        val = df.at[0, col]
        if val not in categories:
            val = categories[0]
            df.at[0, col] = val
        df[col] = pd.Categorical([val], categories=categories)
    return df


def _encode_all(df_native: pd.DataFrame):
    b = _bundle
    df_sklearn = df_native.copy()
    for col in b["cat_cols"]:
        df_sklearn[col] = df_sklearn[col].cat.codes

    df_logreg_raw = df_native.copy()
    for col in b["cat_cols"]:
        df_logreg_raw[col] = df_logreg_raw[col].astype(str)
    df_logreg = pd.get_dummies(df_logreg_raw, columns=b["cat_cols"])
    df_logreg = df_logreg.reindex(columns=b["logreg_columns"], fill_value=0)
    numeric_cols = b["numeric_cols"]
    df_logreg[numeric_cols] = b["scaler"].transform(df_logreg[numeric_cols])

    return {"native": df_native, "sklearn": df_sklearn, "logreg": df_logreg}


def _pick(encodings: dict, enc: str) -> pd.DataFrame:
    return encodings[enc]


def predict_cascade(p: PatientInput):
    if _bundle is None:
        load_artifacts()
    b = _bundle
    df_native = _to_frame(p)
    encodings = _encode_all(df_native)

    X1 = _pick(encodings, b["enc_s1"])
    proba1 = float(_model_s1.predict_proba(X1)[:, 1][0])
    label1 = "Readmitted" if proba1 >= b["best_threshold_s1"] else "Not Readmitted"

    result = {
        "stage1_probability": proba1,
        "stage1_label": label1,
        "stage2_probability": None,
        "final_label": "NO",
        "shap_source": "stage1",
        "shap_X": X1,
        "shap_model_enc": b["enc_s1"],
        "df_native": df_native,
    }

    if label1 == "Readmitted":
        X2 = _pick(encodings, b["enc_s2"])
        proba2 = float(_model_s2.predict_proba(X2)[:, 1][0])
        label2 = "<30" if proba2 >= b["best_threshold_s2"] else ">30"
        result["stage2_probability"] = proba2
        result["final_label"] = label2
        result["shap_source"] = "stage2"
        result["shap_X"] = X2
        result["shap_model_enc"] = b["enc_s2"]

    result["top_shap_features"] = _top_shap_features(result)
    return result


def _top_shap_features(result: dict, top_n: int = 6):
    explainer = _shap_s1 if result["shap_source"] == "stage1" else _shap_s2
    if explainer is None:
        return []  # winning model was logistic regression; SHAP TreeExplainer not applicable
    X = result["shap_X"]
    try:
        shap_values = explainer.shap_values(X)
        sv = np.array(shap_values)
        # Normalize to a flat 1D array of per-feature SHAP values for the single row, positive class.
        if sv.ndim == 3:
            if sv.shape[0] == 1:               # (1, n_features, n_classes)
                sv = sv[0]
                sv = sv[:, 1] if sv.shape[-1] == 2 else sv[:, 0]
            elif sv.shape[1] == 1:             # (n_classes, 1, n_features)
                idx = 1 if sv.shape[0] == 2 else 0
                sv = sv[idx, 0]
            else:
                sv = sv.reshape(-1)
        elif sv.ndim == 2:
            if sv.shape[0] == 1:               # (1, n_features)
                sv = sv[0]
            elif sv.shape[0] == 2:             # (n_classes, n_features), binary
                sv = sv[1]
            else:
                sv = sv.reshape(-1)
        sv = np.asarray(sv).reshape(-1)
    except Exception as e:
        print(f"[ml_service] SHAP computation failed: {type(e).__name__}: {e}")
        return []

    feature_names = X.columns.tolist()
    df_native = result.get("df_native")
    order = np.argsort(-np.abs(sv))[:top_n]
    out = []
    for i in order:
        fname = feature_names[i]
        # Always display the original human-readable value (category string, not
        # an encoded integer code), regardless of which encoding the model used.
        if df_native is not None and fname in df_native.columns:
            val = df_native.iloc[0][fname]
        else:
            val = X.iloc[0, i]
        if isinstance(val, (np.integer, np.floating, np.bool_)):
            val = val.item()
        elif hasattr(val, "item") and getattr(val, "size", 1) == 1:
            try:
                val = val.item()
            except Exception:
                val = str(val)
        else:
            val = str(val)
        val = _humanize_value(fname, val)
        out.append({
            "feature": feature_names[i],
            "value": str(val),
            "shap_value": float(sv[i]),
        })
    return out
