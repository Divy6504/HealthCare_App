import google.generativeai as genai
from .config import settings

_configured = False

# Raw column names -> human-readable labels, used for both the doctor note context
# and (critically) the patient report, which must never show a raw feature name.
FEATURE_LABELS = {
    "race": "race/ethnicity on file",
    "gender": "gender",
    "age": "age group",
    "time_in_hospital": "length of hospital stay",
    "medical_specialty": "treating specialty",
    "num_lab_procedures": "number of lab tests performed",
    "num_procedures": "number of procedures performed",
    "num_medications": "number of medications prescribed",
    "number_outpatient": "outpatient visits in the past year",
    "number_emergency": "emergency room visits in the past year",
    "number_inpatient": "hospital admissions in the past year",
    "number_diagnoses": "number of diagnoses recorded",
    "diag_1": "primary diagnosis category",
    "diag_2": "secondary diagnosis category",
    "diag_3": "additional diagnosis category",
    "max_glu_serum": "glucose serum test result",
    "A1Cresult": "A1C (blood sugar) test result",
    "change": "recent change in diabetes medication",
    "diabetesMed": "whether currently on diabetes medication",
    "admission_type_id": "type of hospital admission",
    "discharge_disposition_id": "discharge plan/destination",
    "admission_source_id": "how the patient was admitted",
    "insulin": "insulin dosage adjustment",
    "metformin": "metformin dosage adjustment",
    "glyburide": "glyburide dosage adjustment",
    "glipizide": "glipizide dosage adjustment",
    "pioglitazone": "pioglitazone dosage adjustment",
    "rosiglitazone": "rosiglitazone dosage adjustment",
}


def _humanize(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def _ensure_configured():
    global _configured
    if not _configured and settings.gemini_api_key:
        genai.configure(api_key=settings.gemini_api_key)
        _configured = True


def _label_text(final_label: str) -> str:
    return {
        "NO": "not likely to be readmitted",
        ">30": "likely to be readmitted, though probably not within the first 30 days",
        "<30": "likely to be readmitted within 30 days",
    }.get(final_label, final_label)


def generate_doctor_note(final_label: str, stage1_probability: float, top_shap_features: list[dict]) -> str:
    """Technical note for the clinician: references SHAP-driven factors directly."""
    if not settings.gemini_api_key:
        return _fallback_doctor_note(final_label, top_shap_features)

    _ensure_configured()
    feature_lines = "\n".join(
        f"- {f['feature']} = {f['value']} (impact: {'+' if f['shap_value'] >= 0 else ''}{f['shap_value']:.3f})"
        for f in top_shap_features
    ) or "- No feature breakdown available"

    prompt = f"""You are a clinical decision-support assistant writing a short note for a hospital
clinician about a diabetes readmission risk assessment. Write it the way a real clinical note
sounds — do not refer to "the model," "the algorithm," or "the prediction" as the subject of
sentences. State findings directly, the way a clinician would phrase a risk stratification note.

Risk assessment: this patient's readmission risk is assessed as {_label_text(final_label)}
(estimated risk score: {stage1_probability:.2f}).

Top contributing factors (SHAP values, sign shows push toward higher risk (+) or lower risk (-)):
{feature_lines}

Write a concise (3-5 sentence) clinical note explaining the prediction in plain language,
referencing the top 2-3 factors naturally, and suggesting one practical next step for the
care team. Do not invent facts not present above. No headers, no bullet points, plain prose."""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[llm_service] Gemini call failed for doctor note: {type(e).__name__}: {e}")
        return _fallback_doctor_note(final_label, top_shap_features)


def generate_patient_report(final_label: str, stage1_probability: float, top_shap_features: list[dict], patient_name: str = None) -> str:
    """Plain-language report for the patient: no SHAP values, no raw column names, no numbers-as-jargon."""
    if not settings.gemini_api_key:
        return _fallback_patient_report(final_label, top_shap_features, patient_name)
    _ensure_configured()
    detail_lines = "\n".join(
        f"- {_humanize(f['feature'])}: {f['value']}" for f in top_shap_features[:6]
    ) or "- General health history"

    name_instruction = (
        f'Address the patient by name: "{patient_name}". Use their name naturally once or twice, '
        f'not in every sentence.' if patient_name else
        'Do not use a placeholder name or bracket like "[Patient Name]" — write it without any '
        'name-based greeting at all, addressing them simply as "you" throughout.'
    )
    prompt = f"""You are writing a detailed, personal readmission-risk report directly for a
patient with no medical background. Do not mention SHAP, probabilities, percentages, statistics,
model names, or any raw data-field names. Do not give a diagnosis. {name_instruction}

Outcome: the care team's assessment suggests the patient is {_label_text(final_label)}.

Specific details that most influenced this assessment (factor: actual value from their record):
{detail_lines}

Write a warm, thorough report for the patient, 3-4 short paragraphs, covering:
1. What the result means in plain terms, stated clearly up front.
2. A detailed, specific explanation of *why* — walk through each of the details above one by
   one, in plain language, explaining how each one relates to their care (e.g. if insulin was
   reduced, explain what that generally means for their treatment; if they had emergency visits
   recently, note that as a relevant pattern). Be concrete and reference the actual values given,
   not vague generalities.
3. What this means for their day-to-day going forward.
4. Clear, encouraging next steps (follow-up appointments, medication adherence, symptoms to
   watch for, when to contact their care team).

Avoid alarming or clinical-sounding language. Write as if speaking directly and personally to
the patient. No headers, no bullet points, plain prose paragraphs."""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[llm_service] Gemini call failed for patient report: {type(e).__name__}: {e}")
        return _fallback_patient_report(final_label, top_shap_features, patient_name)


def _fallback_doctor_note(final_label: str, top_shap_features: list[dict]) -> str:
    top = ", ".join(f["feature"] for f in top_shap_features[:3]) or "the available clinical features"
    return (
        f"Based on the model, this patient is {_label_text(final_label)}. "
        f"The prediction was driven mainly by {top}. "
        f"Recommend clinical review of these factors before discharge planning."
    )


def _fallback_patient_report(final_label: str, top_shap_features: list[dict], patient_name: str = None) -> str:
    detail_bits = [f"{_humanize(f['feature'])} ({f['value']})" for f in top_shap_features[:6]]
    top = "; ".join(detail_bits) or "your recent health history"
    outcome = {
        "NO": "your care team's assessment suggests you are unlikely to need to return to the hospital soon",
        ">30": "your care team's assessment suggests some chance of returning to the hospital, though not right away",
        "<30": "your care team's assessment suggests a higher chance of needing to return to the hospital soon",
    }.get(final_label, "your care team has reviewed your case")
    greeting = f"Dear {patient_name}, " if patient_name else ""
    return (
        f"{greeting}Based on your recent visit, {outcome}. "
        f"This assessment was informed by several details from your record: {top}. "
        f"Each of these plays a role in how your care team thinks about your recovery and risk "
        f"of needing further hospital care. "
        f"Going forward, it's a good idea to attend any follow-up appointments, take your "
        f"medications exactly as prescribed, and keep an eye out for any new or worsening "
        f"symptoms. If anything feels different or concerning, don't hesitate to contact your "
        f"care team — catching small issues early is one of the best ways to avoid a return trip "
        f"to the hospital."
    )
