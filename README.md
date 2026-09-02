ReadmitRisk — Diabetes 30-Day Readmission Risk Platform

A full-stack clinical decision-support tool that predicts diabetes patient readmission risk using a two-stage cascade ML model, explains predictions with SHAP, and generates two distinct AI-written reports from the same prediction — one technical (for doctors), one plain-language (for patients) — via Google Gemini.

Live app: https://healthcareapp-5gdtbvbqiwunwhjuhyz2un.streamlit.app

What this actually does

A two-stage cascade classifier (not a single model) — stage 1 decides readmitted vs. not, and only if stage 1 says "readmitted" does stage 2 run to decide within 30 days vs. after 30 days. Six candidate models (CatBoost, XGBoost, LightGBM, Random Forest, Decision Tree, Logistic Regression) were evaluated per stage; the best performer for each stage was selected and serialized.

SHAP TreeExplainer runs on every live prediction to extract the top contributing features — not a static feature-importance chart, a per-patient explanation computed at inference time. Those SHAP values feed two separate Gemini prompts producing different outputs for the same prediction: a technical clinical note referencing exact feature values and SHAP directionality for the doctor, and a plain-language, jargon-free report for the patient that explicitly never mentions probabilities, SHAP, or raw data-field names — with a human-readable label lookup table (built from the UCI dataset's own ID mapping documentation) so things like discharge_disposition_id = 7 become "left against medical advice" instead of a meaningless number.

Real role-separated auth, not just a UI toggle — doctor and patient accounts get different API responses at the backend level (_to_doctor_view vs _to_patient_view), so a patient account can never retrieve SHAP data or the clinical note even by hitting the API directly.

Dataset

Trained on the UCI Diabetes 130-US hospitals dataset — real de-identified inpatient encounter records from 130 US hospitals, 1999–2008. The dataset is used only to train the models; no original patient records are stored, queried, or served back through this application. All predictions in the live demo are run against hypothetical patient details entered through the form.

Architecture
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Streamlit  │─────▶│   FastAPI    │─────▶│ PostgreSQL  │
│  Frontend   │◀─────│   Backend    │◀─────│   (Neon)    │
└─────────────┘      └──────┬───────┘      └─────────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              SHAP + Cascade      Gemini API
              Models (joblib)   (dual prompts,
                                 run concurrently)

Backend: FastAPI, SQLAlchemy, PostgreSQL. JWT access tokens in httponly cookies + rotating SHA-256-hashed refresh tokens (never stores raw refresh tokens). Role-based access control (doctor / patient). Rate-limited signup and prediction endpoints (slowapi) to prevent quota abuse.

ML inference: reproduces the training notebook's exact preprocessing (category encoding, ICD-9 diagnosis grouping, scaling) at serve time from a single saved preprocessing bundle, so raw form input maps correctly to whichever of the three encodings (native/sklearn/logreg) the winning model per stage actually expects.

Frontend: Streamlit with custom CSS (not default styling), medication multi-select instead of per-drug dropdowns, doctor-patient account linking, separate history/report views per role.

Deployment: Frontend on Streamlit Community Cloud, backend on FastAPI Cloud, database on Neon serverless Postgres — three independently-managed free-tier services rather than a single Docker Compose stack, with UptimeRobot keeping the backend warm to minimize cold-start latency.

Tech stack
Python · FastAPI · PostgreSQL · SQLAlchemy · Streamlit · Docker · Docker Compose · CatBoost · XGBoost · LightGBM · scikit-learn · SHAP · Google Gemini API · JWT · bcrypt · slowapi

Model artifacts (backend/artifacts/*.joblib) are included in this repo. To regenerate them from scratch, run the training notebook end-to-end — the final cell saves both stage models, both SHAP explainers, and the preprocessing bundle.

Known limitations

This is a portfolio/demo project, not a production clinical system — no doctor identity verification, no patient-consent flow for account linking, and the free-tier database isn't durable long-term. These were deliberate scope decisions for a demo.

License
MIT
