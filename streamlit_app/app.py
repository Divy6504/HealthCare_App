import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="ReadmitRisk | Diabetes Readmission Predictor",
                    page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
  --bg: #0b1220; --card: #121b2e; --accent: #4f8cff; --accent2: #22d3a5;
  --danger: #ff5c6c; --text: #e7ecf7; --muted: #92a0bd;
}
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #16213a 0%, var(--bg) 55%); }
.main-title { font-size: 2.1rem; font-weight: 800; color: var(--text); letter-spacing: -0.02em; margin-bottom: 0.1rem; }
.subtitle { color: var(--muted); font-size: 0.95rem; margin-bottom: 1.5rem; }
.card {
  background: var(--card); border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px; padding: 1.4rem 1.6rem; box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  margin-bottom: 1rem;
}
.badge { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.02em; }
.badge-safe { background: rgba(34,211,165,0.15); color: var(--accent2); border: 1px solid rgba(34,211,165,0.35); }
.badge-warn { background: rgba(255,196,77,0.15); color: #ffc44d; border: 1px solid rgba(255,196,77,0.35); }
.badge-danger { background: rgba(255,92,108,0.15); color: var(--danger); border: 1px solid rgba(255,92,108,0.35); }
.llm-note {
  background: linear-gradient(135deg, rgba(79,140,255,0.10), rgba(34,211,165,0.06));
  border-left: 3px solid var(--accent); border-radius: 10px; padding: 1rem 1.2rem;
  color: var(--text); line-height: 1.6; font-size: 1.02rem;
}
.stButton>button {
  background: linear-gradient(135deg, var(--accent), #6f6bff); color: white; border: none;
  border-radius: 10px; padding: 0.55rem 1.4rem; font-weight: 600; transition: transform 0.15s ease;
}
.stButton>button:hover { transform: translateY(-1px); filter: brightness(1.08); }
section[data-testid="stSidebar"] { background: #0e1626; border-right: 1px solid rgba(255,255,255,0.06); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- session ---
if "http" not in st.session_state:
    st.session_state.http = requests.Session()
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "predict"


def api(method, path, timeout=30, **kwargs):
    try:
        r = st.session_state.http.request(method, f"{API_BASE}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the backend server. Make sure `docker compose up` is running and try again.")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("The server took too long to respond. Please try again.")
        st.stop()

    if r.status_code == 401 and path not in ("/auth/refresh", "/auth/login"):
        try:
            refresh_r = st.session_state.http.request("POST", f"{API_BASE}/auth/refresh", timeout=30)
        except requests.exceptions.RequestException:
            refresh_r = None
        if refresh_r is not None and refresh_r.status_code == 200:
            r = st.session_state.http.request(method, f"{API_BASE}{path}", timeout=timeout, **kwargs)

    if r.status_code == 429:
        st.warning("Too many attempts in a short time — please wait a few seconds and try again.")
    return r

def safe_json(r):
    try:
        return r.json()
    except ValueError:
        return {}


def try_restore_session():
    if st.session_state.user is None:
        r = api("GET", "/auth/me")
        if r.status_code == 200:
            st.session_state.user = r.json()


try_restore_session()


# ----------------------------------------------------------------- auth ----
def auth_screen():
    st.markdown('<div class="main-title">🩺 ReadmitRisk</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Diabetes 30-day readmission risk platform</div>', unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            login_role = st.selectbox("Logging in as", ["Doctor / Care team member", "Patient"])
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            role_value = "doctor" if login_role.startswith("Doctor") else "patient"
            r = api("POST", "/auth/login", json={"email": email, "password": password, "role": role_value})
            if r.status_code == 200:
                st.session_state.user = r.json()
                st.rerun()
            else:
                st.error(safe_json(r).get("detail", "Login failed"))

    with tab_signup:
        with st.form("signup_form"):
            name = st.text_input("Full name")
            email2 = st.text_input("Email ", key="signup_email")
            password2 = st.text_input("Password (min 8 chars)", type="password", key="signup_pw")
            role = st.selectbox("I am a...", ["Doctor / Care team member", "Patient"])
            submitted2 = st.form_submit_button("Create account")
        if submitted2:
            role_value = "doctor" if role.startswith("Doctor") else "patient"
            r = api("POST", "/auth/signup",
                    json={"email": email2, "password": password2, "full_name": name, "role": role_value})
            if r.status_code == 201:
                st.success("Account created. Please log in.")
            else:
                st.error(safe_json(r).get("detail", "Signup failed"))


# -------------------------------------------------------------- sidebar ----
def sidebar():
    role = st.session_state.user["role"]
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user['full_name'] or st.session_state.user['email']}")
        st.caption(f"{st.session_state.user['email']} · {role.capitalize()}")
        st.divider()
        if role == "doctor":
            if st.button("🔮 New Prediction", use_container_width=True):
                st.session_state.page = "predict"
            if st.button("📜 Patient History", use_container_width=True):
                st.session_state.page = "history"
        else:
            if st.button("📋 My Reports", use_container_width=True):
                st.session_state.page = "history"
        st.divider()
        if st.button("Log out", use_container_width=True):
            api("POST", "/auth/logout")
            st.session_state.user = None
            st.session_state.http = requests.Session()
            st.rerun()


# ------------------------------------------------------------- constants ---
AGE_BUCKETS = [f"[{i}-{i+10})" for i in range(0, 100, 10)]
COUNT_0_20 = list(range(0, 21))
COUNT_0_10 = list(range(0, 11))


# --------------------------------------------------------- predict page ----
def predict_page():
    st.markdown('<div class="main-title">New Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter encounter details to estimate readmission risk</div>',
                unsafe_allow_html=True)

    patients_r = api("GET", "/patients/")
    patient_options = {"— No linked account (record only) —": None}
    if patients_r.status_code == 200:
        for p in patients_r.json():
            label = f"{p['full_name'] or p['email']} ({p['email']})"
            patient_options[label] = p["email"]

    with st.form("predict_form"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            patient_ref = st.text_input("Patient name / reference")
            patient_choice = st.selectbox(
                "Link to a previously-seen patient account",
                list(patient_options.keys())
            )
            new_patient_email = st.text_input(
                "Or link a new patient by email (first time only)",
                help="If this is the first prediction for this patient, enter their registered account email here instead of using the dropdown above."
            )
            patient_email = new_patient_email.strip() or patient_options[patient_choice]
            race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"])
            gender = st.selectbox("Gender", ["Female", "Male"])
        with c2:
            age = st.selectbox("Age bracket", AGE_BUCKETS, index=6)
            time_in_hospital = st.selectbox("Time in hospital (days)", list(range(1, 15)), index=2)
            num_lab_procedures = st.selectbox("Num lab procedures", list(range(0, 151)), index=40)
            num_procedures = st.selectbox("Num procedures", COUNT_0_10, index=1)
        with c3:
            num_medications = st.selectbox("Num medications", list(range(0, 81)), index=15)
            number_outpatient = st.selectbox("Outpatient visits (prior yr)", COUNT_0_20, index=0)
            number_emergency = st.selectbox("Emergency visits (prior yr)", COUNT_0_20, index=0)
            number_inpatient = st.selectbox("Inpatient visits (prior yr)", COUNT_0_20, index=0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            admission_type_id = st.selectbox("Admission type ID", list(range(1, 9)), index=0)
            discharge_disposition_id = st.selectbox("Discharge disposition ID", list(range(1, 30)), index=0)
            admission_source_id = st.selectbox("Admission source ID", list(range(1, 26)), index=6)
            number_diagnoses = st.selectbox("Number of diagnoses", COUNT_0_20[1:], index=6)
        with c5:
            diag_1 = st.selectbox("Primary diagnosis category",
                                   ["250.01 (Diabetes)", "401.9 (Circulatory)", "486 (Respiratory)",
                                    "578 (Digestive)", "805 (Injury)", "715 (Musculoskeletal)",
                                    "590 (Genitourinary)", "150 (Neoplasms)", "V45 (Other)", "? (Unknown)"])
            diag_2 = st.selectbox("Secondary diagnosis category",
                                   ["401.9 (Circulatory)", "250.01 (Diabetes)", "486 (Respiratory)",
                                    "578 (Digestive)", "805 (Injury)", "715 (Musculoskeletal)",
                                    "590 (Genitourinary)", "150 (Neoplasms)", "V45 (Other)", "? (Unknown)"])
            diag_3 = st.selectbox("Additional diagnosis category",
                                   ["? (Unknown)", "401.9 (Circulatory)", "250.01 (Diabetes)",
                                    "486 (Respiratory)", "578 (Digestive)", "805 (Injury)",
                                    "715 (Musculoskeletal)", "590 (Genitourinary)", "150 (Neoplasms)",
                                    "V45 (Other)"])
        with c6:
            max_glu_serum = st.selectbox("Max glucose serum", ["None", "Norm", ">200", ">300"])
            A1Cresult = st.selectbox("A1C result", ["None", "Norm", ">7", ">8"])
            change = st.selectbox("Medication changed", ["No", "Ch"])
            diabetesMed = st.selectbox("On diabetes medication", ["Yes", "No"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Medication status — select which medications fall into each category")
        med_names = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
                     "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "acarbose",
                     "miglitol", "tolazamide", "insulin"]
        mc1, mc2 = st.columns(2)
        with mc1:
            up_meds = st.multiselect("Dosage increased (Up)", med_names)
            down_meds = st.multiselect("Dosage decreased (Down)", med_names)
        with mc2:
            steady_meds = st.multiselect("Unchanged (Steady)", med_names)
            st.caption("Any medication not selected above is treated as not prescribed (No).")
        med_values = {med: "No" for med in med_names}
        for med in steady_meds:
            med_values[med] = "Steady"
        for med in down_meds:
            med_values[med] = "Down"
        for med in up_meds:
            med_values[med] = "Up"
        glyburide_metformin = st.selectbox("glyburide-metformin", ["No", "Down", "Steady", "Up"])
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Predict readmission risk", use_container_width=True)

    if submitted:
        payload = {
            "patient_ref": patient_ref or None,
            "patient_email": patient_email or None,
            "race": race, "gender": gender, "age": age,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "diag_1": diag_1.split(" ")[0], "diag_2": diag_2.split(" ")[0], "diag_3": diag_3.split(" ")[0],
            "number_diagnoses": number_diagnoses,
            "max_glu_serum": max_glu_serum, "A1Cresult": A1Cresult,
            "change": change, "diabetesMed": diabetesMed,
            "glyburide-metformin": glyburide_metformin,
            **med_values,
        }
        with st.spinner("Analyzing patient data..."):
            r = api("POST", "/predictions/", json=payload, timeout=60)
        if r.status_code == 200:
            render_doctor_result(r.json())
        else:
            st.error(f"Prediction failed: {r.text}")


def render_doctor_result(data):
    label = data["final_label"]
    badge_class = {"NO": "badge-safe", ">30": "badge-warn", "<30": "badge-danger"}.get(label, "badge-warn")
    label_text = {"NO": "Not Readmitted", ">30": "Readmitted after 30 days", "<30": "Readmitted within 30 days"}
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<span class="badge {badge_class}">{label_text.get(label, label)}</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Overall readmission risk", f"{data['stage1_probability']*100:.1f}%")
    if data["stage2_probability"] is not None:
        c2.metric("Risk of readmission within 30 days", f"{data['stage2_probability']*100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    if data.get("top_shap_features"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Key factors in this assessment")
        df = pd.DataFrame(data["top_shap_features"])
        fig = go.Figure(go.Bar(
            x=df["shap_value"], y=df["feature"], orientation="h",
            marker_color=["#22d3a5" if v < 0 else "#ff5c6c" for v in df["shap_value"]],
            text=df["value"], textposition="outside",
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if data.get("doctor_note"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Clinical note")
        st.markdown(f'<div class="llm-note">{data["doctor_note"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_patient_result(data):
    label = data["final_label"]
    badge_class = {"NO": "badge-safe", ">30": "badge-warn", "<30": "badge-danger"}.get(label, "badge-warn")
    label_text = {"NO": "Low current risk", ">30": "Some risk noted", "<30": "Higher risk noted"}
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<span class="badge {badge_class}">{label_text.get(label, label)}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if data.get("patient_report"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What this means for you")
        st.markdown(f'<div class="llm-note">{data["patient_report"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------- history page ----
def history_page():
    role = st.session_state.user["role"]
    title = "Patient History" if role == "doctor" else "My Reports"
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)

    r = api("GET", "/predictions/")
    if r.status_code != 200:
        st.error(f"Could not load history (status {r.status_code}): {r.text}")
        return
    records = r.json()
    if not records:
        if role == "patient":
            st.info("No reports yet. Your doctor hasn't linked a report to your account — "
                     "ask them to select your account when running your assessment.")
        else:
            st.info("No predictions recorded yet.")
        return

    # Build a plain HTML table manually — avoids pandas/pyarrow's native rendering path entirely.
    rows_html = ""
    for rec in records:
        when = str(rec.get("created_at", ""))[:16].replace("T", " ")
        result = str(rec.get("final_label", ""))
        if role == "doctor":
            patient = str(rec.get("patient_ref") or "—")
            risk = rec.get("stage1_probability")
            risk_str = f"{float(risk)*100:.1f}%" if risk is not None else "—"
            rows_html += f"<tr><td>{patient}</td><td>{result}</td><td>{risk_str}</td><td>{when}</td></tr>"
        else:
            rows_html += f"<tr><td>{result}</td><td>{when}</td></tr>"

    if role == "doctor":
        header = "<tr><th>Patient</th><th>Result</th><th>Risk Level</th><th>When</th></tr>"
    else:
        header = "<tr><th>Result</th><th>When</th></tr>"

    table_html = f"""
    <div class="card">
    <table style="width:100%; border-collapse: collapse; color: var(--text);">
    <thead style="text-align:left; border-bottom: 1px solid rgba(255,255,255,0.15);">{header}</thead>
    <tbody>{rows_html}</tbody>
    </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    def _row_label(rec):
        when = str(rec.get("created_at", ""))[:16].replace("T", " ")
        return f"{rec.get('patient_ref') or rec['id'][:8]} — {when}" if role == "doctor" else f"Report — {when}"

    ids = {_row_label(rec): rec["id"] for rec in records}
    choice = st.selectbox("View details", list(ids.keys()))
    if st.button("Load"):
        detail = api("GET", f"/predictions/{ids[choice]}")
        if detail.status_code == 200:
            data = detail.json()
            if role == "doctor":
                render_doctor_result(data)
            else:
                render_patient_result(data)


# ---------------------------------------------------------------- main -----
if st.session_state.user is None:
    auth_screen()
else:
    sidebar()
    role = st.session_state.user["role"]
    if role == "doctor":
        if st.session_state.page == "predict":
            predict_page()
        else:
            history_page()
    else:
        history_page()
