# ReadmitRisk — Diabetes Readmission Predictor

## Structure
```
healthcare_with_save.ipynb   # your notebook + appended save-artifacts cell
backend/                     # FastAPI: auth, cascade model, SHAP, Gemini messages
streamlit_app/                # Streamlit UI
docker-compose.yml
```

## 1. Generate artifacts
Run `healthcare_with_save.ipynb` end to end (Kaggle/Colab). It writes:
`artifacts/model_stage1.joblib`, `model_stage2.joblib`, `preprocessing_bundle.joblib`,
and SHAP explainers (if winner isn't LogisticRegression).
Copy the `artifacts/` folder into `backend/artifacts/`.

## 2. Local run
```bash
cp backend/.env.example backend/.env   # fill in JWT_SECRET, GEMINI_API_KEY
export GEMINI_API_KEY=your-key
docker compose up --build
```
Backend: http://localhost:8000/health · Frontend: http://localhost:8501

## 3. Free deployment that doesn't sleep

Render/Railway free tiers sleep or expire. The only genuinely free, always-on option is
**Oracle Cloud Free Tier** (Always Free ARM Ampere VM — 4 OCPU / 24GB RAM, free forever, no sleep).

Steps:
1. Create an Oracle Cloud account → Compute → create an **Always Free** Ampere A1 instance (Ubuntu 22.04).
2. Open ports 8000 and 8501 in the VM's security list / firewall (`sudo ufw allow 8000,8501/tcp`).
3. SSH in, install Docker: `curl -fsSL https://get.docker.com | sh`.
4. `git clone` your repo, add `backend/.env` and `backend/artifacts/`.
5. `docker compose up -d --build`.
6. Point a free domain (e.g. Cloudflare + a free subdomain, or the VM's public IP) at it; put
   Caddy or Nginx in front for HTTPS (needed for `secure` cookies) — a 5-line Caddyfile with
   automatic Let's Encrypt certs is enough.
7. Set `COOKIE_SECURE=true` and `FRONTEND_ORIGIN` to your real HTTPS frontend URL once behind HTTPS.

Fallback if you don't want to set up a VM: Render's free web service does sleep after 15 min
idle (cold start ~30-50s on next request) — acceptable if "doesn't sleep" isn't a hard requirement,
and it's a 10-minute deploy from GitHub with the same Dockerfiles.

## Security notes
- Passwords hashed with bcrypt.
- Access token: short-lived JWT in an `httponly`, `samesite=lax` cookie.
- Refresh token: opaque random token, stored server-side only as a SHA-256 hash, rotated on each use.
- Login endpoint rate-limited (5/min/IP) via slowapi.
- CORS locked to `FRONTEND_ORIGIN` in production (not `*`).
- Set `COOKIE_SECURE=true` once served over HTTPS — required for cookies to actually be sent cross-site.
