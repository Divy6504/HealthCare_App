from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .database import Base, engine
from .config import settings
from . import ml_service
from .limiter import limiter
from .routers import auth, predictions, patients

app = FastAPI(title="Diabetes Readmission Risk API", version="1.0.0")
app.state.limiter = limiter
from slowapi.middleware import SlowAPIMiddleware
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin] if settings.frontend_origin != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    ml_service.load_artifacts()
    if not settings.gemini_api_key:
        print("[main] WARNING: GEMINI_API_KEY is not set — LLM messages will use generic fallback text.")
    else:
        print(f"[main] Gemini configured, key starts with: {settings.gemini_api_key[:6]}...")


@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(patients.router)
