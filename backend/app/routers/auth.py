import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, Response, Request
from sqlalchemy.orm import Session

from .. import models, schemas, security
from ..database import get_db
from ..config import settings
from ..limiter import limiter

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_auth_cookies(response: Response, access_token: str, refresh_token_raw: str):
    response.set_cookie(
        key=settings.cookie_name,
        value=access_token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token_raw,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=settings.refresh_token_expire_days * 24 * 3600,
        path="/auth/refresh",
    )


@router.post("/signup", response_model=schemas.UserOut, status_code=201)
@limiter.limit("10/hour")
def signup(request: Request, payload: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = models.User(
        email=payload.email,
        hashed_password=security.hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=schemas.UserOut)
@limiter.limit("20/minute")
def login(request: Request, payload: schemas.UserLogin, response: Response, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == payload.email).first()
    if not user or not security.verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")
    if user.role != payload.role:
        raise HTTPException(
            status_code=403,
            detail=f"This account is registered as '{user.role}', not '{payload.role}'. Log in with the correct role."
        )

    access_token = security.create_access_token(user.id)
    refresh_raw, refresh_hash = security.create_refresh_token()
    db.add(models.RefreshToken(
        user_id=user.id,
        token_hash=refresh_hash,
        expires_at=dt.datetime.utcnow() + dt.timedelta(days=settings.refresh_token_expire_days),
    ))
    db.commit()

    _set_auth_cookies(response, access_token, refresh_raw)
    return user


@router.post("/refresh", response_model=schemas.TokenOut)
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    raw = request.cookies.get("refresh_token")
    if not raw:
        raise HTTPException(status_code=401, detail="No refresh token")
    token_hash = security.hash_token(raw)
    record = db.query(models.RefreshToken).filter(
        models.RefreshToken.token_hash == token_hash,
        models.RefreshToken.revoked.is_(False),
    ).first()
    if not record or record.expires_at < dt.datetime.utcnow():
        raise HTTPException(status_code=401, detail="Refresh token invalid or expired")

    # rotate refresh token
    record.revoked = True
    new_raw, new_hash = security.create_refresh_token()
    db.add(models.RefreshToken(
        user_id=record.user_id,
        token_hash=new_hash,
        expires_at=dt.datetime.utcnow() + dt.timedelta(days=settings.refresh_token_expire_days),
    ))
    db.commit()

    access_token = security.create_access_token(record.user_id)
    _set_auth_cookies(response, access_token, new_raw)
    return schemas.TokenOut(access_token=access_token)


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    raw = request.cookies.get("refresh_token")
    if raw:
        token_hash = security.hash_token(raw)
        record = db.query(models.RefreshToken).filter(models.RefreshToken.token_hash == token_hash).first()
        if record:
            record.revoked = True
            db.commit()
    response.delete_cookie(settings.cookie_name, path="/")
    response.delete_cookie("refresh_token", path="/auth/refresh")
    return {"detail": "Logged out"}


@router.get("/me", response_model=schemas.UserOut)
def me(user: models.User = Depends(security.get_current_user)):
    return user
