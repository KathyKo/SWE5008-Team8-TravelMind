"""
backend/routers/auth.py - simple auth endpoints

Provides:
  POST /auth/register
  POST /auth/login
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db import crud

router = APIRouter()


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=6, max_length=128)


class AuthResponse(BaseModel):
    user_id: str
    username: str
    message: str


@router.post("/register", response_model=AuthResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user = crud.create_user(db, request.username.strip(), request.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return AuthResponse(
        user_id=str(user.id),
        username=user.username,
        message="Register success",
    )


@router.post("/login", response_model=AuthResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = crud.authenticate_user(db, request.username.strip(), request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return AuthResponse(
        user_id=str(user.id),
        username=user.username,
        message="Login success",
    )


@router.get("/health")
def health():
    return {"status": "ok", "router": "auth"}
