from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import User, LoginHistory, Base
from jose import jwt, JWTError
from datetime import datetime, timedelta
from pydantic import BaseModel
from summarizer import generate_summary
import pdfplumber
import io

router = APIRouter()

# 🔐 SECURITY
security = HTTPBearer()

SECRET_KEY = "mysecretkey123"
ALGORITHM = "HS256"

# DB init
Base.metadata.create_all(bind=engine)

# DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# SCHEMAS
class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

# JWT
def create_token(data: dict):
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=2)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# SIGNUP
@router.post("/signup")
def signup(data: SignupRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(
        username=data.username,
        email=data.email,
        password=data.password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User created successfully"}

# LOGIN 
@router.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        User.email == data.email,
        User.password == data.password
    ).first()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # ✅ LOGIN HISTORY SAVE
    history = LoginHistory(
        email=user.email,
        username=user.username,
        login_time=datetime.now()
    )
    db.add(history)
    db.commit()

    token = create_token({"email": user.email})

    return {
        "access_token": token,
        "token_type": "bearer"
    }

# AUTH
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------------- SUMMARIZE ----------------
@router.post("/summarize")
async def summarize(
    text: str = Form(None),
    file: UploadFile = File(None),
    user=Depends(get_current_user)
):
    if file:
        try:
            contents = await file.read()
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except:
            raise HTTPException(status_code=400, detail="Invalid PDF file")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    summary, keywords = generate_summary(text)

    return {
        "summary": summary,
        "keywords": keywords
    }

# 🔥 ADMIN DASHBOARD (REAL DATA)
@router.get("/admin/dashboard")
def admin_dashboard(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["email"] != "admin@gmail.com":
        raise HTTPException(status_code=403, detail="Not authorized")

    total_users = db.query(User).count()

    history = db.query(LoginHistory).order_by(LoginHistory.id.desc()).all()

    return {
        "total_users": total_users,
        "total_logins": len(history),
        "login_history": [
            {
                "username": h.username,
                "email": h.email,
                "login_time": h.login_time
            }
            for h in history
        ]
    }