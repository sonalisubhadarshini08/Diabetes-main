from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
import joblib
import numpy as np

# Load model
model_data = joblib.load('model.pkl')
model = model_data['model']
feature_names = model_data['features']

# Database setup
SQLALCHEMY_DATABASE_URL = 'sqlite:///./users.db'
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class PredictRequest(BaseModel):
    # Dynamically add all features as optional fields
    # For hackathon, use dict
    data: dict

class PredictResponse(BaseModel):
    risk: str
    confidence: float
    feature_importances: dict

# Dependency

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Auth endpoints
@app.post('/signup')
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = get_password_hash(req.password)
    new_user = User(name=req.name, email=req.email, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

@app.post('/login')
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "name": user.name, "email": user.email}

# Prediction endpoint
@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    # Prepare input in correct order
    input_data = []
    for feat in feature_names:
        val = req.data.get(feat, -1)
        input_data.append(val)
    arr = np.array(input_data).reshape(1, -1)
    proba = model.predict_proba(arr)[0]
    pred = model.predict(arr)[0]
    risk = "High Risk" if pred == 1 else "Low Risk"
    confidence = float(np.max(proba))
    # Feature importances
    importances = dict(zip(feature_names, model.feature_importances_))
    return PredictResponse(risk=risk, confidence=confidence, feature_importances=importances)

@app.get('/')
def root():
    return {"message": "MedAI backend is running"} 