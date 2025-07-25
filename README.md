# MedAI – Hospital Readmission Risk Predictor

A modern, full-stack web app that predicts hospital readmission risk using machine learning. Built for hackathons with React, Tailwind CSS, FastAPI, and scikit-learn.

## Features
- Landing page with branding and dark/light mode
- Signup & Login (with validation)
- Dashboard: patient info form, AI prediction, feature importance chart
- Medical Q&A chatbot (FAQ-based, easy to upgrade to real AI)
- Responsive, modern UI (hospital blue theme)

## Tech Stack
- **Frontend:** React, Tailwind CSS, Framer Motion, Recharts, Lucide Icons
- **Backend:** FastAPI (Python), SQLite, scikit-learn, pandas, SQLAlchemy
- **ML Model:** Trained on healthcare_dataset.csv

## Setup Instructions

### 1. Backend
```bash
cd backend
python -m venv venv
# Activate venv:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt  # (or install FastAPI, scikit-learn, etc. manually)
python train_model.py           # Trains and saves model.pkl
uvicorn main:app --reload      # Starts backend at http://localhost:8000
```

### 2. Frontend
```bash
cd medai-frontend
npm install
npm start                      # Starts frontend at http://localhost:3000
```

## Usage (Demo Steps)
1. Open the app in your browser (http://localhost:3000)
2. Signup for a new account
3. Login
4. Enter patient info and click "Predict Risk"
5. View AI result and feature chart
6. Click the 💬 button to chat with the MedAI Assistant

---

## PPT Presentation -

https://gamma.app/docs/Predicting-Hospital-Readmission-Risk-60l9v3o00zvgpap?mode=doc
**For hackathon/demo only. Not for real medical use.** 
