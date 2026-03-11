import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import numpy as np
import json
import os

# ── Initialize Firebase ────────────────────────────────────────────
def init_firebase(key_path: str = "firebase-key.json"):
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()


# ── User Management ────────────────────────────────────────────────
def create_user(db, user_id: str, name: str, age: int, gender: str):
    doc_ref = db.collection("users").document(user_id)
    doc_ref.set({
        "name"      : name,
        "age"       : age,
        "gender"    : gender,
        "created_at": datetime.now().isoformat(),
        "sessions"  : 0
    }, merge=True)
    return doc_ref


def get_user(db, user_id: str):
    doc = db.collection("users").document(user_id).get()
    return doc.to_dict() if doc.exists else None


def get_all_users(db):
    users = db.collection("users").stream()
    return {u.id: u.to_dict() for u in users}


# ── Session Management ─────────────────────────────────────────────
def save_session(db, user_id: str, session_data: dict):
    """Save a screening session to Firestore"""
    session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_ref = (
        db.collection("users")
          .document(user_id)
          .collection("sessions")
          .document(session_id)
    )

    session_data.update({
        "session_id"  : session_id,
        "timestamp"   : datetime.now().isoformat(),
        "date"        : datetime.now().strftime("%Y-%m-%d"),
        "time"        : datetime.now().strftime("%H:%M")
    })

    session_ref.set(session_data)

    # Update session count
    db.collection("users").document(user_id).update({
        "sessions": firestore.INCREMENT(1)
    })

    return session_id


def get_sessions(db, user_id: str, limit: int = 50):
    """Get all sessions for a user ordered by date"""
    sessions = (
        db.collection("users")
          .document(user_id)
          .collection("sessions")
          .order_by("timestamp", direction=firestore.Query.DESCENDING)
          .limit(limit)
          .stream()
    )
    return [s.to_dict() for s in sessions]


# ── Analytics ─────────────────────────────────────────────────────
def get_trend_data(db, user_id: str):
    """Get risk scores over time for trend analysis"""
    sessions = get_sessions(db, user_id, limit=50)
    if not sessions:
        return [], [], []

    # Sort by timestamp ascending
    sessions = sorted(sessions, key=lambda x: x.get("timestamp", ""))

    dates  = [s.get("date", "")       for s in sessions]
    scores = [s.get("risk_score", 0)  for s in sessions]
    labels = [s.get("risk_level", "") for s in sessions]

    return dates, scores, labels


def compute_trajectory(scores: list) -> str:
    """Determine if mental health is improving, worsening or stable"""
    if len(scores) < 2:
        return "insufficient_data"

    recent    = scores[-3:] if len(scores) >= 3 else scores
    earlier   = scores[:-3] if len(scores) >= 3 else scores[:1]

    avg_recent  = np.mean(recent)
    avg_earlier = np.mean(earlier) if earlier else avg_recent
    diff        = avg_recent - avg_earlier

    if diff < -0.08:
        return "improving"
    elif diff > 0.08:
        return "worsening"
    else:
        return "stable"


def detect_anomaly(scores: list, threshold: float = 0.20) -> bool:
    """Detect if latest score is anomalous vs baseline"""
    if len(scores) < 3:
        return False
    baseline    = np.mean(scores[:-1])
    latest      = scores[-1]
    return abs(latest - baseline) > threshold


def map_to_phq9(risk_score: float) -> dict:
    """Map model risk score to PHQ-9 equivalent range"""
    phq9_score = int(risk_score * 27)

    if phq9_score <= 4:
        severity = "Minimal"
        color    = "#4CAF50"
        advice   = "No action needed. Keep monitoring."
    elif phq9_score <= 9:
        severity = "Mild"
        color    = "#8BC34A"
        advice   = "Watchful waiting, repeat PHQ-9 at follow-up."
    elif phq9_score <= 14:
        severity = "Moderate"
        color    = "#FF9800"
        advice   = "Consider counseling and follow up."
    elif phq9_score <= 19:
        severity = "Moderately Severe"
        color    = "#FF5722"
        advice   = "Active treatment recommended."
    else:
        severity = "Severe"
        color    = "#F44336"
        advice   = "Immediate treatment and close follow-up."

    return {
        "phq9_score": phq9_score,
        "severity"  : severity,
        "color"     : color,
        "advice"    : advice
    }


def get_weekly_summary(db, user_id: str) -> dict:
    """Get summary stats for the last 7 days"""
    sessions      = get_sessions(db, user_id)
    week_ago      = datetime.now() - timedelta(days=7)
    recent        = [
        s for s in sessions
        if datetime.fromisoformat(
            s.get("timestamp", datetime.now().isoformat())
        ) > week_ago
    ]

    if not recent:
        return {"sessions": 0, "avg_risk": None, "trend": "no_data"}

    scores = [s.get("risk_score", 0) for s in recent]
    return {
        "sessions"  : len(recent),
        "avg_risk"  : np.mean(scores),
        "max_risk"  : max(scores),
        "min_risk"  : min(scores),
        "trend"     : compute_trajectory(scores)
    }
