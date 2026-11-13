"""
train_model.py
----------------
Reads attendance data from the SQLite DB, engineers per-student
features, trains a Logistic Regression model and a Random Forest
model to predict which students are at risk of falling below the
attendance threshold, evaluates both, and saves the better model
(plus the scaler and feature list) to disk with pickle.

Run:  python train_model.py
"""

import sqlite3
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "attendance.db"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

ATTENDANCE_THRESHOLD = 75.0   # % below which a student is "at risk"
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. Load raw data
# ----------------------------------------------------------------------
def load_raw(conn):
    query = """
        SELECT
            s.student_id,
            s.year_of_study,
            e.enrollment_id,
            e.course_id,
            a.class_date,
            a.status
        FROM students s
        JOIN enrollments e ON e.student_id = s.student_id
        JOIN attendance a ON a.enrollment_id = e.enrollment_id
        ORDER BY s.student_id, a.class_date
    """
    return pd.read_sql_query(query, conn, parse_dates=["class_date"])


# ----------------------------------------------------------------------
# 2. Feature engineering (per student, aggregated across all courses)
# ----------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["present_flag"] = df["status"].isin(["Present", "Late"]).astype(int)
    df["absent_flag"] = (df["status"] == "Absent").astype(int)

    records = []
    for sid, g in df.groupby("student_id"):
        g = g.sort_values("class_date")
        total_classes = len(g)
        attended = g["present_flag"].sum()
        attendance_pct = 100.0 * attended / total_classes if total_classes else 0

        # split into first half / second half of semester to capture trend
        midpoint = total_classes // 2
        first_half = g.iloc[:midpoint]
        second_half = g.iloc[midpoint:]
        first_pct = (100.0 * first_half["present_flag"].sum() / len(first_half)
                     if len(first_half) else attendance_pct)
        second_pct = (100.0 * second_half["present_flag"].sum() / len(second_half)
                      if len(second_half) else attendance_pct)
        trend = second_pct - first_pct  # negative = declining attendance

        # longest consecutive-absence streak
        longest_streak = 0
        current_streak = 0
        for flag in g["absent_flag"]:
            if flag == 1:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 0

        # last-4-weeks attendance (most recent ~12 sessions across courses)
        recent = g.tail(12)
        recent_pct = (100.0 * recent["present_flag"].sum() / len(recent)
                      if len(recent) else attendance_pct)

        num_courses = g["course_id"].nunique()
        year_of_study = g["year_of_study"].iloc[0]
        late_pct = 100.0 * (g["status"] == "Late").sum() / total_classes if total_classes else 0
        excused_pct = 100.0 * (g["status"] == "Excused").sum() / total_classes if total_classes else 0

        label = 1 if attendance_pct < ATTENDANCE_THRESHOLD else 0

        records.append({
            "student_id": sid,
            "total_classes": total_classes,
            "attendance_pct": round(attendance_pct, 2),
            "first_half_pct": round(first_pct, 2),
            "second_half_pct": round(second_pct, 2),
            "trend": round(trend, 2),
            "longest_absence_streak": longest_streak,
            "recent_attendance_pct": round(recent_pct, 2),
            "num_courses": num_courses,
            "year_of_study": year_of_study,
            "late_pct": round(late_pct, 2),
            "excused_pct": round(excused_pct, 2),
            "at_risk": label,
        })

    return pd.DataFrame(records)


FEATURE_COLUMNS = [
    "first_half_pct", "second_half_pct", "trend", "longest_absence_streak",
    "recent_attendance_pct", "num_courses", "year_of_study",
    "late_pct", "excused_pct",
]
# Note: attendance_pct itself is deliberately EXCLUDED from model features
# since it (near-)perfectly determines the label by construction; the model
# instead learns the underlying behavioral signals (trend, streaks, recent
# attendance) that predict risk, which is what makes it useful for early
# warning before a student's overall percentage actually drops.


# ----------------------------------------------------------------------
# 3. Train & evaluate
# ----------------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:10s}: {v:.3f}")
    print("  Confusion matrix [ [TN FP] [FN TP] ]:")
    print(" ", confusion_matrix(y_test, preds).tolist())
    return metrics


def main():
    conn = sqlite3.connect(DB_PATH)
    raw = load_raw(conn)
    features_df = build_features(raw)
    conn.close()

    print(f"Built feature table for {len(features_df)} students.")
    print(f"At-risk rate (attendance < {ATTENDANCE_THRESHOLD}%): "
          f"{features_df['at_risk'].mean()*100:.1f}%")

    X = features_df[FEATURE_COLUMNS].values
    y = features_df["at_risk"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Logistic Regression ---
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    lr_metrics = evaluate("Logistic Regression", log_reg, X_test_scaled, y_test)

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)  # tree models don't need scaling
    rf_metrics = evaluate("Random Forest", rf, X_test, y_test)

    # Feature importance from RF
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)
    print("\nRandom Forest feature importances:")
    print(importances.round(3).to_string())

    # --- Pick the better model by F1 score (good for imbalanced classes) ---
    best_name = "random_forest" if rf_metrics["f1"] >= lr_metrics["f1"] else "logistic_regression"
    best_model = rf if best_name == "random_forest" else log_reg
    print(f"\nSelected model: {best_name}")

    # --- Save artifacts ---
    with open(MODEL_DIR / "risk_model.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "model_name": best_name,
            "scaler": scaler,                 # only used if model is logistic_regression
            "feature_columns": FEATURE_COLUMNS,
            "threshold": ATTENDANCE_THRESHOLD,
        }, f)

    features_df.to_csv(MODEL_DIR / "student_features.csv", index=False)

    metrics_df = pd.DataFrame([lr_metrics, rf_metrics])
    metrics_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    print(f"\nSaved model artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
