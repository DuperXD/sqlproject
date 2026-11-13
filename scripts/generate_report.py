"""
generate_report.py
--------------------
Loads the trained model, scores every current student, writes the
predictions back into the risk_predictions table, and produces a
styled HTML report (+ CSV export) listing students flagged as likely
to fall below the attendance threshold, sorted by risk.

Run:  python generate_report.py
"""

import sqlite3
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "attendance.db"
MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

SEMESTER = "Fall-2026"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def score_students(conn, bundle):
    features_df = pd.read_csv(BASE_DIR / "models" / "student_features.csv")
    feat_cols = bundle["feature_columns"]
    model = bundle["model"]
    model_name = bundle["model_name"]

    X = features_df[feat_cols].values
    if model_name == "logistic_regression":
        X = bundle["scaler"].transform(X)

    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    features_df["predicted_risk"] = preds
    features_df["risk_probability"] = probs.round(4)
    features_df["model_used"] = model_name

    # attach student names / roll numbers
    students = pd.read_sql_query(
        "SELECT student_id, roll_no, name, program, year_of_study, email FROM students",
        conn
    )
    merged = features_df.merge(students, on="student_id", suffixes=("", "_s"))
    return merged, model_name


def write_predictions_to_db(conn, scored_df):
    cur = conn.cursor()
    cur.execute("DELETE FROM risk_predictions WHERE semester = ?", (SEMESTER,))
    rows = [
        (int(r.student_id), SEMESTER, float(r.attendance_pct),
         int(r.predicted_risk), float(r.risk_probability), r.model_used)
        for r in scored_df.itertuples()
    ]
    cur.executemany(
        "INSERT INTO risk_predictions "
        "(student_id, semester, attendance_pct, predicted_risk, risk_probability, model_used) "
        "VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def render_html(scored_df: pd.DataFrame, model_name: str, threshold: float) -> str:
    at_risk = scored_df[scored_df["predicted_risk"] == 1].sort_values(
        "risk_probability", ascending=False
    )
    safe_count = (scored_df["predicted_risk"] == 0).sum()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    def risk_band(p):
        if p >= 0.75:
            return "High", "#d64545"
        elif p >= 0.5:
            return "Medium", "#e0a02c"
        else:
            return "Low", "#3f8f5f"

    rows_html = ""
    for r in at_risk.itertuples():
        band, color = risk_band(r.risk_probability)
        rows_html += f"""
        <tr>
            <td>{r.roll_no}</td>
            <td>{r.name}</td>
            <td>{r.program}</td>
            <td>{r.year_of_study}</td>
            <td>{r.attendance_pct:.1f}%</td>
            <td>{r.trend:+.1f} pts</td>
            <td>{r.longest_absence_streak}</td>
            <td>{r.risk_probability*100:.1f}%</td>
            <td><span class="badge" style="background:{color}">{band}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Student Attendance Risk Report — {SEMESTER}</title>
<style>
    :root {{
        --bg: #f7f8fa; --card: #ffffff; --text: #1c2430; --muted: #6b7280;
        --accent: #2f5fb3; --border: #e5e7eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background: var(--bg); color: var(--text); margin: 0; padding: 32px;
    }}
    .container {{ max-width: 1080px; margin: 0 auto; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    .subtitle {{ color: var(--muted); margin-bottom: 24px; font-size: 14px; }}
    .stats {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
    .stat-card {{
        background: var(--card); border: 1px solid var(--border); border-radius: 10px;
        padding: 16px 20px; flex: 1; min-width: 160px;
    }}
    .stat-card .value {{ font-size: 26px; font-weight: 700; }}
    .stat-card .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .03em; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 10px; overflow: hidden; }}
    th, td {{ text-align: left; padding: 10px 14px; font-size: 13px; border-bottom: 1px solid var(--border); }}
    th {{ background: #eef2f8; color: #33475b; font-weight: 600; position: sticky; top: 0; }}
    tr:hover td {{ background: #fafbfc; }}
    .badge {{ color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
    .footer {{ margin-top: 24px; font-size: 12px; color: var(--muted); }}
    .section-title {{ font-size: 16px; font-weight: 600; margin: 28px 0 10px; }}
</style>
</head>
<body>
<div class="container">
    <h1>📋 Student Attendance Risk Report</h1>
    <div class="subtitle">Semester: {SEMESTER} &nbsp;·&nbsp; Generated: {generated} &nbsp;·&nbsp;
        Model: {model_name.replace('_',' ').title()} &nbsp;·&nbsp; Threshold: {threshold:.0f}%</div>

    <div class="stats">
        <div class="stat-card"><div class="value">{len(scored_df)}</div><div class="label">Total Students</div></div>
        <div class="stat-card"><div class="value" style="color:#d64545">{len(at_risk)}</div><div class="label">Flagged At-Risk</div></div>
        <div class="stat-card"><div class="value" style="color:#3f8f5f">{safe_count}</div><div class="label">On Track</div></div>
        <div class="stat-card"><div class="value">{len(at_risk)/len(scored_df)*100:.1f}%</div><div class="label">At-Risk Rate</div></div>
    </div>

    <div class="section-title">Students Flagged At-Risk (sorted by risk probability)</div>
    <table>
        <thead>
            <tr>
                <th>Roll No</th><th>Name</th><th>Program</th><th>Year</th>
                <th>Attendance %</th><th>Trend</th><th>Longest Absence Streak</th>
                <th>Risk Score</th><th>Band</th>
            </tr>
        </thead>
        <tbody>
            {rows_html if rows_html else '<tr><td colspan="9">No students currently flagged.</td></tr>'}
        </tbody>
    </table>

    <div class="footer">
        Auto-generated by the AI-Based Student Attendance Risk Prediction System.
        "Trend" = change in attendance % between first and second half of the semester
        (negative = declining). Risk Score is the model's predicted probability that
        a student's overall attendance will fall below {threshold:.0f}%.
    </div>
</div>
</body>
</html>"""
    return html


def main():
    bundle = load_model()
    conn = sqlite3.connect(DB_PATH)

    scored_df, model_name = score_students(conn, bundle)
    write_predictions_to_db(conn, scored_df)

    html = render_html(scored_df, model_name, bundle["threshold"])
    html_path = REPORTS_DIR / "attendance_risk_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    csv_path = REPORTS_DIR / "at_risk_students.csv"
    at_risk_cols = ["roll_no", "name", "program", "year_of_study", "attendance_pct",
                     "trend", "longest_absence_streak", "risk_probability"]
    scored_df[scored_df["predicted_risk"] == 1][at_risk_cols].sort_values(
        "risk_probability", ascending=False
    ).to_csv(csv_path, index=False)

    n_risk = int((scored_df["predicted_risk"] == 1).sum())
    print(f"Scored {len(scored_df)} students using '{model_name}'.")
    print(f"  {n_risk} students flagged at-risk (threshold {bundle['threshold']}%).")
    print(f"HTML report: {html_path}")
    print(f"CSV export : {csv_path}")

    conn.close()


if __name__ == "__main__":
    main()
