# AI-Based Student Attendance Risk Prediction System

A DBMS + AI mini-project that predicts which students are likely to fall
below an attendance threshold, using historical attendance data stored in
a relational database.

| | |
|---|---|
| **Algorithms** | Logistic Regression / Random Forest |
| **Data Source** | Attendance DB (SQLite) |
| **Output** | Automated risk reports (HTML dashboard + CSV) |
| **Goal** | Predicts students likely to fall below attendance threshold |

---

## 1. Project Structure

```
attendance_risk_project/
├── db/
│   ├── schema.sql              -- table definitions (DDL)
│   └── attendance.db           -- SQLite database (generated)
├── scripts/
│   ├── generate_data.py        -- builds DB + synthetic attendance data
│   ├── train_model.py          -- feature engineering + model training
│   └── generate_report.py      -- scores students, writes report
├── models/
│   ├── risk_model.pkl          -- trained model + scaler (generated)
│   ├── student_features.csv    -- engineered feature table (generated)
│   └── model_comparison.csv    -- LogReg vs RandomForest metrics (generated)
├── reports/
│   ├── attendance_risk_report.html   -- styled dashboard (generated)
│   └── at_risk_students.csv          -- flagged students export (generated)
├── requirements.txt
└── README.md
```

## 2. Database Design

**students** — student_id, roll_no, name, gender, program, year_of_study, email
**courses** — course_id, course_code, course_name, credits, instructor
**enrollments** — links a student to a course for a given semester
**attendance** — one row per class session per enrolled student (Present / Absent / Late / Excused)
**risk_predictions** — model output: predicted_risk, risk_probability, model_used, refreshed each report run

This is a standard normalized schema: students and courses are independent
entities, enrollments is the many-to-many join table between them, and
attendance hangs off enrollments (so it's automatically scoped to the right
student + course + semester). `risk_predictions` is the AI layer's output
table, kept separate from raw attendance so re-scoring never touches source data.

## 3. Machine Learning Approach

**Label:** a student is "at risk" if their overall attendance % falls below
the threshold (default 75%).

**Features engineered per student** (deliberately *not* including the raw
overall attendance % itself, to avoid trivially leaking the label):
- `first_half_pct` / `second_half_pct` — attendance in each half of the semester
- `trend` — change between the two halves (negative = declining)
- `longest_absence_streak` — longest run of consecutive missed classes
- `recent_attendance_pct` — attendance over the last ~12 sessions
- `num_courses`, `year_of_study`, `late_pct`, `excused_pct`

**Models trained:** Logistic Regression (with feature scaling) and Random
Forest, both with `class_weight="balanced"` to handle the at-risk class
being a minority. Both are evaluated on a held-out test split (accuracy,
precision, recall, F1, ROC-AUC); the model with the higher F1 score is
saved and used for scoring.

On the generated sample data both models score **~0.90–0.94 F1**, and the
Random Forest's feature importances confirm `first_half_pct`,
`second_half_pct` and `longest_absence_streak` are the strongest predictors
— i.e. the model is genuinely picking up on attendance *behavior*, not just
memorizing the label.

## 4. How to Run

```bash
cd attendance_risk_project
pip install -r requirements.txt

# 1. Build the database and populate it with sample data
python scripts/generate_data.py

# 2. Train the models and save the best one
python scripts/train_model.py

# 3. Score all students and generate the report
python scripts/generate_report.py
```

Open `reports/attendance_risk_report.html` in a browser to view the dashboard.

## 5. Using Your Own Data

Replace the synthetic data step: instead of running `generate_data.py`,
load `db/schema.sql` into SQLite and `INSERT` your real students, courses,
enrollments and attendance records (e.g. from a CSV export of your
college's attendance system). Then just run `train_model.py` followed by
`generate_report.py` as above — the pipeline works on any data that fits
the schema.

## 6. Extending the Project

- **Automated alerts**: add a script that emails/SMS's students in the
  "High" risk band using the `email` column already in `students`.
- **Scheduling**: run `generate_report.py` weekly via cron so the report
  and `risk_predictions` table stay current.
- **Web front-end**: build a small Flask/FastAPI app over the same SQLite
  DB to let faculty query risk by course or program interactively.
- **More features**: add day-of-week patterns, per-course (not just
  overall) risk scores, or assignment/exam performance if available.
