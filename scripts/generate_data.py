"""
generate_data.py
-----------------
Builds attendance.db from schema.sql and populates it with realistic
synthetic data: students, courses, enrollments and day-by-day attendance
records across one semester. Attendance patterns are generated with
several underlying "student behavior profiles" (regular, average,
at-risk, improving, declining) so the resulting dataset has genuine
signal for the ML models to learn from.

Run:  python generate_data.py
"""

import sqlite3
import random
from datetime import date, timedelta
from pathlib import Path

RNG_SEED = 42
random.seed(RNG_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "attendance.db"
SCHEMA_PATH = BASE_DIR / "db" / "schema.sql"

SEMESTER = "Fall-2026"
NUM_STUDENTS = 220
PROGRAMS = ["B.Tech CSE", "B.Tech ECE", "B.Tech ME", "BCA", "MCA"]

COURSES = [
    ("CS101", "Database Management Systems", 4, "Dr. Sharma"),
    ("CS102", "Data Structures & Algorithms", 4, "Dr. Rao"),
    ("CS103", "Machine Learning", 3, "Dr. Iyer"),
    ("CS104", "Computer Networks", 3, "Dr. Nair"),
    ("CS105", "Operating Systems", 4, "Dr. Kulkarni"),
]

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
               "Ayaan", "Krishna", "Ishaan", "Ananya", "Diya", "Saanvi", "Aadhya",
               "Kiara", "Myra", "Anika", "Riya", "Pari", "Ira", "Rohan", "Kabir",
               "Advait", "Dhruv", "Shaurya", "Meera", "Isha", "Navya", "Tara", "Zoya"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Patel", "Reddy", "Iyer", "Nair",
              "Das", "Roy", "Bose", "Mishra", "Chatterjee", "Rao", "Kulkarni",
              "Menon", "Pillai", "Agarwal", "Bhatt", "Joshi"]

# Semester runs ~16 weeks, classes Mon/Wed/Fri per course
SEMESTER_START = date(2026, 8, 3)   # a Monday
NUM_WEEKS = 16
CLASS_WEEKDAYS = [0, 2, 4]  # Mon, Wed, Fri

# Behavior profiles: (weight, base_attendance_prob, trend_per_week)
# trend_per_week shifts probability across the semester (declining/improving)
PROFILES = [
    ("regular",    0.35,  0.94,  0.000),
    ("average",    0.30,  0.82, -0.002),
    ("at_risk",    0.15,  0.55, -0.010),
    ("improving",  0.10,  0.60,  0.012),
    ("declining",  0.10,  0.85, -0.018),
]


def build_schema(conn: sqlite3.Connection):
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()


def pick_profile():
    r = random.random()
    cum = 0.0
    for name, weight, base_p, trend in PROFILES:
        cum += weight
        if r <= cum:
            return name, base_p, trend
    return PROFILES[-1][0], PROFILES[-1][2], PROFILES[-1][3]


def class_dates():
    dates = []
    d = SEMESTER_START
    end = SEMESTER_START + timedelta(weeks=NUM_WEEKS)
    while d < end:
        if d.weekday() in CLASS_WEEKDAYS:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    build_schema(conn)
    cur = conn.cursor()

    # --- Courses ---
    course_ids = {}
    for code, name, credits, instructor in COURSES:
        cur.execute(
            "INSERT INTO courses (course_code, course_name, credits, instructor) VALUES (?,?,?,?)",
            (code, name, credits, instructor),
        )
        course_ids[code] = cur.lastrowid

    # --- Students ---
    used_names = set()
    student_ids = []
    student_profiles = {}
    for i in range(1, NUM_STUDENTS + 1):
        while True:
            fn = random.choice(FIRST_NAMES)
            ln = random.choice(LAST_NAMES)
            full = f"{fn} {ln}"
            key = (full, i)
            if key not in used_names:
                used_names.add(key)
                break
        roll_no = f"2026{i:04d}"
        gender = random.choice(["M", "F"])
        program = random.choice(PROGRAMS)
        year = random.choice([1, 2, 3, 4])
        email = f"{fn.lower()}.{ln.lower()}{i}@college.edu"

        cur.execute(
            "INSERT INTO students (roll_no, name, gender, program, year_of_study, email) "
            "VALUES (?,?,?,?,?,?)",
            (roll_no, full, gender, program, year, email),
        )
        sid = cur.lastrowid
        student_ids.append(sid)
        student_profiles[sid] = pick_profile()

    # --- Enrollments: each student takes 3-5 random courses ---
    enrollment_ids = {}  # (student_id, course_id) -> enrollment_id
    for sid in student_ids:
        n_courses = random.randint(3, 5)
        chosen = random.sample(list(course_ids.values()), n_courses)
        for cid in chosen:
            cur.execute(
                "INSERT INTO enrollments (student_id, course_id, semester) VALUES (?,?,?)",
                (sid, cid, SEMESTER),
            )
            enrollment_ids[(sid, cid)] = cur.lastrowid

    conn.commit()

    # --- Attendance: generate per enrollment across the semester ---
    dates = class_dates()
    attendance_rows = []
    for (sid, cid), eid in enrollment_ids.items():
        profile_name, base_p, trend = student_profiles[sid]
        for week_idx, d in enumerate(dates):
            week_number = week_idx // len(CLASS_WEEKDAYS)
            prob_present = base_p + trend * week_number
            prob_present = min(max(prob_present, 0.05), 0.99)

            r = random.random()
            if r < prob_present:
                status = "Present"
            elif r < prob_present + 0.03:
                status = "Late"
            elif r < prob_present + 0.05:
                status = "Excused"
            else:
                status = "Absent"

            attendance_rows.append((eid, d.isoformat(), status))

    cur.executemany(
        "INSERT INTO attendance (enrollment_id, class_date, status) VALUES (?,?,?)",
        attendance_rows,
    )
    conn.commit()

    # --- Summary printout ---
    n_students = cur.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    n_courses = cur.execute("SELECT COUNT(*) FROM courses").fetchone()[0]
    n_enroll = cur.execute("SELECT COUNT(*) FROM enrollments").fetchone()[0]
    n_att = cur.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]

    print(f"Database created at: {DB_PATH}")
    print(f"  Students:     {n_students}")
    print(f"  Courses:      {n_courses}")
    print(f"  Enrollments:  {n_enroll}")
    print(f"  Attendance rows: {n_att}")

    conn.close()


if __name__ == "__main__":
    main()
