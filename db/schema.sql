-- =====================================================================
-- AI-Based Student Attendance Risk Prediction System
-- Database Schema (SQLite)
-- =====================================================================

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS attendance;
DROP TABLE IF EXISTS enrollments;
DROP TABLE IF EXISTS courses;
DROP TABLE IF EXISTS students;
DROP TABLE IF EXISTS risk_predictions;

-- ---------------------------------------------------------------------
-- STUDENTS
-- ---------------------------------------------------------------------
CREATE TABLE students (
    student_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no         TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL,
    gender          TEXT CHECK (gender IN ('M', 'F', 'O')),
    program         TEXT NOT NULL,
    year_of_study   INTEGER NOT NULL,
    email           TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------
-- COURSES
-- ---------------------------------------------------------------------
CREATE TABLE courses (
    course_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code     TEXT UNIQUE NOT NULL,
    course_name     TEXT NOT NULL,
    credits         INTEGER NOT NULL,
    instructor      TEXT
);

-- ---------------------------------------------------------------------
-- ENROLLMENTS  (student <-> course, per semester)
-- ---------------------------------------------------------------------
CREATE TABLE enrollments (
    enrollment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    course_id       INTEGER NOT NULL REFERENCES courses(course_id) ON DELETE CASCADE,
    semester        TEXT NOT NULL,          -- e.g. 'Fall-2026'
    UNIQUE (student_id, course_id, semester)
);

-- ---------------------------------------------------------------------
-- ATTENDANCE  (one row per class session per enrolled student)
-- ---------------------------------------------------------------------
CREATE TABLE attendance (
    attendance_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    enrollment_id   INTEGER NOT NULL REFERENCES enrollments(enrollment_id) ON DELETE CASCADE,
    class_date      TEXT NOT NULL,           -- ISO date 'YYYY-MM-DD'
    status          TEXT NOT NULL CHECK (status IN ('Present', 'Absent', 'Late', 'Excused'))
);

-- ---------------------------------------------------------------------
-- RISK_PREDICTIONS  (output of the AI model, refreshed by report job)
-- ---------------------------------------------------------------------
CREATE TABLE risk_predictions (
    prediction_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id        INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    semester          TEXT NOT NULL,
    attendance_pct    REAL NOT NULL,
    predicted_risk    INTEGER NOT NULL,      -- 1 = at risk, 0 = safe
    risk_probability  REAL NOT NULL,         -- model confidence (0-1)
    model_used        TEXT NOT NULL,
    generated_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Helpful indexes
CREATE INDEX idx_attendance_enrollment ON attendance(enrollment_id);
CREATE INDEX idx_enrollment_student   ON enrollments(student_id);
CREATE INDEX idx_enrollment_course    ON enrollments(course_id);
CREATE INDEX idx_risk_student         ON risk_predictions(student_id);
