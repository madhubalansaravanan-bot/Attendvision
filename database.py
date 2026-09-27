"""
AttendVision - database layer.

Uses raw sqlite3 (no ORM) so the schema in the spec maps 1:1 onto tables.
All functions open a short-lived connection per call, which is fine at
prototype scale and keeps Streamlit's rerun model simple.
"""
import sqlite3
import hashlib
import secrets
from datetime import datetime, date as date_cls

import config


def get_connection():
    conn = sqlite3.connect(config.DATABASE_URL)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def hash_password(password: str, salt: str = None) -> str:
    """PBKDF2 password hash, stored as 'salt$hash'. Good enough for a
    local prototype; swap for a proper auth provider in production."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, _ = stored_hash.split("$")
    except ValueError:
        return False
    return secrets.compare_digest(hash_password(password, salt), stored_hash)


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT,
            year TEXT,
            section TEXT,
            email TEXT,
            face_enrolled INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS faculty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            department TEXT,
            role TEXT NOT NULL DEFAULT 'faculty',
            created_at TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_code TEXT UNIQUE NOT NULL,
            subject_name TEXT NOT NULL,
            department TEXT,
            year TEXT,
            semester TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER NOT NULL REFERENCES subjects(id),
            faculty_id INTEGER NOT NULL REFERENCES faculty(id),
            date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            section TEXT,
            start_time TEXT,
            end_time TEXT,
            status TEXT NOT NULL DEFAULT 'open'
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES attendance_sessions(id),
            student_id INTEGER NOT NULL REFERENCES students(id),
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'present',
            confidence REAL,
            UNIQUE(session_id, student_id)
        )
    """)

    conn.commit()
    conn.close()


def seed_demo_data():
    """Insert a demo admin/faculty account, a couple of subjects and
    students, ONLY if the database is empty. Safe to call every startup."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM faculty")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT INTO faculty (name, email, password_hash, department, role, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("Demo Faculty", "faculty@example.edu", hash_password("faculty123"),
             "Computer Science", "faculty", datetime.now().isoformat()),
        )

    c.execute("SELECT COUNT(*) FROM subjects")
    if c.fetchone()[0] == 0:
        demo_subjects = [
            ("CS201", "Engineering Mathematics", "Computer Science", "2", "3"),
            ("CS205", "Data Structures", "Computer Science", "2", "3"),
        ]
        c.executemany(
            "INSERT INTO subjects (subject_code, subject_name, department, year, semester) "
            "VALUES (?, ?, ?, ?, ?)", demo_subjects,
        )

    c.execute("SELECT COUNT(*) FROM students")
    if c.fetchone()[0] == 0:
        demo_students = [
            ("CS001", "Aditi Rao", "Computer Science", "2", "A", "aditi@example.edu"),
            ("CS002", "Rohan Mehta", "Computer Science", "2", "A", "rohan@example.edu"),
            ("CS003", "Sara Iqbal", "Computer Science", "2", "A", "sara@example.edu"),
        ]
        for roll, name, dept, year, section, email in demo_students:
            c.execute(
                "INSERT INTO students (roll_number, name, department, year, section, email, "
                "face_enrolled, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                (roll, name, dept, year, section, email, datetime.now().isoformat()),
            )

    conn.commit()
    conn.close()


# --- Students --------------------------------------------------------------

def add_student(roll_number, name, department, year, section, email):
    conn = get_connection()
    conn.execute(
        "INSERT INTO students (roll_number, name, department, year, section, email, "
        "face_enrolled, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        (roll_number, name, department, year, section, email, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def update_student(student_id, name, department, year, section, email):
    conn = get_connection()
    conn.execute(
        "UPDATE students SET name=?, department=?, year=?, section=?, email=? WHERE id=?",
        (name, department, year, section, email, student_id),
    )
    conn.commit()
    conn.close()


def set_face_enrolled(student_id, enrolled=True):
    conn = get_connection()
    conn.execute("UPDATE students SET face_enrolled=? WHERE id=?", (int(enrolled), student_id))
    conn.commit()
    conn.close()


def delete_student(student_id):
    conn = get_connection()
    conn.execute("DELETE FROM students WHERE id=?", (student_id,))
    conn.commit()
    conn.close()


def get_students(department=None, section=None):
    conn = get_connection()

    query = "SELECT * FROM students WHERE 1=1"
    params = []

    if department:
        query += " AND LOWER(TRIM(department)) = LOWER(TRIM(?))"
        params.append(department)

    if section:
        query += " AND LOWER(TRIM(section)) = LOWER(TRIM(?))"
        params.append(section)

    query += " ORDER BY roll_number"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(r) for r in rows]

def get_student_by_roll(roll_number):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM students WHERE roll_number=?",
        (roll_number,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_student(student_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# --- Faculty -----------------------------------------------------------

def get_faculty_by_email(email):
    conn = get_connection()
    row = conn.execute("SELECT * FROM faculty WHERE email=?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def add_faculty(name, email, password, department, role="faculty"):
    conn = get_connection()
    conn.execute(
        "INSERT INTO faculty (name, email, password_hash, department, role, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, email, hash_password(password), department, role, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


# --- Subjects ----------------------------------------------------------

def add_subject(subject_code, subject_name, department, year, semester):
    conn = get_connection()
    conn.execute(
        "INSERT INTO subjects (subject_code, subject_name, department, year, semester) "
        "VALUES (?, ?, ?, ?, ?)",
        (subject_code, subject_name, department, year, semester),
    )
    conn.commit()
    conn.close()


def update_subject(subject_id, subject_name, department, year, semester):
    conn = get_connection()
    conn.execute(
        "UPDATE subjects SET subject_name=?, department=?, year=?, semester=? WHERE id=?",
        (subject_name, department, year, semester, subject_id),
    )
    conn.commit()
    conn.close()


def delete_subject(subject_id):
    conn = get_connection()
    conn.execute("DELETE FROM subjects WHERE id=?", (subject_id,))
    conn.commit()
    conn.close()


def get_subjects():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM subjects ORDER BY subject_code").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Attendance sessions -------------------------------------------------

def create_session(subject_id, faculty_id, session_date, hour, section):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO attendance_sessions (subject_id, faculty_id, date, hour, section, "
        "start_time, status) VALUES (?, ?, ?, ?, ?, ?, 'open')",
        (subject_id, faculty_id, session_date, hour, section, datetime.now().isoformat()),
    )
    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    return session_id


def find_open_session(subject_id, session_date, hour, section):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM attendance_sessions WHERE subject_id=? AND date=? AND hour=? "
        "AND section=? AND status='open'",
        (subject_id, session_date, hour, section),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def end_session(session_id):
    conn = get_connection()
    conn.execute(
        "UPDATE attendance_sessions SET status='closed', end_time=? WHERE id=?",
        (datetime.now().isoformat(), session_id),
    )
    conn.commit()
    conn.close()


def get_session(session_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM attendance_sessions WHERE id=?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# --- Attendance records --------------------------------------------------

def mark_attendance(session_id, student_id, confidence, status="present"):
    """Marks a student present for a session. Returns True if a new record
    was inserted, False if the student was already marked (duplicate)."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO attendance_records (session_id, student_id, timestamp, status, "
            "confidence) VALUES (?, ?, ?, ?, ?)",
            (session_id, student_id, datetime.now().isoformat(), status, confidence),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_session_records(session_id):
    conn = get_connection()
    rows = conn.execute("""
        SELECT ar.*, s.name AS student_name, s.roll_number
        FROM attendance_records ar
        JOIN students s ON s.id = ar.student_id
        WHERE ar.session_id=?
        ORDER BY ar.timestamp
    """, (session_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_attendance_history(date_filter=None, subject_id=None, hour=None,
                            student_id=None, section=None):
    conn = get_connection()
    query = """
        SELECT ar.id, ar.timestamp, ar.status, ar.confidence,
               s.roll_number, s.name AS student_name,
               sub.subject_name, sub.subject_code,
               sess.date, sess.hour, sess.section
        FROM attendance_records ar
        JOIN students s ON s.id = ar.student_id
        JOIN attendance_sessions sess ON sess.id = ar.session_id
        JOIN subjects sub ON sub.id = sess.subject_id
        WHERE 1=1
    """
    params = []
    if date_filter:
        query += " AND sess.date = ?"
        params.append(date_filter)
    if subject_id:
        query += " AND sub.id = ?"
        params.append(subject_id)
    if hour:
        query += " AND sess.hour = ?"
        params.append(hour)
    if student_id:
        query += " AND s.id = ?"
        params.append(student_id)
    if section:
        query += " AND sess.section = ?"
        params.append(section)
    query += " ORDER BY sess.date DESC, sess.hour DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dashboard_stats(today: str = None):
    today = today or date_cls.today().isoformat()
    conn = get_connection()
    total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    todays_classes = conn.execute(
        "SELECT COUNT(*) FROM attendance_sessions WHERE date=?", (today,)
    ).fetchone()[0]
    present_today = conn.execute("""
        SELECT COUNT(DISTINCT ar.student_id) FROM attendance_records ar
        JOIN attendance_sessions sess ON sess.id = ar.session_id
        WHERE sess.date=? AND ar.status='present'
    """, (today,)).fetchone()[0]
    conn.close()
    absent_today = max(total_students - present_today, 0)
    pct = round((present_today / total_students) * 100, 1) if total_students else 0.0
    return {
        "todays_classes": todays_classes,
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "attendance_percentage": pct,
    }


def get_student_attendance_percentage(student_id):
    conn = get_connection()
    total_sessions = conn.execute("""
        SELECT COUNT(*) FROM attendance_sessions sess
        JOIN students s ON s.section = sess.section OR sess.section IS NULL
        WHERE s.id = ?
    """, (student_id,)).fetchone()[0]
    present = conn.execute(
        "SELECT COUNT(*) FROM attendance_records WHERE student_id=? AND status='present'",
        (student_id,),
    ).fetchone()[0]
    conn.close()
    if total_sessions == 0:
        return 0.0
    return round((present / total_sessions) * 100, 1)
