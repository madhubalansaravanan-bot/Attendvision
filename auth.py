import config
import database as db


def login(identifier: str, password: str):
    identifier = identifier.strip()

    # ================= ADMIN LOGIN =================
    if (
        identifier == config.ADMIN_USERNAME
        and password == config.ADMIN_PASSWORD
    ):
        return {
            "id": None,
            "name": "Administrator",
            "email": config.ADMIN_USERNAME,
            "role": "admin",
            "department": None,
        }

    # ================= FACULTY LOGIN =================
    faculty = db.get_faculty_by_email(identifier)

    if faculty and db.verify_password(
        password,
        faculty["password_hash"]
    ):
        return {
            "id": faculty["id"],
            "name": faculty["name"],
            "email": faculty["email"],
            "role": faculty["role"],
            "department": faculty["department"],
        }

    # ================= STUDENT LOGIN =================
    # Username = Roll Number
    # Password = student123

    if password == "student123":

        conn = db.get_connection()

        student = conn.execute(
            """
            SELECT
                id,
                roll_number,
                name,
                department,
                year,
                section,
                email
            FROM students
            WHERE roll_number = ?
            """,
            (identifier,)
        ).fetchone()

        conn.close()

        if student:
            return {
                "id": student["id"],
                "name": student["name"],
                "roll_number": student["roll_number"],
                "email": student["email"],
                "role": "student",
                "department": student["department"],
                "year": student["year"],
                "section": student["section"],
            }

    return None