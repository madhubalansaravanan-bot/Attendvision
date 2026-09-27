# AttendVision

Smart, hour-wise attendance using computer vision. A local Streamlit + OpenCV
+ SQLite prototype: a faculty member picks a subject and hour, starts the
webcam, and enrolled students get marked present automatically as they're
recognized.

## How recognition works

- **Detection:** OpenCV's built-in Haar cascade (`haarcascade_frontalface_default.xml`).
- **Recognition:** OpenCV's LBPH (Local Binary Patterns Histograms) face
  recognizer, from `opencv-contrib-python`. It's trained from scratch on
  every enrolled student's saved face samples whenever someone new is
  enrolled.
- A match is only accepted below a confidence-distance threshold
  (`config.RECOGNITION_CONFIDENCE_THRESHOLD`) **and** only after it's
  been seen for several consecutive frames
  (`config.CONSECUTIVE_MATCH_FRAMES_REQUIRED`). Anything less confident
  is shown as **Unknown** and is never auto-marked.
- **This is not claimed to be 100% accurate.** Lighting, camera angle,
  and image quality all affect it — treat it as an assistive tool, with a
  faculty member able to review and correct the record afterward (via
  Attendance History).

## Setup

```bash
cd attendvision
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Optional — override the demo defaults (see `.env.example`):

```bash
export ADMIN_USERNAME=admin
export ADMIN_PASSWORD=your-own-password
export SECRET_KEY=some-random-string
```

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Demo login

| Role    | Username / Email          | Password      |
|---------|----------------------------|---------------|
| Admin   | `admin` (or your env var)  | `admin123` (or your env var) |
| Faculty | `faculty@example.edu`      | `faculty123`  |

Three demo students (unenrolled) and two demo subjects are seeded
automatically on first run.

## Using it

1. **Log in** as admin or faculty.
2. **Students → Add / Edit / Delete**: add students (roll number, name,
   department, year, section, email).
3. **Students → Enroll Face**: select a student, click **Start Capture**.
   Have the student face the camera directly, alone in frame, for a few
   seconds while ~25 samples are captured; the recognizer retrains
   automatically afterward.
4. **Subjects**: add at least one subject if you want your own instead of
   the demo ones.
5. **Take Attendance**: pick department/section/subject/date/hour, click
   **START CAMERA**. Recognized, enrolled students are marked present
   live; unrecognized faces show as `Unknown` and are not marked. Click
   **End Session** when done.
6. **Attendance History** / **Reports**: filter past sessions and export
   CSV.

## Project structure

```
attendvision/
├── app.py              # entry point: page config, login gate, router
├── auth.py              # login (admin env-var account + faculty table)
├── config.py             # paths, env vars, recognition thresholds
├── database.py           # SQLite schema + all queries
├── views/                # one module per screen (not named "pages/" —
│   ├── dashboard.py       #  see note below)
│   ├── attendance.py      # the live webcam attendance session
│   ├── students.py        # CRUD + face enrollment
│   ├── subjects.py
│   ├── history.py
│   ├── reports.py
│   └── settings.py
├── vision/
│   ├── camera.py          # cv2.VideoCapture wrapper with error handling
│   ├── detector.py        # Haar cascade face detection
│   ├── recognizer.py      # LBPH training/prediction
│   └── enrollment.py      # save face samples to disk, retrain
├── data/
│   ├── students/<roll_number>/*.jpg   # saved face samples
│   ├── models/                         # trained LBPH model + label map
│   └── attendance/                     # (reserved for exported reports)
└── requirements.txt
```

**Note on `views/` vs. the spec's `pages/`:** Streamlit auto-detects any
folder literally named `pages/` and turns every file inside into its own
top-level, always-visible page in the sidebar — which would bypass the
login gate. The folder was renamed to `views/` and routing is instead
done manually in `app.py` after checking `st.session_state["user"]`, so
the required flow (Login → Dashboard → …) is enforced.

## Data model

Matches the spec: `students`, `faculty`, `subjects`, `attendance_sessions`,
`attendance_records`, with a `UNIQUE(session_id, student_id)` constraint on
`attendance_records` so a student can't be marked twice in the same
session — enforced at the database level, not just in the UI.

One deliberate simplification vs. the spec's `students.face_data` column:
face images are stored as files under `data/students/<roll_number>/`
rather than as a database blob, with a `face_enrolled` flag on the
`students` row. This keeps the database small and makes retraining (just
re-reading files) straightforward.

## Privacy & consent

This system processes biometric data (face images). Before enrolling any
student:

- Obtain the student's consent and the college's authorization.
- Store only what's necessary, and only for as long as the attendance
  system needs it.
- Treat `data/students/` and the SQLite database as sensitive data —
  don't commit them to version control or share them outside the
  authorized system.

## Known limitations (prototype, not production)

- Single local webcam, single machine — no multi-camera or cloud sync.
- LBPH is CPU-light but less accurate than deep-learning face embeddings;
  fine for a small, controlled classroom roster, not for large-scale or
  adversarial use.
- Recognition quality depends heavily on enrollment lighting/angle
  variety — enroll students under conditions similar to where attendance
  will be taken.
- No password reset flow; an admin adds faculty accounts directly.
- `attendance_sessions.section` is used as a simple text match for
  "which students belong to this class" — there's no separate
  class/section table.

## Testing checklist (Phase 9 from the spec)

- [ ] Correct student is recognized and marked present
- [ ] Unrecognized face shows "Unknown" and isn't marked
- [ ] Multiple students in frame are each tracked independently
- [ ] Re-showing an already-marked student's face doesn't create a duplicate record
- [ ] Unplugging/disabling the camera mid-session shows an error and ends the session safely
- [ ] Wrong login credentials are rejected
- [ ] Attendance for different hours/subjects on the same day stay separate
- [ ] CSV export opens correctly and matches the on-screen filtered table
