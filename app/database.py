import os
import sqlite3
from datetime import datetime
from contextlib import contextmanager


# Resolve the database file relative to the project root (one level above /app).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'edubot.db')


@contextmanager
def get_connection():
    """Yield a sqlite3 connection with row-factory set to dict-like access.

    Using a context manager guarantees the connection closes even if the
    caller raises - important when this runs inside a Flask request.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema():
    """Create all tables if they don't exist. Idempotent - safe to call repeatedly."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with get_connection() as conn:
        cur = conn.cursor()

        cur.executescript("""
        CREATE TABLE IF NOT EXISTS courses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            code            TEXT NOT NULL UNIQUE,
            name            TEXT NOT NULL,
            level           TEXT NOT NULL,
            faculty         TEXT NOT NULL,
            duration_years  REAL NOT NULL,
            fee_per_year    INTEGER NOT NULL,
            description     TEXT,
            updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS faculty (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            title           TEXT NOT NULL,
            name            TEXT NOT NULL,
            department      TEXT NOT NULL,
            expertise       TEXT,
            email           TEXT,
            office_hours    TEXT,
            is_dean         INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            start_date      TEXT NOT NULL,
            end_date        TEXT,
            location        TEXT,
            category        TEXT,
            description     TEXT
        );

        CREATE TABLE IF NOT EXISTS exams (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_type       TEXT NOT NULL,
            start_date      TEXT NOT NULL,
            end_date        TEXT NOT NULL,
            format          TEXT,
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS scholarships (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL UNIQUE,
            max_percentage  INTEGER NOT NULL,
            eligibility     TEXT,
            description     TEXT
        );

        CREATE TABLE IF NOT EXISTS hostel_rooms (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            room_type           TEXT NOT NULL UNIQUE,
            capacity            INTEGER NOT NULL,
            price_per_semester  INTEGER NOT NULL,
            amenities           TEXT
        );

        -- Generic key/value store for facts that don't deserve their own
        -- table (library hours, contact info, timetable rules, etc.).
        CREATE TABLE IF NOT EXISTS kv_facts (
            key      TEXT PRIMARY KEY,
            value    TEXT NOT NULL,
            category TEXT
        );

        -- Feedback drives the ML loop: 0 = thumbs down, 1 = thumbs up.
        CREATE TABLE IF NOT EXISTS feedback (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message      TEXT NOT NULL,
            bot_response      TEXT NOT NULL,
            predicted_intent  TEXT,
            confidence        REAL,
            helpful           INTEGER NOT NULL,
            expected_intent   TEXT,
            created_at        TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Patterns the bot has been taught since the last training run.
        -- Two-tier trust model:
        --   approved = 1: vetted, will be merged into the next retrain
        --   approved = 0: pending admin review (typically from end-user
        --                 thumbs-down feedback)
        -- used_in_model flips to 1 once a retrain has actually consumed
        -- the row.
        CREATE TABLE IF NOT EXISTS learned_patterns (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern       TEXT NOT NULL,
            intent        TEXT NOT NULL,
            source        TEXT DEFAULT 'user_taught',
            approved      INTEGER DEFAULT 0,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            used_in_model INTEGER DEFAULT 0
        );

        -- Full conversation log (handy for the test plan + future analytics).
        CREATE TABLE IF NOT EXISTS chat_history (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message      TEXT NOT NULL,
            bot_response      TEXT NOT NULL,
            intent            TEXT,
            confidence        REAL,
            response_source   TEXT,
            created_at        TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """)

        # Idempotent migration: older databases (created before the
        # trust-tier split was introduced) won't have the `approved`
        # column. Add it on the fly and grandfather every existing row
        # in as approved so the model's behaviour doesn't change after
        # an upgrade.
        cols = [r['name'] for r in cur.execute(
            "PRAGMA table_info(learned_patterns)"
        )]
        if 'approved' not in cols:
            cur.execute(
                "ALTER TABLE learned_patterns "
                "ADD COLUMN approved INTEGER DEFAULT 0"
            )
            cur.execute("UPDATE learned_patterns SET approved = 1")


# -------- Read helpers (used by the inference engine) --------

def list_courses():
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM courses ORDER BY level, name"
        )]


def find_course(keyword):
    """Look up courses by free-text match in name/code/faculty."""
    pattern = f"%{keyword.lower()}%"
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT * FROM courses
               WHERE LOWER(name) LIKE ?
                  OR LOWER(code) LIKE ?
                  OR LOWER(faculty) LIKE ?""",
            (pattern, pattern, pattern)
        )]


def list_faculty():
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM faculty ORDER BY is_dean DESC, name"
        )]


def get_dean():
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM faculty WHERE is_dean = 1 LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def list_events():
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM events ORDER BY start_date"
        )]


def list_exams():
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM exams ORDER BY start_date"
        )]


def list_scholarships():
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM scholarships ORDER BY max_percentage DESC"
        )]


def list_hostel_rooms():
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM hostel_rooms ORDER BY price_per_semester"
        )]


def get_fact(key):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM kv_facts WHERE key = ?", (key,)
        ).fetchone()
        return row['value'] if row else None


def get_facts_by_category(category):
    with get_connection() as conn:
        return {r['key']: r['value'] for r in conn.execute(
            "SELECT key, value FROM kv_facts WHERE category = ?", (category,)
        )}


# -------- Write helpers (used by the learning loop) --------

def log_chat(user_message, bot_response, intent, confidence, source):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO chat_history
               (user_message, bot_response, intent, confidence, response_source)
               VALUES (?, ?, ?, ?, ?)""",
            (user_message, bot_response, intent, confidence, source)
        )


def log_feedback(user_message, bot_response, predicted_intent,
                 confidence, helpful, expected_intent=None):
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO feedback
               (user_message, bot_response, predicted_intent,
                confidence, helpful, expected_intent)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_message, bot_response, predicted_intent,
             confidence, int(bool(helpful)), expected_intent)
        )
        return cur.lastrowid


def add_learned_pattern(pattern, intent, source='user_taught', approved=False):
    """Persist a new training example.

    Two trust tiers:
      - approved=False: stays in 'pending review' until an admin OKs it
      - approved=True:  ready to be picked up on the next train run

    train.py only consumes approved rows.
    """
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO learned_patterns
               (pattern, intent, source, approved)
               VALUES (?, ?, ?, ?)""",
            (pattern, intent, source, 1 if approved else 0)
        )
        return cur.lastrowid


def get_learned_patterns(approved_only=False, only_unused=False):
    """Return rows from learned_patterns.

    approved_only - True when called from train.py so unvetted suggestions
                    don't leak into the model
    only_unused   - True when filtering for patterns that haven't yet been
                    baked into the model
    """
    where = []
    if approved_only:
        where.append("approved = 1")
    if only_unused:
        where.append("used_in_model = 0")
    sql = "SELECT * FROM learned_patterns"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at"
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(sql)]


def get_pending_patterns():
    """Patterns awaiting admin review (not yet approved)."""
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM learned_patterns "
            "WHERE approved = 0 ORDER BY created_at DESC"
        )]


def approve_pattern(pattern_id):
    """Admin says yes - flip approved to 1 so train.py will consume it."""
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE learned_patterns SET approved = 1 WHERE id = ?",
            (pattern_id,)
        )
        return cur.rowcount > 0


def discard_pattern(pattern_id):
    """Admin says no - delete the row so it never enters training."""
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM learned_patterns WHERE id = ?",
            (pattern_id,)
        )
        return cur.rowcount > 0


def mark_patterns_used():
    """Called by train.py after a successful retrain. Only flips rows
    that were actually approved + used in this run."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE learned_patterns SET used_in_model = 1 "
            "WHERE approved = 1"
        )


def count_pending_patterns():
    """Count of approved-but-not-yet-trained patterns. Drives the
    auto-retrain threshold in learning.py."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM learned_patterns "
            "WHERE approved = 1 AND used_in_model = 0"
        ).fetchone()
        return row['c']


def count_pending_review():
    """Count of patterns still waiting for an admin decision."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM learned_patterns WHERE approved = 0"
        ).fetchone()
        return row['c']


def stats():
    """Snapshot of DB content - shown on /admin and used in tests."""
    with get_connection() as conn:
        c = conn.cursor()
        return {
            'courses':            c.execute("SELECT COUNT(*) FROM courses").fetchone()[0],
            'faculty':            c.execute("SELECT COUNT(*) FROM faculty").fetchone()[0],
            'events':             c.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            'exams':              c.execute("SELECT COUNT(*) FROM exams").fetchone()[0],
            'scholarships':       c.execute("SELECT COUNT(*) FROM scholarships").fetchone()[0],
            'hostel_rooms':       c.execute("SELECT COUNT(*) FROM hostel_rooms").fetchone()[0],
            'kv_facts':           c.execute("SELECT COUNT(*) FROM kv_facts").fetchone()[0],
            'feedback':           c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0],
            'helpful_feedback':   c.execute("SELECT COUNT(*) FROM feedback WHERE helpful = 1").fetchone()[0],
            'unhelpful_feedback': c.execute("SELECT COUNT(*) FROM feedback WHERE helpful = 0").fetchone()[0],
            'learned_patterns':   c.execute("SELECT COUNT(*) FROM learned_patterns").fetchone()[0],
            'pending_review':     c.execute("SELECT COUNT(*) FROM learned_patterns WHERE approved = 0").fetchone()[0],
            'pending_patterns':   c.execute("SELECT COUNT(*) FROM learned_patterns WHERE approved = 1 AND used_in_model = 0").fetchone()[0],
            'chat_history':       c.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0],
        }


if __name__ == "__main__":
    init_schema()
    print(f"DB initialised at {DB_PATH}")
    print(f"Stats: {stats()}")
