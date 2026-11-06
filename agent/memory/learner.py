"""Step 10: the tutor's notes on each student, one SQLite file per student.

Tables:
  mastery        score per concept (0 to 1), last updated
  wrong_beliefs  a misconception, its concept, when it was seen
  facts          true claims only, each with the id of the passage that supports it
  session_notes  what the student struggled with in one past session, written by reflection
  concepts       the notebook's concept list, one row per (concept, file); not on the memory page
shortcut: session notes are not embedded yet, add with the vector memory variant (ticket 10)
"""

import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_DIR = Path("data/students")
ALPHA = 0.3 # how far the latest answer moves the score
START = 0.5 # score for a concept never seen: unknown, not weak

SCHEMA = """
CREATE TABLE IF NOT EXISTS mastery (
    concept TEXT PRIMARY KEY,
    score   REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS wrong_beliefs (
    id      INTEGER PRIMARY KEY,
    concept TEXT NOT NULL,
    belief  TEXT NOT NULL,
    seen    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    id       INTEGER PRIMARY KEY,
    concept  TEXT NOT NULL,
    fact     TEXT NOT NULL,
    passage_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_notes (
    id      INTEGER PRIMARY KEY,
    note    TEXT NOT NULL,
    written REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS concepts (
    concept TEXT NOT NULL,
    file    TEXT NOT NULL,
    PRIMARY KEY (concept, file)
);
"""
TABLE_KEYS = {"mastery": "concept", "wrong_beliefs": "id", "facts": "id", "session_notes": "id"}

# Setup comms w/ DB
#   checks student_id if exist, if not create a new table for them
#   handles the connection, commit the statement if success, roll back if fail
@contextmanager
def _db(student_id):
    """Open this student's database, commit on success, always close."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", student_id):
        raise ValueError(f"bad student_id: {student_id!r}")  # stops '../bob' reaching another file
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_DIR / f"{student_id}.sqlite")
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        with conn:          # commits, or rolls back on error
            yield conn
    finally:
        conn.close()        # `with conn` alone does not close

# Normalizes concept name by cleaning it
def norm(concept):
    """'  LoRA ' and 'lora' are the same concept."""
    return " ".join(concept.lower().split())

# Learning step
#   Gets old score or start at 0.5
#   Based on ALPHA set, will move 30% to 1 if answer correct, 30% to 0 if wrong
#   If misconception observed, will log it into wrong_beliefs table
def update_mastery(student_id, concept, correct, wrong_belief=None):
    """Move the concept's score toward 1 (correct) or 0 (wrong). Returns the new score."""
    concept = norm(concept)
    with _db(student_id) as conn:
        row = conn.execute(
            "SELECT score FROM mastery WHERE concept = ?",
            (concept, )
        ).fetchone()

        old = row["score"] if row else START
        new = old + ALPHA  * ((1.0 if correct else 0.0) - old)
        conn.execute(
            "INSERT INTO mastery VALUES (?, ?, ?) "
            "ON CONFLICT(concept) DO UPDATE SET score = excluded.score, updated = excluded.updated",
            (concept, new, time.time())
        )

        if wrong_belief:
            conn.execute(
                "INSERT INTO wrong_beliefs (concept, belief, seen) VALUES (?, ?, ?)",
                (concept, wrong_belief, time.time())
            )
        return round(new, 3)

# Stores a true claim with the id of its supporting passage.
# With no source, raises ValueError and stores nothing
def add_fact(student_id, concept, fact, passage_id):
    """Store a true claim. No supporting passage, no fact."""
    if not passage_id:
        raise ValueError("a fact needs the passage_id that supports it")
    with _db(student_id) as conn:
        conn.execute(
            "INSERT INTO facts (concept, fact, passage_id) VALUES (?, ?, ?)",
     (norm(concept), fact, passage_id)
        )

# Returns up to 3 concepts scoring under 0.6 w/ weakest 1st.
# will be placed in system prompt so agent knows what topics to target
def weak_concepts(student_id, limit=3, below=0.6):
    """Lowest scores first, for the system prompt."""
    with _db(student_id) as conn:
        rows = conn.execute(
            "SELECT concept, score FROM mastery WHERE score < ? ORDER BY score LIMIT ?",
            (below, limit)
        ).fetchall()
    return [dict(r) for r in rows]

def add_session_note(student_id, note):
    """Store reflection's summary of one session."""
    with _db(student_id) as conn:
        conn.execute("INSERT INTO session_notes (note, written) VALUES (?, ?)", (note, time.time()))


# shortcut: limit hardcoded, move to config.yaml once the team agrees on it
def wrong_beliefs(student_id, limit=5):
    """The latest wrong beliefs, newest first, for the system prompt."""
    with _db(student_id) as conn:
        rows = conn.execute("SELECT concept, belief FROM wrong_beliefs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]

# Returns every row of the memory-page tables. Used in memory page to show everything the agent remembers
def read_all(student_id):
    """Everything the tutor remembers, for the memory page."""
    with _db(student_id) as conn:
        return {t: [dict(r) for r in conn.execute(f"SELECT * FROM {t}")] for t in TABLE_KEYS}

# delete 1 row
def delete(student_id, table, key):
    """Delete one row by its key (concept for mastery, id for the others)."""
    if table not in TABLE_KEYS:
        raise ValueError(f"unknown table: {table!r}")
    with _db(student_id) as conn:
        conn.execute(
            f"DELETE FROM {table} WHERE {TABLE_KEYS[table]} = ?",
            (key,)
        )

# Concept list: built from the files at upload, so kept out of TABLE_KEYS and the memory page
def add_concepts(student_id, file, names):
    """Record the concepts found in one file. Names are normalised, so 'LoRA' and ' lora ' are one.
    Returns the normalised names, each once."""
    names = list(dict.fromkeys(norm(n) for n in names if n.strip()))
    with _db(student_id) as conn:
        conn.executemany("INSERT OR IGNORE INTO concepts VALUES (?, ?)", [(n, file) for n in names])
    return names


def remove_file(student_id, file):
    """Drop a deleted file's concepts. A concept another file still has stays."""
    with _db(student_id) as conn:
        conn.execute("DELETE FROM concepts WHERE file = ?", (file,))


def concepts(student_id):
    """The notebook's concept list, sorted, each name once."""
    with _db(student_id) as conn:
        return [r["concept"] for r in conn.execute("SELECT DISTINCT concept FROM concepts ORDER BY concept")]

# shortcut: no edit function yet, memory page (step 17) can delete + re-add; add edit when the UI needs it