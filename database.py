import os
import uuid
import psycopg2
import psycopg2.extras
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    """
    Creates and returns a new PostgreSQL connection.

    We create a fresh connection per operation rather than
    keeping one open permanently. This is safer and avoids
    connection timeout issues in long-running servers.
    """
    conn = psycopg2.connect(DATABASE_URL)
    return conn


def init_database():
    """
    Creates the sessions and messages tables if they don't exist.
    Called once when the FastAPI server starts.

    This is idempotent, safe to call multiple times.
    CREATE TABLE IF NOT EXISTS never fails if table already exists.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized successfully.")


def create_session() -> str:
    """
    Creates a new session in the database and returns its UUID.

    This is called when a new user opens the chatbot.
    The UUID is stored in the browser (localStorage) and sent
    with every request so the server knows which session it belongs to.

    Returns:
        A UUID string representing the new session.
    """
    session_id = str(uuid.uuid4())

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO sessions (id) VALUES (%s)",
        (session_id,)
    )

    conn.commit()
    cursor.close()
    conn.close()

    return session_id


def save_message(session_id: str, role: str, content: str, category: str = ""):
    """
    Saves a single message to the messages table.

    Called after every user message and every agent response.

    Args:
        session_id: The UUID of the current session.
        role: Either 'user' or 'assistant'.
        content: The message text.
        category: Which agent handled it (only for assistant messages).
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO messages (session_id, role, content, category)
        VALUES (%s, %s, %s, %s)
    """, (session_id, role, content, category))

    cursor.execute("""
        UPDATE sessions SET last_active = CURRENT_TIMESTAMP WHERE id = %s
    """, (session_id,))

    conn.commit()
    cursor.close()
    conn.close()


def get_chat_history(session_id: str) -> list:
    """
    Retrieves all messages for a session from the database.

    Returns them in the format expected by run_graph() in graph.py:
        [{"role": "user", "content": "..."}, ...]

    This is how the agent gets memory of previous turns even
    after a page refresh, we reload history from PostgreSQL.

    Args:
        session_id: The UUID of the session to load.

    Returns:
        List of message dicts ordered from oldest to newest.
    """
    conn = get_connection()

    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute("""
        SELECT role, content
        FROM messages
        WHERE session_id = %s
        ORDER BY created_at ASC
    """, (session_id,))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [{"role": row["role"], "content": row["content"]} for row in rows]


def session_exists(session_id: str) -> bool:
    """
    Checks whether a session ID exists in the database.

    Used to validate session IDs sent from the browser.
    If a session doesn't exist (e.g., old/invalid ID), we create a new one.

    Args:
        session_id: The UUID to check.

    Returns:
        True if session exists, False otherwise.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id FROM sessions WHERE id = %s",
        (session_id,)
    )

    exists = cursor.fetchone() is not None
    cursor.close()
    conn.close()

    return exists
