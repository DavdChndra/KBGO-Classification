import os
import secrets
import sqlite3
from datetime import datetime, timezone

from flask import g
from werkzeug.security import generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DB_PATH = os.path.join(INSTANCE_DIR, "app.db")

# Nilai bawaan pengaturan website. Semua bisa diubah dari menu admin.
DEFAULT_SETTINGS = {
    "site_name": "Deteksi KBGO",
    "announcement": "",
    "maintenance_mode": "0",
    "max_text_length": "2000",
    "store_text": "1",
    "log_page_views": "1",
    "rate_limit_per_minute": "20",
    "llm_enabled": "0",
    "llm_mode": "hybrid",  # explain | hybrid
    "llm_model": "openai/gpt-oss-120b",
    "llm_override_threshold": "0.75",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_login TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    visitor_id TEXT NOT NULL,
    ip TEXT,
    text TEXT,
    text_len INTEGER NOT NULL,
    ml_label TEXT NOT NULL,
    ml_prob REAL,
    llm_used INTEGER NOT NULL DEFAULT 0,
    llm_label TEXT,
    llm_category TEXT,
    llm_confidence REAL,
    llm_explanation TEXT,
    llm_error TEXT,
    final_label TEXT NOT NULL,
    disagree INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);
CREATE INDEX IF NOT EXISTS idx_pred_visitor ON predictions(visitor_id);
CREATE TABLE IF NOT EXISTS activity_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    ip TEXT,
    action TEXT NOT NULL,
    detail TEXT,
    user_agent TEXT
);
CREATE INDEX IF NOT EXISTS idx_log_ts ON activity_logs(ts);
CREATE INDEX IF NOT EXISTS idx_log_action ON activity_logs(action);
"""


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Buat tabel, isi pengaturan bawaan, dan buat admin pertama bila belum ada."""
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)
    # Migrasi untuk database lama yang dibuat sebelum kolom ini ada.
    cols = {r[1] for r in db.execute("PRAGMA table_info(predictions)")}
    if "llm_confidence" not in cols:
        db.execute("ALTER TABLE predictions ADD COLUMN llm_confidence REAL")
    for key, value in DEFAULT_SETTINGS.items():
        db.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (key, value))

    if db.execute("SELECT COUNT(*) FROM admins").fetchone()[0] == 0:
        username = os.environ.get("ADMIN_USERNAME", "admin")
        password = os.environ.get("ADMIN_PASSWORD")
        generated = password is None
        if generated:
            password = secrets.token_urlsafe(12)
        db.execute(
            "INSERT INTO admins(username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), now()),
        )
        if generated:
            print("=" * 60)
            print(f"Akun admin pertama dibuat -> username: {username}  password: {password}")
            print("Simpan password ini lalu ganti lewat menu Admin > Akun.")
            print("=" * 60)
    db.commit()
    db.close()


def get_settings():
    rows = get_db().execute("SELECT key, value FROM settings").fetchall()
    data = dict(DEFAULT_SETTINGS)
    data.update({r["key"]: r["value"] for r in rows})
    return data


def set_setting(key, value):
    get_db().execute(
        "INSERT INTO settings(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
