"""
database.py
------------
Tiny SQLite layer for Mood Beats.

Stores every mood prediction permanently, along with which songs were
shown, so the user can look back at their mood history over time.

No ORM on purpose — this app is small enough that raw sqlite3 keeps it
easy to read and easy to debug.
"""

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timedelta

DB_PATH = "mood_beats.db"
POOL_TTL_HOURS = 12  # how long a mood's Spotify track pool stays fresh


@contextmanager
def get_db():
    """Yield a connection with row access by column name, closing safely."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist yet. Safe to call on every startup."""
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mood_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                energy REAL NOT NULL,
                danceability REAL NOT NULL,
                valence REAL NOT NULL,
                mood TEXT NOT NULL,
                songs_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        # Multi-modal mood detections (face / text / voice / manual fusion).
        # Kept separate from the legacy mood_history table above (which only
        # ever recorded the three-slider path) so existing rows are never
        # touched by this migration-free upgrade.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mood_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mood TEXT NOT NULL,
                modalities TEXT NOT NULL,
                breakdown_json TEXT,
                songs_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        # Cache of track lookups (via Deezer/iTunes, formerly Spotify) so we
        # don't re-hit the API for songs we've already resolved. Column/
        # table names say "spotify" for historical reasons only — no
        # migration needed, they just hold the same shape of data now.
        # name+artist is the natural key.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spotify_cache (
                song_key TEXT PRIMARY KEY,
                track_id TEXT,
                image_url TEXT,
                spotify_url TEXT,
                preview_url TEXT,
                found INTEGER NOT NULL,
                cached_at TEXT NOT NULL
            )
            """
        )
        # Cache of the whole search-result pool per mood, so the app isn't
        # limited to a hand-picked list — it draws from dozens of tracks
        # pulled from Spotify search, refreshed periodically.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mood_track_pool (
                mood TEXT PRIMARY KEY,
                tracks_json TEXT NOT NULL,
                cached_at TEXT NOT NULL
            )
            """
        )
        # Real listen log — written when a song is actually played (Spotify
        # embed playback_update event, or local <audio> play event), not
        # just when it was shown as a recommendation.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS play_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                artist TEXT NOT NULL,
                mood TEXT,
                image_url TEXT,
                spotify_url TEXT,
                played_at TEXT NOT NULL
            )
            """
        )


def save_prediction(energy, danceability, valence, mood, songs):
    """Persist one mood prediction + the songs that were recommended for it."""
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO mood_history (energy, danceability, valence, mood, songs_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                energy,
                danceability,
                valence,
                mood,
                json.dumps([{"name": s["name"], "artist": s["artist"]} for s in songs]),
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )


def save_mood_event(mood, modalities, breakdown, songs):
    """Persist one multi-modal (face/text/voice/manual) mood detection."""
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO mood_events (mood, modalities, breakdown_json, songs_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                mood,
                modalities or "unknown",
                json.dumps(breakdown or {}),
                json.dumps([{"name": s["name"], "artist": s["artist"]} for s in songs]),
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )


def get_history(limit=50):
    """Most recent detections first (multi-modal events + legacy slider-only rows)."""
    with get_db() as conn:
        events = conn.execute(
            "SELECT * FROM mood_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        legacy = conn.execute(
            "SELECT * FROM mood_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

        history = []
        for row in events:
            history.append(
                {
                    "id": f"e{row['id']}",
                    "mood": row["mood"],
                    "modalities": row["modalities"],
                    "breakdown": json.loads(row["breakdown_json"] or "{}"),
                    "songs": json.loads(row["songs_json"] or "[]"),
                    "created_at": row["created_at"],
                }
            )
        for row in legacy:
            history.append(
                {
                    "id": f"h{row['id']}",
                    "mood": row["mood"],
                    "modalities": "manual",
                    "breakdown": {
                        "manual": {
                            "energy": row["energy"],
                            "danceability": row["danceability"],
                            "valence": row["valence"],
                        }
                    },
                    "songs": json.loads(row["songs_json"] or "[]"),
                    "created_at": row["created_at"],
                }
            )
        history.sort(key=lambda h: h["created_at"], reverse=True)
        return history[:limit]


def get_mood_counts():
    """How many times each mood has come up — used for the little history chart."""
    with get_db() as conn:
        counts = {}
        for row in conn.execute("SELECT mood, COUNT(*) as count FROM mood_events GROUP BY mood"):
            counts[row["mood"]] = counts.get(row["mood"], 0) + row["count"]
        for row in conn.execute("SELECT mood, COUNT(*) as count FROM mood_history GROUP BY mood"):
            counts[row["mood"]] = counts.get(row["mood"], 0) + row["count"]
        return counts


def clear_history():
    with get_db() as conn:
        conn.execute("DELETE FROM mood_history")
        conn.execute("DELETE FROM mood_events")


# ---- Spotify cache helpers -------------------------------------------------

def cache_key(name, artist):
    return f"{name.strip().lower()}::{artist.strip().lower()}"


def get_cached_track(name, artist):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM spotify_cache WHERE song_key = ?",
            (cache_key(name, artist),),
        ).fetchone()
        return dict(row) if row else None


def set_cached_track(name, artist, track_id, image_url, spotify_url, preview_url, found):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO spotify_cache (song_key, track_id, image_url, spotify_url, preview_url, found, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(song_key) DO UPDATE SET
                track_id=excluded.track_id,
                image_url=excluded.image_url,
                spotify_url=excluded.spotify_url,
                preview_url=excluded.preview_url,
                found=excluded.found,
                cached_at=excluded.cached_at
            """,
            (
                cache_key(name, artist),
                track_id,
                image_url,
                spotify_url,
                preview_url,
                1 if found else 0,
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )


# ---- Mood track pool (Spotify search results, not a manual list) ----------

def get_mood_pool(mood):
    """Return the cached track pool for a mood, or None if missing/stale."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM mood_track_pool WHERE mood = ?", (mood,)
        ).fetchone()
        if row is None:
            return None
        cached_at = datetime.fromisoformat(row["cached_at"])
        if datetime.utcnow() - cached_at > timedelta(hours=POOL_TTL_HOURS):
            return None
        return json.loads(row["tracks_json"])


def set_mood_pool(mood, tracks):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO mood_track_pool (mood, tracks_json, cached_at)
            VALUES (?, ?, ?)
            ON CONFLICT(mood) DO UPDATE SET
                tracks_json=excluded.tracks_json,
                cached_at=excluded.cached_at
            """,
            (mood, json.dumps(tracks), datetime.utcnow().isoformat(timespec="seconds")),
        )


# ---- Real play log (what was actually listened to) -------------------------

def log_play(name, artist, mood, image_url, spotify_url):
    """Record a real listen. Re-playing a song bumps it back to the top
    instead of creating duplicate entries, so 'last 3 listened' stays
    meaningful."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM play_log WHERE name = ? AND artist = ?", (name, artist)
        )
        conn.execute(
            """
            INSERT INTO play_log (name, artist, mood, image_url, spotify_url, played_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, artist, mood, image_url, spotify_url,
             datetime.utcnow().isoformat(timespec="seconds")),
        )
        # Housekeeping: no need to keep more than the most recent 50 plays.
        conn.execute(
            """
            DELETE FROM play_log WHERE id NOT IN (
                SELECT id FROM play_log ORDER BY id DESC LIMIT 50
            )
            """
        )


def get_recent_plays(limit=3):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM play_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
