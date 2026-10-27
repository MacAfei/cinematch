"""
Local SQLite persistence module for CineMatch.
Stores user profiles, personal ratings, watchlists, and recent viewing activity.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from src.config import BASE_DIR

DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH = DB_DIR / "app.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initializes tables for profiles, ratings, watchlist, and history."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                avatar TEXT DEFAULT '🎬',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_movielens INTEGER DEFAULT 0,
                movielens_user_id INTEGER DEFAULT NULL,
                fav_genres TEXT DEFAULT '',
                onboarded INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                profile_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (profile_id, movie_id),
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                profile_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (profile_id, movie_id),
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )
        """)
        conn.commit()


def create_profile(
    name: str,
    avatar: str = "🎬",
    is_movielens: bool = False,
    movielens_user_id: Optional[int] = None,
    fav_genres: Optional[List[str]] = None,
    onboarded: bool = False,
    db_path: Optional[Path] = None
) -> int:
    init_db(db_path)
    fav_str = ",".join(fav_genres) if fav_genres else ""
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO profiles (name, avatar, is_movielens, movielens_user_id, fav_genres, onboarded)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, avatar, 1 if is_movielens else 0, movielens_user_id, fav_str, 1 if onboarded else 0)
        )
        conn.commit()
        return cur.lastrowid


def list_profiles(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM profiles ORDER BY created_at ASC").fetchall()
        return [dict(r) for r in rows]


def get_profile(profile_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        return dict(row) if row else None


def set_profile_onboarded(
    profile_id: int,
    fav_genres: Optional[List[str]] = None,
    db_path: Optional[Path] = None
) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        if fav_genres is not None:
            fav_str = ",".join(fav_genres)
            conn.execute(
                "UPDATE profiles SET onboarded = 1, fav_genres = ? WHERE id = ?",
                (fav_str, profile_id)
            )
        else:
            conn.execute("UPDATE profiles SET onboarded = 1 WHERE id = ?", (profile_id,))
        conn.commit()


def add_or_update_rating(
    profile_id: int,
    movie_id: int,
    rating: float,
    db_path: Optional[Path] = None
) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ratings (profile_id, movie_id, rating, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(profile_id, movie_id) DO UPDATE SET
                rating = excluded.rating,
                created_at = CURRENT_TIMESTAMP
            """,
            (profile_id, movie_id, float(rating))
        )
        conn.commit()


def remove_rating(profile_id: int, movie_id: int, db_path: Optional[Path] = None) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM ratings WHERE profile_id = ? AND movie_id = ?", (profile_id, movie_id))
        conn.commit()


def get_user_ratings(profile_id: int, db_path: Optional[Path] = None) -> Dict[int, float]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT movie_id, rating FROM ratings WHERE profile_id = ?",
            (profile_id,)
        ).fetchall()
        return {r["movie_id"]: float(r["rating"]) for r in rows}


def get_user_ratings_df(profile_id: int, db_path: Optional[Path] = None) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as conn:
        query = "SELECT movie_id AS movieId, rating, created_at FROM ratings WHERE profile_id = ? ORDER BY created_at DESC"
        return pd.read_sql_query(query, conn, params=(profile_id,))


def toggle_watchlist(profile_id: int, movie_id: int, db_path: Optional[Path] = None) -> bool:
    """Toggles movie presence in watchlist. Returns True if now in watchlist, False if removed."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM watchlist WHERE profile_id = ? AND movie_id = ?",
            (profile_id, movie_id)
        ).fetchone()
        if row:
            conn.execute("DELETE FROM watchlist WHERE profile_id = ? AND movie_id = ?", (profile_id, movie_id))
            conn.commit()
            return False
        else:
            conn.execute(
                "INSERT INTO watchlist (profile_id, movie_id, added_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (profile_id, movie_id)
            )
            conn.commit()
            return True


def is_in_watchlist(profile_id: int, movie_id: int, db_path: Optional[Path] = None) -> bool:
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM watchlist WHERE profile_id = ? AND movie_id = ?",
            (profile_id, movie_id)
        ).fetchone()
        return row is not None


def get_watchlist(profile_id: int, db_path: Optional[Path] = None) -> List[int]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT movie_id FROM watchlist WHERE profile_id = ? ORDER BY added_at DESC",
            (profile_id,)
        ).fetchall()
        return [r["movie_id"] for r in rows]


def record_view(profile_id: int, movie_id: int, db_path: Optional[Path] = None) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO history (profile_id, movie_id, action) VALUES (?, ?, 'view')",
            (profile_id, movie_id)
        )
        conn.commit()


def get_recently_viewed(profile_id: int, limit: int = 5, db_path: Optional[Path] = None) -> List[int]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT movie_id FROM history
            WHERE profile_id = ? AND action = 'view'
            ORDER BY timestamp DESC LIMIT ?
            """,
            (profile_id, limit)
        ).fetchall()
        # Return unique ordered list
        seen = set()
        res = []
        for r in rows:
            mid = r["movie_id"]
            if mid not in seen:
                seen.add(mid)
                res.append(mid)
        return res


def seed_default_profiles_if_empty(db_path: Optional[Path] = None) -> None:
    """Seeds starter MovieLens profiles if table is empty."""
    init_db(db_path)
    profiles = list_profiles(db_path)
    if not profiles:
        create_profile("Afeefa (Guest)", avatar="🍿", is_movielens=False, onboarded=False, db_path=db_path)
        create_profile("MovieLens User #1", avatar="🤖", is_movielens=True, movielens_user_id=1, onboarded=True, db_path=db_path)
        create_profile("MovieLens User #414 (Heavy Rater)", avatar="🎬", is_movielens=True, movielens_user_id=414, onboarded=True, db_path=db_path)
        create_profile("MovieLens User #599 (Discerning)", avatar="⭐", is_movielens=True, movielens_user_id=599, onboarded=True, db_path=db_path)
