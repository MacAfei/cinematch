"""
TMDb API Client & Disk Caching Service.
Fetches:
- poster_path, backdrop_path, overview, runtime, top 5 cast, director
- Caches to artifacts/tmdb_cache.json
- Threaded background prefetching with timeouts and rate limit protection
- Robust fallback to styled gradient placeholders when API key or internet is missing
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import requests

from src.config import ARTIFACTS_DIR, TMDB_IMAGE_BASE_URL

CACHE_FILE = ARTIFACTS_DIR / "tmdb_cache.json"

# In-memory cache loaded from disk
_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_DIRTY = False


def get_tmdb_api_key() -> str:
    """Retrieves TMDB_API_KEY from st.secrets first, then environment or .streamlit/secrets.toml."""
    # 1. Try Streamlit secrets first (safely wrapped in try/except)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "TMDB_API_KEY" in st.secrets:
            key = str(st.secrets["TMDB_API_KEY"]).strip()
            if key:
                return key
    except Exception:
        pass

    # 2. Try environment variable
    try:
        env_key = os.environ.get("TMDB_API_KEY", "").strip()
        if env_key:
            return env_key
    except Exception:
        pass

    # 3. Try local secrets.toml file directly
    try:
        secrets_path = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            import toml
            data = toml.load(secrets_path)
            file_key = str(data.get("TMDB_API_KEY", "")).strip()
            if file_key:
                return file_key
    except Exception:
        pass

    return ""


def load_tmdb_cache() -> Dict[str, Dict[str, Any]]:
    global _MEMORY_CACHE
    if _MEMORY_CACHE:
        return _MEMORY_CACHE
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _MEMORY_CACHE = json.load(f)
        except Exception:
            _MEMORY_CACHE = {}
    return _MEMORY_CACHE


def save_tmdb_cache() -> None:
    global _CACHE_DIRTY
    if not _CACHE_DIRTY:
        return
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_MEMORY_CACHE, f)
        _CACHE_DIRTY = False
    except Exception:
        pass


def fetch_movie_metadata(tmdb_id: Any) -> Dict[str, Any]:
    """
    Fetches details for a single tmdb_id, using memory cache and disk cache.
    Returns dict with keys:
    - poster_url, backdrop_url, overview, runtime, cast, director
    """
    if not tmdb_id or str(tmdb_id).strip() == "" or str(tmdb_id).lower() == "nan":
        return {}

    try:
        clean_id = str(int(float(tmdb_id)))
    except (ValueError, TypeError):
        return {}

    cache = load_tmdb_cache()
    if clean_id in cache:
        return cache[clean_id]

    api_key = get_tmdb_api_key()
    if not api_key:
        return {}

    url = f"https://api.themoviedb.org/3/movie/{clean_id}?api_key={api_key}&append_to_response=credits"
    try:
        resp = requests.get(url, timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()

            poster_path = data.get("poster_path")
            backdrop_path = data.get("backdrop_path")

            poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None
            backdrop_url = f"https://image.tmdb.org/t/p/w1280{backdrop_path}" if backdrop_path else None

            credits_data = data.get("credits", {})
            cast_list = [c["name"] for c in credits_data.get("cast", [])[:5]]
            directors = [c["name"] for c in credits_data.get("crew", []) if c.get("job") == "Director"]
            director = directors[0] if directors else None

            meta = {
                "tmdb_id": clean_id,
                "poster_url": poster_url,
                "backdrop_url": backdrop_url,
                "overview": data.get("overview", ""),
                "runtime": data.get("runtime", None),
                "release_date": data.get("release_date", ""),
                "vote_average": data.get("vote_average", None),
                "vote_count": data.get("vote_count", None),
                "cast": cast_list,
                "director": director
            }

            global _CACHE_DIRTY
            cache[clean_id] = meta
            _CACHE_DIRTY = True
            save_tmdb_cache()
            return meta
    except Exception:
        pass

    return {}


def prefetch_metadata_batch(tmdb_ids: List[Any], max_workers: int = 5) -> None:
    """Prefetches metadata for a list of tmdb_ids in background threads."""
    api_key = get_tmdb_api_key()
    if not api_key:
        return

    cache = load_tmdb_cache()
    clean_ids = []
    for tid in tmdb_ids:
        try:
            if tid and str(tid).lower() != "nan":
                cid = str(int(float(tid)))
                if cid not in cache:
                    clean_ids.append(cid)
        except Exception:
            continue

    if not clean_ids:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(fetch_movie_metadata, clean_ids[:30])


def generate_gradient_placeholder(title: str, genres: str, width: int = 300, height: int = 450) -> str:
    """
    Generates an inline SVG data URI for a stylish movie poster placeholder.
    Uses dark-plum and coral gradients with the movie title and genre badge.
    """
    import html
    safe_title = html.escape(title[:30] + ("..." if len(title) > 30 else ""))
    first_genre = genres.split("|")[0] if genres else "Cinema"
    safe_genre = html.escape(first_genre)
    initials = "".join([w[0] for w in title.split() if w and w[0].isalnum()])[:3].upper() or "🎬"

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#2A1424;stop-opacity:1" />
                <stop offset="60%" style="stop-color:#4B2542;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#E36F61;stop-opacity:0.85" />
            </linearGradient>
            <radialGradient id="glow" cx="50%" cy="30%" r="50%">
                <stop offset="0%" style="stop-color:#E36F61;stop-opacity:0.35" />
                <stop offset="100%" style="stop-color:#4B2542;stop-opacity:0" />
            </radialGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#grad)" rx="12" />
        <circle cx="{width/2}" cy="{height*0.35}" r="{width*0.25}" fill="url(#glow)" />
        <text x="50%" y="38%" font-family="sans-serif" font-size="38" font-weight="900" fill="#FAF5EC" text-anchor="middle" opacity="0.95">{initials}</text>
        <rect x="20" y="{height - 85}" width="{width - 40}" height="24" rx="12" fill="#FAF5EC" opacity="0.15" />
        <text x="50%" y="{height - 69}" font-family="sans-serif" font-size="12" font-weight="700" fill="#FAF5EC" text-anchor="middle" letter-spacing="1">{safe_genre.upper()}</text>
        <text x="50%" y="{height - 35}" font-family="sans-serif" font-size="14" font-weight="bold" fill="#FFFFFF" text-anchor="middle">{safe_title}</text>
    </svg>
    """
    import base64
    encoded = base64.b64encode(svg.strip().encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"
