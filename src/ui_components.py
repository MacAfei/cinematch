"""
Streamlit UI Components, Styling, and Detail Modals for CineMatch.
Provides:
- Dark streaming platform styling (Netflix/Prime Video aesthetic)
- Horizontal scroll poster rows with hover animations
- Hero banner with backdrop, synopsis, and action buttons
- Detail dialog with full cast, director, runtime, and live rating/watchlist actions
- SVG poster placeholders with dark-plum/coral gradients
"""

import os
import re
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import streamlit as st
from rapidfuzz import process, fuzz

from src.config import TMDB_IMAGE_BASE_URL
from src.tmdb_client import (
    get_tmdb_api_key,
    fetch_movie_metadata,
    generate_gradient_placeholder
)
from src.db import (
    add_or_update_rating,
    get_user_ratings,
    toggle_watchlist,
    is_in_watchlist,
    record_view
)

# Dark streaming theme CSS
CUSTOM_CSS = """
<style>
/* Global Dark Streaming Theme */
:root {
    --bg-dark: #0E0B14;
    --card-bg: #181222;
    --card-border: #2D1F38;
    --accent-coral: #E36F61;
    --accent-coral-hover: #D85A4A;
    --accent-plum: #4B2542;
    --text-cream: #FAF5EC;
    --text-muted: #A896A6;
    --star-gold: #FFB300;
}

/* Base resets */
.main, .stApp {
    background-color: var(--bg-dark) !important;
    color: var(--text-cream) !important;
}

/* Header Navbar */
.cinematch-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0 16px 0;
    border-bottom: 1px solid #23182C;
    margin-bottom: 20px;
}

.cinematch-logo {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.55rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #E36F61 0%, #FF9082 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.profile-pill {
    background-color: #1F162A;
    border: 1px solid #372348;
    border-radius: 20px;
    padding: 5px 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88rem;
    color: var(--text-cream);
}

/* Hero Banner */
.hero-banner-container {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 28px;
    background-color: #160F20;
    border: 1px solid #2E1F3B;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
}

.hero-backdrop-img {
    width: 100%;
    max-height: 400px;
    object-fit: cover;
    opacity: 0.45;
}

.hero-overlay-content {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 28px 32px;
    background: linear-gradient(180deg, rgba(14, 11, 20, 0) 0%, rgba(14, 11, 20, 0.95) 85%);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 900;
    color: #FFFFFF;
    margin: 0 0 6px 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.8);
}

.hero-meta-row {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.92rem;
    color: #D4C3D0;
    margin-bottom: 10px;
}

.hero-synopsis {
    font-size: 0.95rem;
    color: #ECE0E9;
    max-width: 650px;
    line-height: 1.45;
    margin-bottom: 14px;
}

/* Genre Chips */
.genre-badge {
    background-color: rgba(227, 111, 97, 0.18);
    border: 1px solid rgba(227, 111, 97, 0.45);
    color: #FF9082;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 12px;
}

/* Horizontal Poster Rows */
.row-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 22px 0 10px 0;
}

.row-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 8px;
}

.row-tooltip {
    font-size: 0.8rem;
    color: var(--text-muted);
}

/* Poster Card */
.poster-card {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    flex-direction: column;
    height: 100%;
}

.poster-card:hover {
    transform: translateY(-4px) scale(1.02);
    border-color: var(--accent-coral);
    box-shadow: 0 8px 24px rgba(227, 111, 97, 0.22);
}

.poster-img-container {
    position: relative;
    width: 100%;
    aspect-ratio: 2/3;
    overflow: hidden;
    background-color: #120D1A;
}

.poster-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.poster-details {
    padding: 10px;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}

.poster-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 4px 0;
    line-height: 1.25;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.poster-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.poster-why {
    font-size: 0.72rem;
    color: #D6BDCE;
    background-color: rgba(75, 37, 66, 0.4);
    padding: 4px 6px;
    border-radius: 4px;
    border-left: 2px solid var(--accent-coral);
    line-height: 1.25;
    margin-top: auto;
}

/* Metric card overrides for dark mode */
div[data-testid="stMetric"] {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
}

div[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: var(--accent-coral) !important;
    font-weight: 800 !important;
}

/* Buttons */
button[kind="primary"] {
    background-color: var(--accent-coral) !important;
    border-color: var(--accent-coral) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

button[kind="primary"]:hover {
    background-color: var(--accent-coral-hover) !important;
    border-color: var(--accent-coral-hover) !important;
}

/* Custom scrollbar for modern feel */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0E0B14;
}
::-webkit-scrollbar-thumb {
    background: #2D1F38;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent-coral);
}
</style>
"""


def inject_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_tmdb_banner_if_missing():
    """Shows an informative one-time setup hint if TMDB_API_KEY is not configured."""
    key = get_tmdb_api_key()
    if not key:
        with st.expander("🔑 Enable High-Resolution Movie Posters (Free TMDb Setup)", expanded=False):
            st.markdown(
                "**Want official HD posters and backdrops?**  \n"
                "CineMatch works 100% offline with stylized generated posters. To activate official Hollywood posters:\n"
                "1. Get a free API key at [themoviedb.org](https://www.themoviedb.org/settings/api).\n"
                "2. Add it to `.streamlit/secrets.toml`:  \n"
                "   ```toml\n"
                "   TMDB_API_KEY = \"your_api_key_here\"\n"
                "   ```  \n"
                "3. Or set the environment variable: `$env:TMDB_API_KEY=\"your_key\"`."
            )


def render_top_navbar(
    profile_name: str = "Guest",
    avatar: str = "🍿",
    show_search: bool = True,
    movies_df: Optional[pd.DataFrame] = None,
    links_df: Optional[pd.DataFrame] = None
):
    """Renders the top branding, compact global search bar, and active profile pill."""
    api_key = get_tmdb_api_key()
    tmdb_badge = "🟢 HD Posters" if api_key else "⚪ Stylized Mode"

    nav_c1, nav_c2, nav_c3 = st.columns([2.6, 4.4, 3.0], vertical_alignment="center")

    with nav_c1:
        st.markdown(
            """
            <div class="cinematch-logo">
                🎬 CineMatch
            </div>
            """,
            unsafe_allow_html=True
        )

    with nav_c2:
        if show_search:
            global_q = st.text_input(
                "Search movies",
                placeholder="🔍 Search movies, e.g. 'matrix'...",
                label_visibility="collapsed",
                key="global_nav_search_input"
            )
            if global_q and global_q.strip():
                # Detect query change / submission
                if st.session_state.get("prev_global_search_q") != global_q.strip():
                    st.session_state["prev_global_search_q"] = global_q.strip()
                    st.session_state["search_query_val"] = global_q.strip()
                    st.session_state["search_trigger"] = True
                    try:
                        st.switch_page("pages/01_Search.py")
                    except Exception:
                        pass

                # Live suggestions if movies_df is provided
                if movies_df is not None and not movies_df.empty:
                    suggs = fuzzy_search_movie(global_q.strip(), movies_df, limit=5, score_cutoff=45)
                    if not suggs.empty:
                        with st.expander(f"✨ Suggestions for '{global_q.strip()}' (Click to open)", expanded=True):
                            s_cols = st.columns(len(suggs))
                            for idx, (_, s_row) in enumerate(suggs.iterrows()):
                                with s_cols[idx]:
                                    p_url = get_movie_poster_url(s_row, links_df)
                                    st.image(p_url, use_container_width=True)
                                    title_short = s_row.get("clean_title", s_row["title"])[:16]
                                    st.caption(f"**{title_short}**")
                                    if st.button("Open", key=f"nav_open_{s_row['movieId']}", use_container_width=True):
                                        st.session_state["search_query_val"] = s_row.get("clean_title", s_row["title"])
                                        try:
                                            st.switch_page("pages/01_Search.py")
                                        except Exception:
                                            pass

    with nav_c3:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; justify-content:flex-end; gap:8px;">
                <span style="font-size:0.75rem; color:#A896A6; background:#1C1424; padding:4px 8px; border-radius:12px; border:1px solid #30203D;">
                    {tmdb_badge}
                </span>
                <div class="profile-pill">
                    <span>{avatar}</span>
                    <strong>{profile_name}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("<div style='border-bottom:1px solid #23182C; margin-bottom:16px;'></div>", unsafe_allow_html=True)


def get_movie_poster_url(movie_row: Dict[str, Any], links_df: Optional[pd.DataFrame] = None) -> str:
    """
    Retrieves either the official TMDb poster URL or a stylized SVG data URI placeholder.
    """
    movie_id = movie_row.get("movieId")
    title = movie_row.get("clean_title", movie_row.get("title", "Movie"))
    genres = movie_row.get("genres", "")

    tmdb_id = movie_row.get("tmdbId")
    if (not tmdb_id or pd.isna(tmdb_id)) and links_df is not None:
        matched = links_df.loc[links_df["movieId"] == movie_id, "tmdbId"]
        if not matched.empty:
            tmdb_id = matched.iloc[0]

    if tmdb_id and pd.notna(tmdb_id):
        meta = fetch_movie_metadata(tmdb_id)
        if meta and meta.get("poster_url"):
            return meta["poster_url"]

    return generate_gradient_placeholder(title, genres)


@st.dialog("🎬 Movie Details", width="large")
def show_movie_detail_dialog(movie_id: int, profile_id: int, engine: Dict[str, Any]):
    """
    Rich modal displaying movie backdrop, overview, cast, director,
    ratings, watchlist action, and multi-model recommendation tabs.
    """
    movies_df = engine["movies_df"]
    links_df = engine["links_df"]
    collab_rec = engine["collab_recommender"]
    content_rec = engine["content_recommender"]
    hybrid_rec = engine["hybrid_recommender"]

    m_row_matches = movies_df.loc[movies_df["movieId"] == movie_id]
    if m_row_matches.empty:
        st.error("Movie not found.")
        return
    movie = m_row_matches.iloc[0]

    # Record view in history
    record_view(profile_id, movie_id)

    # TMDb metadata
    tmdb_id = None
    if links_df is not None:
        matched = links_df.loc[links_df["movieId"] == movie_id, "tmdbId"]
        if not matched.empty:
            tmdb_id = matched.iloc[0]

    tmdb_meta = fetch_movie_metadata(tmdb_id) if tmdb_id else {}

    poster_url = tmdb_meta.get("poster_url") or generate_gradient_placeholder(movie["clean_title"], movie["genres"])
    backdrop_url = tmdb_meta.get("backdrop_url")
    overview = tmdb_meta.get("overview") or f"{movie['clean_title']} is a featured cinematic title classified under {movie['genres']}."
    runtime_str = f"{tmdb_meta['runtime']} mins" if tmdb_meta.get("runtime") else ""
    director_str = f"Director: **{tmdb_meta['director']}**" if tmdb_meta.get("director") else ""
    cast_str = f"Cast: {', '.join(tmdb_meta['cast'])}" if tmdb_meta.get("cast") else ""

    # Backdrop banner if present
    if backdrop_url:
        st.markdown(
            f"""
            <div style="border-radius:10px; overflow:hidden; max-height:220px; margin-bottom:12px;">
                <img src="{backdrop_url}" style="width:100%; object-fit:cover;" />
            </div>
            """,
            unsafe_allow_html=True
        )

    # Top info columns: Poster on left, Details on right
    c_img, c_info = st.columns([1, 2])
    with c_img:
        st.image(poster_url, use_container_width=True)

    with c_info:
        st.markdown(f"## {movie['clean_title']}")
        sub_items = [f"📅 {int(movie['year'])}" if pd.notna(movie['year']) else "", runtime_str, f"★ {movie['avg_rating']:.1f} ({movie['n_ratings']:,} ratings)"]
        st.markdown(" • ".join([s for s in sub_items if s]))

        # Genre chips
        genre_chips = "".join([f'<span class="genre-badge" style="margin-right:6px;">{g}</span>' for g in movie["genre_list"]])
        st.markdown(genre_chips, unsafe_allow_html=True)

        st.markdown(f"<p style='color:#ECE0E9; line-height:1.45; margin-top:10px;'>{overview}</p>", unsafe_allow_html=True)
        if director_str:
            st.markdown(director_str)
        if cast_str:
            st.caption(cast_str)

        st.markdown("---")

        # Action row: Watchlist & Live Rating
        btn_col, rate_col = st.columns([1.2, 2])
        with btn_col:
            in_wl = is_in_watchlist(profile_id, movie_id)
            wl_label = "✅ In My List" if in_wl else "➕ Add to My List"
            if st.button(wl_label, key=f"wl_modal_{movie_id}", use_container_width=True):
                new_state = toggle_watchlist(profile_id, movie_id)
                st.toast("Added to My List!" if new_state else "Removed from My List!")
                st.rerun()

        with rate_col:
            user_ratings = get_user_ratings(profile_id)
            current_r = user_ratings.get(movie_id, 4.0)
            user_score = st.slider(
                "Your Rating:",
                min_value=0.5,
                max_value=5.0,
                value=float(current_r),
                step=0.5,
                key=f"modal_rate_slider_{movie_id}"
            )
            if st.button("⭐ Save Rating", key=f"save_rate_btn_{movie_id}", use_container_width=True):
                add_or_update_rating(profile_id, movie_id, user_score)
                st.toast(f"Saved {user_score}★ rating for {movie['clean_title']}! Recommendations updated.")
                st.rerun()

    # Recommendations Tabs under detail
    st.markdown("### Related Movies")
    t1, t2, t3 = st.tabs(["✨ Best For You (Hybrid)", "👥 People Also Liked (Collaborative)", "🏷️ Similar Genres (Content)"])

    with t1:
        recs_h = hybrid_rec.recommend_hybrid(movie_id=movie_id, w=0.2, n=6)
        if not recs_h.empty:
            cols = st.columns(len(recs_h))
            for i, (_, r_row) in enumerate(recs_h.iterrows()):
                with cols[i]:
                    p_url = get_movie_poster_url(r_row, links_df)
                    st.image(p_url, use_container_width=True)
                    st.caption(f"**{r_row['clean_title']}** ({int(r_row['year']) if pd.notna(r_row['year']) else ''})")
                    st.caption(f"★ {r_row['avg_rating']:.1f}")

    with t2:
        try:
            recs_co = collab_rec.recommend_collab(movie_id=movie_id, n=6)
            if not recs_co.empty:
                cols = st.columns(len(recs_co))
                for i, (_, r_row) in enumerate(recs_co.iterrows()):
                    with cols[i]:
                        p_url = get_movie_poster_url(r_row, links_df)
                        st.image(p_url, use_container_width=True)
                        st.caption(f"**{r_row['clean_title']}**")
                        st.caption(f"Collab Match: {r_row['collab_score']:.2f}")
        except Exception:
            st.info("Limited collaborative co-ratings available for this title.")

    with t3:
        recs_c = content_rec.recommend_content(movie_id=movie_id, n=6)
        if not recs_c.empty:
            cols = st.columns(len(recs_c))
            for i, (_, r_row) in enumerate(recs_c.iterrows()):
                with cols[i]:
                    p_url = get_movie_poster_url(r_row, links_df)
                    st.image(p_url, use_container_width=True)
                    st.caption(f"**{r_row['clean_title']}**")
                    st.caption(f"Cosine: {r_row['content_score']:.2f}")


def fuzzy_search_movie(
    query: str,
    movies_df: pd.DataFrame,
    limit: int = 100,
    score_cutoff: int = 40
) -> pd.DataFrame:
    """Fuzzy title search using RapidFuzz matching across titles."""
    if not query or not str(query).strip():
        res = movies_df.head(limit).copy()
        res["match_score"] = 0.0
        return res

    clean_q = str(query).strip()
    title_list = movies_df["clean_title"].tolist() if "clean_title" in movies_df.columns else movies_df["title"].tolist()
    matches = process.extract(
        clean_q,
        title_list,
        scorer=fuzz.WRatio,
        limit=limit,
        score_cutoff=score_cutoff
    )

    if not matches:
        return pd.DataFrame(columns=movies_df.columns.tolist() + ["match_score"])

    matched_indices = [match[2] for match in matches]
    scores = [match[1] for match in matches]
    res = movies_df.iloc[matched_indices].copy()
    res["match_score"] = scores
    return res


def filter_and_sort_movies(
    movies_df: pd.DataFrame,
    query: str = "",
    selected_genres: Optional[List[str]] = None,
    genre_match_mode: str = "Match ANY",
    year_range: Optional[Tuple[int, int]] = None,
    min_rating: float = 0.0,
    sort_by: str = "Best match",
    limit: Optional[int] = None,
    score_cutoff: int = 40
) -> pd.DataFrame:
    """
    Applies title fuzzy search, multi-genre filtering (ANY or ALL), release year bounds,
    minimum rating cutoff, and sorting orders.
    """
    if movies_df.empty:
        return pd.DataFrame()

    # 1. Title Search via RapidFuzz
    clean_q = str(query).strip() if query else ""
    if clean_q:
        candidates = fuzzy_search_movie(clean_q, movies_df, limit=limit or 500, score_cutoff=score_cutoff)
    else:
        candidates = movies_df.copy()
        candidates["match_score"] = 0.0

    if candidates.empty:
        return candidates

    # 2. Genre Filtering (ANY vs ALL)
    if selected_genres:
        if genre_match_mode == "Match ALL":
            candidates = candidates[candidates["genre_list"].apply(
                lambda glist: all(g in glist for g in selected_genres) if glist else False
            )]
        else:
            candidates = candidates[candidates["genre_list"].apply(
                lambda glist: any(g in glist for g in selected_genres) if glist else False
            )]

    if candidates.empty:
        return candidates

    # 3. Year Range Filtering
    if year_range is not None:
        min_y, max_y = year_range
        candidates = candidates[
            candidates["year"].apply(lambda y: pd.isna(y) or (min_y <= y <= max_y))
        ]

    if candidates.empty:
        return candidates

    # 4. Minimum Rating Filtering
    if min_rating > 0.0 and "avg_rating" in candidates.columns:
        candidates = candidates[candidates["avg_rating"].fillna(0.0) >= min_rating]

    if candidates.empty:
        return candidates

    # 5. Sorting
    # Options: Best match, Highest rated, Most rated, Newest, Oldest
    if sort_by == "Highest rated":
        sort_cols = ["avg_rating", "bayesian_rating", "n_ratings"]
        asc = [False, False, False]
    elif sort_by == "Most rated":
        sort_cols = ["n_ratings", "bayesian_rating", "avg_rating"]
        asc = [False, False, False]
    elif sort_by == "Newest":
        sort_cols = ["year", "bayesian_rating"]
        asc = [False, False]
    elif sort_by == "Oldest":
        sort_cols = ["year", "bayesian_rating"]
        asc = [True, False]
    else:  # "Best match"
        if clean_q and "match_score" in candidates.columns:
            sort_cols = ["match_score", "bayesian_rating", "n_ratings"]
            asc = [False, False, False]
        else:
            sort_cols = ["bayesian_rating", "n_ratings"]
            asc = [False, False]

    avail_sort_cols = [c for c in sort_cols if c in candidates.columns]
    avail_asc = asc[:len(avail_sort_cols)]
    if avail_sort_cols:
        candidates = candidates.sort_values(by=avail_sort_cols, ascending=avail_asc, na_position="last")

    return candidates.reset_index(drop=True)
