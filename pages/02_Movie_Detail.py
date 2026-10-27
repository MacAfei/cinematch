"""
CineMatch - Movie Detail Dedicated View.
Provides an immersive dedicated page for any selected movie with:
- High-res poster, backdrop, synopsis, runtime, director, and top cast
- Live interactive rating slider and My List toggle (synced with SQLite)
- Multi-strategy recommendation tabs: Similar Movies, People Also Liked, Best For You
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.app_state import get_app_engine
from src.db import (
    list_profiles,
    get_profile,
    add_or_update_rating,
    get_user_ratings,
    toggle_watchlist,
    is_in_watchlist,
    record_view
)
from src.tmdb_client import fetch_movie_metadata, generate_gradient_placeholder
from src.ui_components import (
    inject_custom_css,
    render_top_navbar,
    get_movie_poster_url
)

st.set_page_config(
    page_title="Movie Details | CineMatch",
    page_icon="🎬",
    layout="wide"
)

inject_custom_css()

try:
    engine = get_app_engine()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

movies_df = engine["movies_df"]
links_df = engine["links_df"]
collab_rec = engine["collab_recommender"]
content_rec = engine["content_recommender"]
hybrid_rec = engine["hybrid_recommender"]

# Active profile
if "active_profile_id" not in st.session_state:
    profiles = list_profiles()
    st.session_state.active_profile_id = profiles[0]["id"] if profiles else None

profile = get_profile(st.session_state.active_profile_id) if st.session_state.active_profile_id else None
profile_name = profile["name"] if profile else "Guest"
profile_avatar = profile["avatar"] if profile else "🍿"

render_top_navbar(
    profile_name=profile_name,
    avatar=profile_avatar,
    movies_df=movies_df,
    links_df=links_df
)

# Movie Selector
all_titles = movies_df["title"].tolist()
default_movie_title = "Toy Story (1995)"
default_idx = all_titles.index(default_movie_title) if default_movie_title in all_titles else 0

sel_c1, _ = st.columns([3, 1])
with sel_c1:
    selected_title = st.selectbox("Select or search a movie to inspect:", options=all_titles, index=default_idx)

movie_id = int(movies_df.loc[movies_df["title"] == selected_title, "movieId"].iloc[0])
movie = movies_df.loc[movies_df["movieId"] == movie_id].iloc[0]

# Record view
record_view(st.session_state.active_profile_id, movie_id)

# TMDb metadata
tmdb_id = None
if links_df is not None:
    matched = links_df.loc[links_df["movieId"] == movie_id, "tmdbId"]
    if not matched.empty:
        tmdb_id = matched.iloc[0]

tmdb_meta = fetch_movie_metadata(tmdb_id) if tmdb_id else {}
poster_url = tmdb_meta.get("poster_url") or generate_gradient_placeholder(movie["clean_title"], movie["genres"])
backdrop_url = tmdb_meta.get("backdrop_url")
overview = tmdb_meta.get("overview") or f"{movie['clean_title']} is a recognized classic classified under {movie['genres']}."
runtime_str = f"⏱️ {tmdb_meta['runtime']} mins" if tmdb_meta.get("runtime") else ""
director_str = f"Director: **{tmdb_meta['director']}**" if tmdb_meta.get("director") else ""
cast_str = f"Starring: **{', '.join(tmdb_meta['cast'])}**" if tmdb_meta.get("cast") else ""

if backdrop_url:
    st.markdown(
        f"""
        <div style="border-radius:14px; overflow:hidden; max-height:300px; margin-bottom:20px; box-shadow:0 8px 24px rgba(0,0,0,0.5);">
            <img src="{backdrop_url}" style="width:100%; object-fit:cover;" />
        </div>
        """,
        unsafe_allow_html=True
    )

d_col1, d_col2 = st.columns([1, 2.5])

with d_col1:
    st.image(poster_url, use_container_width=True)
    in_wl = is_in_watchlist(st.session_state.active_profile_id, movie_id)
    if st.button("✅ In My List" if in_wl else "➕ Add to My List", type="primary", use_container_width=True):
        new_state = toggle_watchlist(st.session_state.active_profile_id, movie_id)
        st.toast("Added to My List!" if new_state else "Removed from My List!")
        st.rerun()

with d_col2:
    st.markdown(f"# {movie['clean_title']}")
    meta_bits = [f"📅 {int(movie['year'])}" if pd.notna(movie['year']) else "", runtime_str, f"★ {movie['avg_rating']:.2f} ({movie['n_ratings']:,} community ratings)"]
    st.markdown(" • ".join([b for b in meta_bits if b]))

    genre_badges = "".join([f'<span class="genre-badge" style="margin-right:8px; font-size:0.85rem;">{g}</span>' for g in movie["genre_list"]])
    st.markdown(genre_badges, unsafe_allow_html=True)

    st.markdown(f"<p style='font-size:1.05rem; color:#ECE0E9; line-height:1.55; margin-top:14px;'>{overview}</p>", unsafe_allow_html=True)
    if director_str:
        st.markdown(director_str)
    if cast_str:
        st.markdown(cast_str)

    st.markdown("---")
    st.markdown("### Rate This Movie")
    user_ratings = get_user_ratings(st.session_state.active_profile_id)
    current_val = user_ratings.get(movie_id, 4.0)

    rc1, rc2 = st.columns([3, 1])
    with rc1:
        new_score = st.slider("Your Score:", min_value=0.5, max_value=5.0, value=float(current_val), step=0.5)
    with rc2:
        st.write("")
        st.write("")
        if st.button("⭐ Save Rating", use_container_width=True):
            add_or_update_rating(st.session_state.active_profile_id, movie_id, new_score)
            st.toast(f"Saved {new_score}★ for {movie['clean_title']}!")
            st.rerun()

st.markdown("---")
st.markdown("## Recommendations Based on This Movie")
tab1, tab2, tab3 = st.tabs(["✨ Best For You (Hybrid)", "👥 People Also Liked (Collaborative)", "🏷️ Similar Genres & Metadata (Content)"])

with tab1:
    recs_h = hybrid_rec.recommend_hybrid(movie_id=movie_id, w=0.2, n=6)
    if not recs_h.empty:
        cols = st.columns(len(recs_h))
        for i, (_, r_row) in enumerate(recs_h.iterrows()):
            with cols[i]:
                p_url = get_movie_poster_url(r_row, links_df)
                st.image(p_url, use_container_width=True)
                st.caption(f"**{r_row['clean_title']}** ({int(r_row['year']) if pd.notna(r_row['year']) else ''})")
                st.caption(f"★ {r_row['avg_rating']:.1f}")
                st.caption(f"Match: {r_row['hybrid_score']:.2f}")

with tab2:
    try:
        recs_co = collab_rec.recommend_collab(movie_id=movie_id, n=6)
        if not recs_co.empty:
            cols = st.columns(len(recs_co))
            for i, (_, r_row) in enumerate(recs_co.iterrows()):
                with cols[i]:
                    p_url = get_movie_poster_url(r_row, links_df)
                    st.image(p_url, use_container_width=True)
                    st.caption(f"**{r_row['clean_title']}**")
                    st.caption(f"Collab: {r_row['collab_score']:.2f}")
    except Exception:
        st.info("Limited collaborative support for this movie.")

with tab3:
    recs_c = content_rec.recommend_content(movie_id=movie_id, n=6)
    if not recs_c.empty:
        cols = st.columns(len(recs_c))
        for i, (_, r_row) in enumerate(recs_c.iterrows()):
            with cols[i]:
                p_url = get_movie_poster_url(r_row, links_df)
                st.image(p_url, use_container_width=True)
                st.caption(f"**{r_row['clean_title']}**")
                st.caption(f"Cosine: {r_row['content_score']:.2f}")
