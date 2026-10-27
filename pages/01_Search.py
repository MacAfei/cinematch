"""
CineMatch - Search & Filter Catalog.
Provides:
- Fuzzy title search using RapidFuzz (handles typos, partial matches, and aliases)
- Advanced filters: Genres, Release Year Range, Minimum Rating Threshold
- Interactive poster cards with quick detail modal
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.app_state import get_app_engine
from src.db import list_profiles, get_profile
from src.ui_components import (
    inject_custom_css,
    render_top_navbar,
    get_movie_poster_url,
    show_movie_detail_dialog,
    fuzzy_search_movie
)

st.set_page_config(
    page_title="Search & Explore | CineMatch",
    page_icon="🔍",
    layout="wide"
)

inject_custom_css()

try:
    engine = get_app_engine()
except Exception as e:
    st.error(f"Error initializing data: {str(e)}")
    st.stop()

movies_df = engine["movies_df"]
links_df = engine["links_df"]

# Active profile
if "active_profile_id" not in st.session_state:
    profiles = list_profiles()
    st.session_state.active_profile_id = profiles[0]["id"] if profiles else None

profile = get_profile(st.session_state.active_profile_id) if st.session_state.active_profile_id else None
profile_name = profile["name"] if profile else "Guest"
profile_avatar = profile["avatar"] if profile else "🍿"

render_top_navbar(profile_name=profile_name, avatar=profile_avatar)

st.title("🔍 Search & Filter Catalog")
st.markdown("Search through all 9,700+ movies in the MovieLens catalog with typo-tolerant fuzzy matching and multi-attribute filters.")

# Search bar
search_query = st.text_input(
    "Search by movie title:",
    value="",
    placeholder="e.g. Inception, Dark Knight, Interstellar, Pulp Fiction, Godfthr...",
    key="search_main_query"
)

# Filters row
with st.expander("⚙️ Refine by Genre, Year & Rating", expanded=True):
    f_c1, f_c2, f_c3 = st.columns([2, 2, 1.5])
    with f_c1:
        all_genres = sorted(list({g for sub in movies_df["genre_list"] for g in sub}))
        selected_genres = st.multiselect("Filter by Genre(s):", options=all_genres, default=[])
    with f_c2:
        min_y = int(movies_df["year"].dropna().min())
        max_y = int(movies_df["year"].dropna().max())
        year_range = st.slider("Release Year Range:", min_value=min_y, max_value=max_y, value=(min_y, max_y))
    with f_c3:
        min_star = st.slider("Minimum Rating:", min_value=0.5, max_value=5.0, value=2.5, step=0.5)

# Execute Search
if search_query.strip():
    results = fuzzy_search_movie(search_query, movies_df, limit=30, score_cutoff=40)
else:
    results = movies_df.head(24).copy()

# Apply filters
if selected_genres:
    results = results[results["genre_list"].apply(
        lambda glist: any(g in glist for g in selected_genres) if glist else False
    )]

min_year_val, max_year_val = year_range
results = results[
    results["year"].apply(lambda y: pd.isna(y) or (min_year_val <= y <= max_year_val))
]

if "avg_rating" in results.columns:
    results = results[results["avg_rating"].fillna(0.0) >= min_star]

st.markdown(f"### Matching Titles ({len(results)})")

if results.empty:
    st.info(f"No movies found matching '{search_query}' with the selected filters. Try broadening your criteria.")
else:
    # Render in 4-column poster grid
    cols_per_row = 4
    results_list = results.to_dict(orient="records")

    for i in range(0, len(results_list), cols_per_row):
        row_slice = results_list[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, m in enumerate(row_slice):
            with cols[j]:
                p_url = get_movie_poster_url(m, links_df)
                st.image(p_url, use_container_width=True)
                st.markdown(f"<div class='poster-title'>{m['clean_title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='poster-sub'><span>★ {m.get('avg_rating', 3.5):.1f}</span><span>{int(m['year']) if pd.notna(m.get('year')) else ''}</span></div>", unsafe_allow_html=True)
                if st.button("ℹ️ View Details", key=f"search_btn_{m['movieId']}", use_container_width=True):
                    show_movie_detail_dialog(m["movieId"], st.session_state.active_profile_id, engine)
