"""
CineMatch - Search & Filter Catalog.
Enhanced Netflix/Prime Video style discovery with:
- Form-based search with primary coral submit button and Enter-to-submit
- Real-time typo-tolerant RapidFuzz fuzzy matching
- Multi-attribute filters: Genres (Match ANY / ALL), Year Range, Minimum Rating
- Interactive filter chips with one-click removal
- Result summary line and 5 sort orders
- Pagination / Load More (24 per page)
- One-click smart empty state recovery suggestions
- Persistent session state across movie details navigation
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
    filter_and_sort_movies
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

# Top navbar (search input hidden on Search page to avoid duplicate search boxes)
render_top_navbar(profile_name=profile_name, avatar=profile_avatar, show_search=False)

st.markdown("<h1 style='color:#FAF5EC; font-size:2.2rem; font-weight:900; margin-bottom:4px;'>🔍 Explore & Search Catalog</h1>", unsafe_allow_html=True)
st.caption("Search across 9,700+ titles with typo-tolerant matching, genre logic, rating cutoffs, and release eras.")

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION FOR PERSISTENT SEARCH
# -----------------------------------------------------------------------------
min_catalog_year = int(movies_df["year"].dropna().min()) if "year" in movies_df.columns else 1900
max_catalog_year = int(movies_df["year"].dropna().max()) if "year" in movies_df.columns else 2030

if "search_query_val" not in st.session_state:
    st.session_state.search_query_val = ""
if "search_selected_genres" not in st.session_state:
    st.session_state.search_selected_genres = []
if "search_genre_mode" not in st.session_state:
    st.session_state.search_genre_mode = "Match ANY"
if "search_year_range" not in st.session_state:
    st.session_state.search_year_range = (min_catalog_year, max_catalog_year)
if "search_min_rating" not in st.session_state:
    st.session_state.search_min_rating = 0.0
if "search_sort_by" not in st.session_state:
    st.session_state.search_sort_by = "Best match"
if "search_page_limit" not in st.session_state:
    st.session_state.search_page_limit = 24

# Handle global search trigger from top navbar if passed
if st.session_state.get("search_trigger", False):
    st.session_state.search_trigger = False
    st.session_state.search_page_limit = 24

# -----------------------------------------------------------------------------
# ACTIVE FILTERS BADGE CALCULATION
# -----------------------------------------------------------------------------
active_filter_tags = []
if st.session_state.search_query_val.strip():
    active_filter_tags.append(("query", f"Title: '{st.session_state.search_query_val.strip()}'"))
for g in st.session_state.search_selected_genres:
    active_filter_tags.append((f"genre_{g}", f"{g}"))
if st.session_state.search_genre_mode == "Match ALL" and len(st.session_state.search_selected_genres) > 1:
    active_filter_tags.append(("genre_mode", "Match ALL"))
if st.session_state.search_year_range != (min_catalog_year, max_catalog_year):
    active_filter_tags.append(("year", f"{st.session_state.search_year_range[0]}–{st.session_state.search_year_range[1]}"))
if float(st.session_state.search_min_rating) > 0.0:
    active_filter_tags.append(("rating", f"★ ≥ {st.session_state.search_min_rating:.1f}"))

filter_badge_text = f" ({len(active_filter_tags)} active)" if active_filter_tags else ""

# -----------------------------------------------------------------------------
# 1. SEARCH FORM
# -----------------------------------------------------------------------------
all_genres = sorted(list({g for sub in movies_df["genre_list"] for g in sub if sub}))

with st.form("search_form", clear_on_submit=False):
    # Title box with placeholder & magnifier
    title_input = st.text_input(
        "Search movie titles:",
        value=st.session_state.search_query_val,
        placeholder="🔍 Search movies, e.g. 'matrix' or 'toy story'",
        help="Press Enter or click 'Search' below to find matching titles."
    )

    with st.expander(f"⚙️ Refine Filters{filter_badge_text}", expanded=True):
        col_f1, col_f2 = st.columns([2.5, 1.2])
        with col_f1:
            form_genres = st.multiselect(
                "Filter by Genre(s):",
                options=all_genres,
                default=st.session_state.search_selected_genres,
                placeholder="Choose one or more genres..."
            )
        with col_f2:
            genre_modes = ["Match ANY", "Match ALL"]
            mode_idx = genre_modes.index(st.session_state.search_genre_mode) if st.session_state.search_genre_mode in genre_modes else 0
            form_genre_mode = st.radio(
                "Genre logic:",
                genre_modes,
                index=mode_idx,
                horizontal=True,
                help="'ANY' shows movies with at least one genre; 'ALL' requires all selected genres."
            )

        col_r1, col_r2, col_r3 = st.columns([2.2, 1.4, 1.4])
        with col_r1:
            curr_y_range = st.session_state.search_year_range
            form_year_range = st.slider(
                "Release Era:",
                min_value=min_catalog_year,
                max_value=max_catalog_year,
                value=curr_y_range
            )
        with col_r2:
            form_min_rating = st.slider(
                "Minimum Avg Rating:",
                min_value=0.0,
                max_value=5.0,
                value=float(st.session_state.search_min_rating),
                step=0.5
            )
        with col_r3:
            sort_options = ["Best match", "Highest rated", "Most rated", "Newest", "Oldest"]
            curr_sort_idx = sort_options.index(st.session_state.search_sort_by) if st.session_state.search_sort_by in sort_options else 0
            form_sort_by = st.selectbox(
                "Sort order:",
                sort_options,
                index=curr_sort_idx
            )

    btn_c1, btn_c2 = st.columns([3, 1])
    with btn_c1:
        submit_btn = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)
    with btn_c2:
        clear_btn = st.form_submit_button("🔄 Clear Filters", use_container_width=True)

# -----------------------------------------------------------------------------
# FORM ACTIONS: SUBMIT & CLEAR
# -----------------------------------------------------------------------------
if clear_btn:
    st.session_state.search_query_val = ""
    st.session_state.search_selected_genres = []
    st.session_state.search_genre_mode = "Match ANY"
    st.session_state.search_year_range = (min_catalog_year, max_catalog_year)
    st.session_state.search_min_rating = 0.0
    st.session_state.search_sort_by = "Best match"
    st.session_state.search_page_limit = 24
    st.rerun()

if submit_btn:
    st.session_state.search_query_val = title_input.strip()
    st.session_state.search_selected_genres = form_genres
    st.session_state.search_genre_mode = form_genre_mode
    st.session_state.search_year_range = form_year_range
    st.session_state.search_min_rating = form_min_rating
    st.session_state.search_sort_by = form_sort_by
    st.session_state.search_page_limit = 24
    st.rerun()

# -----------------------------------------------------------------------------
# 2. EXECUTE SEARCH & FILTERING (WITH SPINNER)
# -----------------------------------------------------------------------------
with st.spinner("Searching catalog..."):
    results = filter_and_sort_movies(
        movies_df=movies_df,
        query=st.session_state.search_query_val,
        selected_genres=st.session_state.search_selected_genres,
        genre_match_mode=st.session_state.search_genre_mode,
        year_range=st.session_state.search_year_range,
        min_rating=st.session_state.search_min_rating,
        sort_by=st.session_state.search_sort_by
    )

total_matches = len(results)
display_limit = min(st.session_state.search_page_limit, total_matches)

# -----------------------------------------------------------------------------
# 3. FEEDBACK: SUMMARY LINE & INTERACTIVE FILTER CHIPS
# -----------------------------------------------------------------------------
st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

query_mention = f" for '**{st.session_state.search_query_val}**'" if st.session_state.search_query_val.strip() else ""
sort_mention = f" • Sorted by *{st.session_state.search_sort_by}*"

st.markdown(
    f"<div style='font-size:1.15rem; font-weight:700; color:#FAF5EC; margin-bottom:8px;'>"
    f"Showing {display_limit} of {total_matches} movies{query_mention}"
    f"<span style='font-size:0.88rem; font-weight:400; color:#A896A6;'>{sort_mention}</span>"
    f"</div>",
    unsafe_allow_html=True
)

# Render interactive filter chips with '✕' removal
if active_filter_tags:
    st.caption("Active filters (click ✕ to remove):")
    chip_cols = st.columns(len(active_filter_tags) + 1)
    for c_idx, (f_type, f_label) in enumerate(active_filter_tags):
        with chip_cols[c_idx]:
            if st.button(f"{f_label} ✕", key=f"chip_btn_{f_type}_{c_idx}", use_container_width=True):
                if f_type == "query":
                    st.session_state.search_query_val = ""
                elif f_type.startswith("genre_"):
                    g_to_rm = f_type.replace("genre_", "")
                    if g_to_rm in st.session_state.search_selected_genres:
                        st.session_state.search_selected_genres.remove(g_to_rm)
                elif f_type == "genre_mode":
                    st.session_state.search_genre_mode = "Match ANY"
                elif f_type == "year":
                    st.session_state.search_year_range = (min_catalog_year, max_catalog_year)
                elif f_type == "rating":
                    st.session_state.search_min_rating = 0.0
                st.session_state.search_page_limit = 24
                st.rerun()

    with chip_cols[-1]:
        if st.button("Clear All ✕", key="chip_clear_all", use_container_width=True):
            st.session_state.search_query_val = ""
            st.session_state.search_selected_genres = []
            st.session_state.search_genre_mode = "Match ANY"
            st.session_state.search_year_range = (min_catalog_year, max_catalog_year)
            st.session_state.search_min_rating = 0.0
            st.session_state.search_page_limit = 24
            st.rerun()

# -----------------------------------------------------------------------------
# 4. EMPTY STATE & SMART RECOVERY SUGGESTIONS
# -----------------------------------------------------------------------------
if results.empty:
    st.markdown(
        """
        <div style="background:#1B1224; border:1px solid #372348; border-radius:14px; padding:28px 24px; text-align:center; margin:24px 0;">
            <div style="font-size:2.8rem; margin-bottom:8px;">🎬🔍</div>
            <div style="font-size:1.25rem; font-weight:800; color:#FAF5EC; margin-bottom:6px;">No movies found matching your criteria</div>
            <div style="font-size:0.95rem; color:#A896A6; max-width:550px; margin:0 auto 16px auto;">
                Try relaxing one or more filters below to broaden your recommendations:
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    sug_c1, sug_c2, sug_c3 = st.columns(3)
    with sug_c1:
        if st.session_state.search_min_rating > 0.0:
            if st.button("⭐ Remove Rating Cutoff (Any Rating)", key="fix_rating_btn", use_container_width=True):
                st.session_state.search_min_rating = 0.0
                st.rerun()
        elif st.session_state.search_query_val.strip():
            if st.button("🔍 Clear Title Query", key="fix_query_btn", use_container_width=True):
                st.session_state.search_query_val = ""
                st.rerun()

    with sug_c2:
        if st.session_state.search_selected_genres:
            if st.session_state.search_genre_mode == "Match ALL" and len(st.session_state.search_selected_genres) > 1:
                if st.button("🔀 Switch to 'Match ANY' Genres", key="fix_genre_mode_btn", use_container_width=True):
                    st.session_state.search_genre_mode = "Match ANY"
                    st.rerun()
            else:
                if st.button("🎭 Clear Genre Filters", key="fix_genres_btn", use_container_width=True):
                    st.session_state.search_selected_genres = []
                    st.rerun()

    with sug_c3:
        if st.button("🔄 Reset All Filters to Default", key="fix_reset_all_btn", use_container_width=True):
            st.session_state.search_query_val = ""
            st.session_state.search_selected_genres = []
            st.session_state.search_genre_mode = "Match ANY"
            st.session_state.search_year_range = (min_catalog_year, max_catalog_year)
            st.session_state.search_min_rating = 0.0
            st.rerun()

# -----------------------------------------------------------------------------
# 5. RESULTS GRID & PAGINATION ("LOAD MORE")
# -----------------------------------------------------------------------------
else:
    cols_per_row = 4
    results_slice = results.iloc[:display_limit].to_dict(orient="records")

    for i in range(0, len(results_slice), cols_per_row):
        row_slice = results_slice[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, m in enumerate(row_slice):
            with cols[j]:
                p_url = get_movie_poster_url(m, links_df)
                st.image(p_url, use_container_width=True)
                st.markdown(f"<div class='poster-title'>{m['clean_title']}</div>", unsafe_allow_html=True)
                
                yr_str = str(int(m["year"])) if pd.notna(m.get("year")) else ""
                rt_str = f"★ {m.get('avg_rating', 3.5):.1f}"
                st.markdown(f"<div class='poster-sub'><span>{rt_str}</span><span>{yr_str}</span></div>", unsafe_allow_html=True)
                
                genres_short = m.get("genres", "").replace("|", ", ")
                st.caption(f"{genres_short[:30]}..." if len(genres_short) > 30 else genres_short)
                
                if st.button("ℹ️ View Details", key=f"search_btn_{m['movieId']}", use_container_width=True):
                    show_movie_detail_dialog(m["movieId"], st.session_state.active_profile_id, engine)

    # Pagination: Load More Button
    if total_matches > display_limit:
        st.markdown("<div style='text-align:center; margin:28px 0;'>", unsafe_allow_html=True)
        rem_count = min(24, total_matches - display_limit)
        if st.button(f"⬇️ Load More Movies (+{rem_count})", key="load_more_btn", type="primary", use_container_width=True):
            st.session_state.search_page_limit += 24
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
