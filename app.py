"""
CineMatch - Main Streaming Application (Home).
Netflix / Prime Video Style Experience with:
- Profile switcher and new user onboarding (rate 5 movies to kickstart profile)
- Hero recommendation banner with backdrop and synopsis
- Horizontal poster rows: Top picks, Because you liked, Popular, Hidden gems, My List
- Movie detail dialog with live ratings, watchlist actions, and multi-model tabs
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.app_state import get_app_engine
from src.db import (
    list_profiles,
    get_profile,
    create_profile,
    set_profile_onboarded,
    get_user_ratings,
    add_or_update_rating,
    toggle_watchlist,
    is_in_watchlist
)
from src.tmdb_client import (
    prefetch_metadata_batch,
    fetch_movie_metadata
)
from src.ui_components import (
    inject_custom_css,
    render_top_navbar,
    render_tmdb_banner_if_missing,
    get_movie_poster_url,
    show_movie_detail_dialog
)
from src.recommendation_service import get_home_rows

st.set_page_config(
    page_title="CineMatch | Stream Personal Cinema",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# Load engines and datasets
try:
    engine = get_app_engine()
except Exception as e:
    st.error(f"Error initializing CineMatch engine: {str(e)}")
    st.stop()

movies_df = engine["movies_df"]
links_df = engine["links_df"]

# Initialize session state for active profile
if "active_profile_id" not in st.session_state:
    profiles = list_profiles()
    if profiles:
        st.session_state.active_profile_id = profiles[0]["id"]
    else:
        st.session_state.active_profile_id = None

# Active Profile Check
active_profile = get_profile(st.session_state.active_profile_id) if st.session_state.active_profile_id else None

# =========================================================================
# SCREEN 1: PROFILE PICKER / CREATOR
# =========================================================================
if "show_profile_picker" not in st.session_state:
    st.session_state.show_profile_picker = False

if active_profile is None or st.session_state.show_profile_picker:
    st.markdown("<h1 style='text-align:center; color:#E36F61; margin-top:20px;'>Who is watching?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#A896A6;'>Select a profile or create a personalized taste identity</p>", unsafe_allow_html=True)

    profiles = list_profiles()
    cols = st.columns(min(len(profiles) + 1, 5))

    for idx, p in enumerate(profiles):
        with cols[idx % 5]:
            st.markdown(
                f"""
                <div style="background:#1A1324; border:1px solid #372348; border-radius:14px; padding:20px 10px; text-align:center; margin-bottom:12px;">
                    <div style="font-size:3rem; margin-bottom:8px;">{p['avatar']}</div>
                    <div style="font-weight:700; color:#FAF5EC; font-size:1.05rem;">{p['name']}</div>
                    <div style="font-size:0.75rem; color:#A896A6; margin-top:4px;">{'MovieLens Cohort' if p['is_movielens'] else 'Personal Profile'}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(f"Select {p['name'].split()[0]}", key=f"sel_prof_{p['id']}", use_container_width=True):
                st.session_state.active_profile_id = p["id"]
                st.session_state.show_profile_picker = False
                st.rerun()

    # Create new profile form
    with st.expander("➕ Create New Profile", expanded=(len(profiles) == 0)):
        new_c1, new_c2 = st.columns([2, 1])
        with new_c1:
            new_name = st.text_input("Profile Name:", placeholder="e.g. Alex, Cinephile, Maya...")
        with new_c2:
            new_avatar = st.selectbox("Avatar:", ["🍿", "🎬", "🤖", "⭐", "🦊", "🚀", "🎭", "🍕"])

        if st.button("Create Profile", type="primary"):
            if new_name.strip():
                new_id = create_profile(name=new_name.strip(), avatar=new_avatar, is_movielens=False, onboarded=False)
                st.session_state.active_profile_id = new_id
                st.session_state.show_profile_picker = False
                st.success(f"Profile '{new_name}' created!")
                st.rerun()
            else:
                st.warning("Please enter a valid profile name.")
    st.stop()

# =========================================================================
# SCREEN 2: NEW USER ONBOARDING (Rate at least 5 movies)
# =========================================================================
if not active_profile.get("onboarded", 0):
    st.markdown(f"## Welcome to CineMatch, {active_profile['name']}! 👋")
    st.markdown("Let's calibrate your taste profile so our hybrid algorithms can deliver tailored recommendations.")

    user_ratings = get_user_ratings(active_profile["id"])
    rated_count = len(user_ratings)
    progress_val = min(1.0, rated_count / 5.0)

    st.progress(progress_val)
    st.markdown(f"**Onboarding Progress: {rated_count} of 5 movies rated**")

    all_genres = sorted(list({g for sub in movies_df["genre_list"] for g in sub}))
    fav_genres = st.multiselect(
        "Choose your favorite genres:",
        options=all_genres,
        default=["Action", "Sci-Fi", "Animation"]
    )

    st.markdown("### Rate popular titles you have seen:")

    # Top popular movies to seed onboarding
    seed_movie_ids = [1, 260, 296, 318, 356, 527, 593, 858, 2571, 2959, 4993, 7153]
    seed_movies = movies_df[movies_df["movieId"].isin(seed_movie_ids)].to_dict(orient="records")

    cols = st.columns(4)
    for idx, m in enumerate(seed_movies):
        with cols[idx % 4]:
            p_url = get_movie_poster_url(m, links_df)
            st.image(p_url, use_container_width=True)
            st.caption(f"**{m['clean_title']}** ({int(m['year']) if pd.notna(m['year']) else ''})")
            
            mid = m["movieId"]
            current_r = user_ratings.get(mid, 0.0)
            star_val = st.selectbox(
                "Your rating:",
                options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                index=int(current_r),
                key=f"onboard_rate_{mid}",
                format_func=lambda x: "Not seen" if x == 0.0 else f"{int(x)} ★"
            )
            if star_val > 0 and star_val != current_r:
                add_or_update_rating(active_profile["id"], mid, star_val)
                st.rerun()

    if rated_count >= 5:
        st.markdown("---")
        if st.button("🎉 Complete Onboarding & Enter CineMatch", type="primary", use_container_width=True):
            set_profile_onboarded(active_profile["id"], fav_genres)
            st.toast("Taste profile established!")
            st.rerun()
    else:
        st.info("💡 Please rate at least 5 movies above to unlock your personalized home screen.")
    st.stop()

# =========================================================================
# SCREEN 3: MAIN STREAMING EXPERIENCE (Netflix/Prime Video Home)
# =========================================================================

# Top Navigation Bar & TMDb Banner
top_c1, top_c2 = st.columns([5, 1])
with top_c1:
    render_top_navbar(profile_name=active_profile["name"], avatar=active_profile["avatar"])
with top_c2:
    if st.button("🔄 Switch Profile", use_container_width=True):
        st.session_state.show_profile_picker = True
        st.rerun()

render_tmdb_banner_if_missing()

# =========================================================================
# HERO BANNER
# =========================================================================
# Featured title (Pulp Fiction or Inception or top recommendation)
user_ratings = get_user_ratings(active_profile["id"])
hero_movie_id = 296 if 296 not in user_ratings else (2571 if 2571 not in user_ratings else 1)
hero_match = movies_df.loc[movies_df["movieId"] == hero_movie_id]

if not hero_match.empty:
    hero_row = hero_match.iloc[0]
    hero_tmdb_id = links_df.loc[links_df["movieId"] == hero_movie_id, "tmdbId"].iloc[0] if links_df is not None else None
    hero_meta = fetch_movie_metadata(hero_tmdb_id) if hero_tmdb_id else {}
    hero_backdrop = hero_meta.get("backdrop_url") or "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=1200&auto=format&fit=crop"
    hero_overview = hero_meta.get("overview") or "A landmark cinematic masterpiece featuring an ensemble cast, unforgettable dialogue, and iconic storytelling."

    st.markdown(
        f"""
        <div class="hero-banner-container">
            <img src="{hero_backdrop}" class="hero-backdrop-img" />
            <div class="hero-overlay-content">
                <span class="genre-badge" style="font-size:0.8rem; margin-bottom:6px; display:inline-block;">⭐ FEATURED SPOTLIGHT</span>
                <h1 class="hero-title">{hero_row['clean_title']}</h1>
                <div class="hero-meta-row">
                    <span>📅 {int(hero_row['year']) if pd.notna(hero_row['year']) else ''}</span>
                    <span>•</span>
                    <span>★ {hero_row['avg_rating']:.1f}/5.0</span>
                    <span>•</span>
                    <span>{hero_row['genres'].replace('|', ', ')}</span>
                </div>
                <p class="hero-synopsis">{hero_overview}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    btn_h1, btn_h2, _ = st.columns([1.2, 1.2, 4])
    with btn_h1:
        if st.button("ℹ️ View Details", key="hero_view_details", type="primary", use_container_width=True):
            show_movie_detail_dialog(hero_movie_id, active_profile["id"], engine)
    with btn_h2:
        hero_in_wl = is_in_watchlist(active_profile["id"], hero_movie_id)
        if st.button("✅ In My List" if hero_in_wl else "➕ Add to My List", key="hero_add_wl", use_container_width=True):
            new_s = toggle_watchlist(active_profile["id"], hero_movie_id)
            st.toast("Updated My List!")
            st.rerun()

st.markdown("---")

# =========================================================================
# HORIZONTAL ROWS (Netflix Style)
# =========================================================================
with st.spinner("Curating your streaming catalog rows..."):
    home_rows = get_home_rows(active_profile["id"], engine)

# Background prefetch posters for first 20 movies across rows
all_row_tmdb_ids = []
for r in home_rows:
    for m in r["movies"][:6]:
        mid = m["movieId"]
        matched_tmdb = links_df.loc[links_df["movieId"] == mid, "tmdbId"] if links_df is not None else pd.Series()
        if not matched_tmdb.empty:
            all_row_tmdb_ids.append(matched_tmdb.iloc[0])
prefetch_metadata_batch(all_row_tmdb_ids[:25])

for r_idx, row_data in enumerate(home_rows):
    st.markdown(
        f"""
        <div class="row-header">
            <div class="row-title">{row_data['title']}</div>
            <div class="row-tooltip">{row_data['tooltip']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    movies_in_row = row_data["movies"]
    display_limit = min(len(movies_in_row), 6)
    cols = st.columns(display_limit)

    for i in range(display_limit):
        m = movies_in_row[i]
        with cols[i]:
            p_url = get_movie_poster_url(m, links_df)
            st.image(p_url, use_container_width=True)
            st.markdown(f"<div class='poster-title'>{m['clean_title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='poster-sub'><span>★ {m.get('avg_rating', 3.5):.1f}</span><span>{int(m['year']) if pd.notna(m.get('year')) else ''}</span></div>", unsafe_allow_html=True)
            
            why_txt = m.get("why_recommended", "")
            if why_txt:
                st.markdown(f"<div class='poster-why'>{why_txt}</div>", unsafe_allow_html=True)

            if st.button("ℹ️ Details", key=f"det_{r_idx}_{m['movieId']}", use_container_width=True):
                show_movie_detail_dialog(m["movieId"], active_profile["id"], engine)
