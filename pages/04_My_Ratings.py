"""
CineMatch - My Ratings (Personal Ratings History).
Displays all user-rated movies from SQLite:
- Grid of rated movie posters with user stars and timestamp
- Interactive rating edit / remove action
- Export to CSV button
- Immediate feedback loop updating recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.app_state import get_app_engine
from src.db import (
    list_profiles,
    get_profile,
    get_user_ratings_df,
    add_or_update_rating,
    remove_rating
)
from src.ui_components import (
    inject_custom_css,
    render_top_navbar,
    get_movie_poster_url,
    show_movie_detail_dialog
)

st.set_page_config(
    page_title="My Ratings | CineMatch",
    page_icon="⭐",
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

st.title("⭐ My Ratings & Viewing Activity")
st.markdown("Your personal rating history. Every rating you give dynamically refines your personalized recommendations.")

ratings_df_user = get_user_ratings_df(st.session_state.active_profile_id)

if ratings_df_user.empty:
    st.info("You haven't rated any movies yet. Open any movie details or complete onboarding to record your ratings!")
else:
    merged = ratings_df_user.merge(
        movies_df[["movieId", "title", "clean_title", "year", "genres", "avg_rating"]],
        on="movieId",
        how="left"
    )

    r_c1, r_c2 = st.columns([3, 1])
    with r_c1:
        avg_given = merged["rating"].mean()
        st.markdown(f"**{len(merged)} movie(s) rated • Your Average Score: {avg_given:.2f} ★**")
    with r_c2:
        csv_data = merged[["movieId", "clean_title", "year", "genres", "rating", "created_at"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Ratings (CSV)",
            data=csv_data,
            file_name=f"ratings_{profile_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")

    # Display grid
    cols_per_row = 4
    rated_list = merged.to_dict(orient="records")

    for i in range(0, len(rated_list), cols_per_row):
        row_slice = rated_list[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, m in enumerate(row_slice):
            with cols[j]:
                p_url = get_movie_poster_url(m, links_df)
                st.image(p_url, use_container_width=True)
                st.markdown(f"<div class='poster-title'>{m['clean_title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#FFB300; font-weight:bold; font-size:0.95rem;'>Your Score: {m['rating']:.1f} ★</span>", unsafe_allow_html=True)
                st.caption(f"Rated on {str(m.get('created_at', ''))[:10]}")

                # Quick edit rating
                new_r = st.selectbox(
                    "Update score:",
                    options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                    index=int(m["rating"] * 2 - 2) if (1.0 <= m["rating"] <= 5.0) else 6,
                    key=f"edit_rate_{m['movieId']}"
                )

                b_up, b_del = st.columns(2)
                with b_up:
                    if st.button("Save", key=f"save_r_{m['movieId']}", use_container_width=True):
                        add_or_update_rating(st.session_state.active_profile_id, m["movieId"], new_r)
                        st.toast("Rating updated!")
                        st.rerun()
                with b_del:
                    if st.button("Delete", key=f"del_r_{m['movieId']}", use_container_width=True):
                        remove_rating(st.session_state.active_profile_id, m["movieId"])
                        st.toast(f"Removed rating for {m['clean_title']}")
                        st.rerun()
