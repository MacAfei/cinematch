"""
CineMatch - My List (Saved Watchlist).
Displays all saved movies from SQLite:
- Grid of saved movie posters
- Quick actions: Remove from My List, View Details, Export to CSV
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.app_state import get_app_engine
from src.db import (
    list_profiles,
    get_profile,
    get_watchlist,
    toggle_watchlist
)
from src.ui_components import (
    inject_custom_css,
    render_top_navbar,
    get_movie_poster_url,
    show_movie_detail_dialog
)

st.set_page_config(
    page_title="My List | CineMatch",
    page_icon="📌",
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

st.title("📌 My List")
st.markdown("Your saved movies queue. Movies added to My List are excluded from general recommendations so you only discover new titles.")

watchlist_ids = get_watchlist(st.session_state.active_profile_id)

if not watchlist_ids:
    st.info("Your watchlist is currently empty. Click '+ Add to My List' on any movie card or detail view to save titles here!")
else:
    wl_movies = movies_df[movies_df["movieId"].isin(watchlist_ids)].copy()

    # Top action bar with CSV export
    act_c1, act_c2 = st.columns([3, 1])
    with act_c1:
        st.markdown(f"**{len(wl_movies)} movie(s) saved in queue**")
    with act_c2:
        csv_data = wl_movies[["movieId", "title", "year", "genres", "avg_rating"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export My List (CSV)",
            data=csv_data,
            file_name=f"watchlist_{profile_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")

    # Display grid
    cols_per_row = 4
    wl_list = wl_movies.to_dict(orient="records")

    for i in range(0, len(wl_list), cols_per_row):
        row_slice = wl_list[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, m in enumerate(row_slice):
            with cols[j]:
                p_url = get_movie_poster_url(m, links_df)
                st.image(p_url, use_container_width=True)
                st.markdown(f"<div class='poster-title'>{m['clean_title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='poster-sub'><span>★ {m.get('avg_rating', 3.5):.1f}</span><span>{int(m['year']) if pd.notna(m.get('year')) else ''}</span></div>", unsafe_allow_html=True)
                
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("ℹ️ Details", key=f"wl_det_{m['movieId']}", use_container_width=True):
                        show_movie_detail_dialog(m["movieId"], st.session_state.active_profile_id, engine)
                with b2:
                    if st.button("🗑️ Remove", key=f"wl_rem_{m['movieId']}", use_container_width=True):
                        toggle_watchlist(st.session_state.active_profile_id, m["movieId"])
                        st.toast(f"Removed {m['clean_title']} from My List")
                        st.rerun()
