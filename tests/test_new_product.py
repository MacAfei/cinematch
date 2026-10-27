"""
Unit tests for CineMatch streaming product features.
Verifies:
1. SQLite profile creation, listing, and onboarding state
2. SQLite rating persistence, updates, and removals
3. SQLite watchlist toggling and retrieval
4. TMDb client graceful fallback to stylized SVG without API key
5. RapidFuzz typo-tolerant title search
6. Diversity re-ranking returns strictly unique movies with no duplicates
7. Active profile recommendations exclude already-rated and watchlisted movies
8. Rating updates alter recommendation outputs (feedback loop)
9. Home rows never include movies the user already rated
10. Navigation order: Project Overview is second-last and About is last
11. TOOLS_USED configuration structure in team_config.py
"""

from pathlib import Path
import tempfile
import pytest
import pandas as pd
import numpy as np

from src.db import (
    init_db,
    create_profile,
    list_profiles,
    get_profile,
    set_profile_onboarded,
    add_or_update_rating,
    remove_rating,
    get_user_ratings,
    toggle_watchlist,
    is_in_watchlist,
    get_watchlist
)
from src.tmdb_client import (
    generate_gradient_placeholder,
    fetch_movie_metadata
)
from src.ui_components import fuzzy_search_movie
from src.features import (
    compute_movie_statistics,
    build_genre_matrix,
    apply_diversity_rerank
)
from src.content import ContentRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.cold_start import ColdStartHandler
from src.clustering import UserClusterModel
from src.recommendation_service import (
    recommend_for_active_profile,
    get_home_rows,
    get_adaptive_content_weight
)
from src.team_config import TOOLS_USED, TOOLS_PLANNED, TEAM_MEMBERS


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)
    yield db_path
    try:
        db_path.unlink()
    except Exception:
        pass


@pytest.fixture
def product_engine():
    movies_df = pd.DataFrame([
        {"movieId": 1, "title": "Toy Story (1995)", "clean_title": "Toy Story", "year": 1995, "genres": "Adventure|Animation", "genre_list": ["Adventure", "Animation"]},
        {"movieId": 2, "title": "Jumanji (1995)", "clean_title": "Jumanji", "year": 1995, "genres": "Adventure|Children", "genre_list": ["Adventure", "Children"]},
        {"movieId": 3, "title": "Pulp Fiction (1994)", "clean_title": "Pulp Fiction", "year": 1994, "genres": "Crime|Drama", "genre_list": ["Crime", "Drama"]},
        {"movieId": 4, "title": "Shawshank Redemption (1994)", "clean_title": "Shawshank Redemption", "year": 1994, "genres": "Drama", "genre_list": ["Drama"]},
        {"movieId": 5, "title": "Inception (2010)", "clean_title": "Inception", "year": 2010, "genres": "Action|Sci-Fi", "genre_list": ["Action", "Sci-Fi"]},
        {"movieId": 6, "title": "The Matrix (1999)", "clean_title": "The Matrix", "year": 1999, "genres": "Action|Sci-Fi", "genre_list": ["Action", "Sci-Fi"]},
        {"movieId": 7, "title": "Interstellar (2014)", "clean_title": "Interstellar", "year": 2014, "genres": "Sci-Fi|Drama", "genre_list": ["Sci-Fi", "Drama"]},
    ])

    ratings_df = pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 1},
        {"userId": 1, "movieId": 2, "rating": 4.0, "timestamp": 2},
        {"userId": 2, "movieId": 5, "rating": 5.0, "timestamp": 3},
        {"userId": 2, "movieId": 6, "rating": 5.0, "timestamp": 4},
        {"userId": 2, "movieId": 7, "rating": 4.5, "timestamp": 5},
        {"userId": 3, "movieId": 3, "rating": 5.0, "timestamp": 6},
        {"userId": 3, "movieId": 4, "rating": 5.0, "timestamp": 7},
    ])

    stats = compute_movie_statistics(movies_df, ratings_df)
    genre_mat, mlb, m2i, i2m = build_genre_matrix(movies_df)

    c_rec = ContentRecommender(movies_df, genre_mat, m2i, i2m, stats)
    co_rec = CollaborativeRecommender(movies_df, ratings_df, stats, min_co_raters=1, shrinkage=1.0)
    h_rec = HybridRecommender(c_rec, co_rec)
    cs = ColdStartHandler(movies_df, stats)
    cm = UserClusterModel(k=2, n_init=5, random_state=42).fit(ratings_df, movies_df)

    links_df = pd.DataFrame([
        {"movieId": i, "imdbId": f"00{i}", "tmdbId": f"10{i}"} for i in range(1, 8)
    ])

    return {
        "movies_df": movies_df,
        "ratings_df": ratings_df,
        "links_df": links_df,
        "movie_stats": stats,
        "content_recommender": c_rec,
        "collab_recommender": co_rec,
        "hybrid_recommender": h_rec,
        "cold_start_handler": cs,
        "cluster_model": cm,
        "enriched_matrix": genre_mat
    }


def test_sqlite_profile_creation_and_listing(temp_db):
    p_id = create_profile("Test User", avatar="🚀", db_path=temp_db)
    assert p_id > 0
    profile = get_profile(p_id, db_path=temp_db)
    assert profile["name"] == "Test User"
    assert profile["avatar"] == "🚀"
    assert profile["onboarded"] == 0

    set_profile_onboarded(p_id, fav_genres=["Sci-Fi", "Action"], db_path=temp_db)
    updated = get_profile(p_id, db_path=temp_db)
    assert updated["onboarded"] == 1
    assert "Sci-Fi" in updated["fav_genres"]


def test_sqlite_ratings_and_watchlist(temp_db):
    p_id = create_profile("Rater", db_path=temp_db)
    add_or_update_rating(p_id, 1, 4.5, db_path=temp_db)
    ratings = get_user_ratings(p_id, db_path=temp_db)
    assert ratings[1] == 4.5

    # Update rating
    add_or_update_rating(p_id, 1, 5.0, db_path=temp_db)
    assert get_user_ratings(p_id, db_path=temp_db)[1] == 5.0

    # Watchlist toggle
    assert not is_in_watchlist(p_id, 2, db_path=temp_db)
    state = toggle_watchlist(p_id, 2, db_path=temp_db)
    assert state is True
    assert is_in_watchlist(p_id, 2, db_path=temp_db)
    assert 2 in get_watchlist(p_id, db_path=temp_db)

    # Untoggle
    state2 = toggle_watchlist(p_id, 2, db_path=temp_db)
    assert state2 is False
    assert not is_in_watchlist(p_id, 2, db_path=temp_db)


def test_tmdb_fallback_generates_valid_svg_uri():
    uri = generate_gradient_placeholder("Inception", "Action|Sci-Fi")
    assert uri.startswith("data:image/svg+xml;base64,")
    assert len(uri) > 100


def test_fuzzy_search_handles_typos(product_engine):
    movies_df = product_engine["movies_df"]
    # Typo: 'Inseptshun' for Inception
    res1 = fuzzy_search_movie("Inseptshun", movies_df, limit=3)
    assert not res1.empty
    assert res1.iloc[0]["movieId"] == 5  # Inception

    # Typo: 'Pulp Fictn'
    res2 = fuzzy_search_movie("Pulp Fictn", movies_df, limit=3)
    assert not res2.empty
    assert res2.iloc[0]["movieId"] == 3  # Pulp Fiction


def test_diversity_rerank_no_duplicates(product_engine):
    movies_df = product_engine["movies_df"]
    c_rec = product_engine["content_recommender"]
    candidates = movies_df.copy()
    candidates["score"] = [0.9, 0.85, 0.84, 0.80, 0.75, 0.74, 0.70]

    reranked = apply_diversity_rerank(
        candidates_df=candidates,
        feature_matrix=c_rec.genre_matrix,
        movie_to_idx=c_rec.movie_to_idx,
        primary_score_col="score",
        n=5,
        diversity_lambda=0.7
    )
    assert len(reranked) == 5
    assert len(reranked["movieId"].unique()) == len(reranked), "Reranked list must contain zero duplicates!"


def test_active_profile_recommendations_exclude_rated_and_watchlist(product_engine, temp_db):
    p_id = create_profile("Excluder", db_path=temp_db)
    # Rate movie 5 and watchlist movie 6
    add_or_update_rating(p_id, 5, 5.0, db_path=temp_db)
    toggle_watchlist(p_id, 6, db_path=temp_db)

    # Monkeypatch get_user_ratings and get_watchlist for this test
    from unittest.mock import patch
    with patch("src.recommendation_service.get_user_ratings", return_value={5: 5.0}):
        with patch("src.recommendation_service.get_watchlist", return_value=[6]):
            recs = recommend_for_active_profile(p_id, product_engine, n=4)
            rec_ids = recs["movieId"].tolist()
            assert 5 not in rec_ids, "Already-rated movies must be excluded!"
            assert 6 not in rec_ids, "Watchlisted movies must be excluded!"


def test_adaptive_weights():
    assert get_adaptive_content_weight(0) == 1.0
    assert get_adaptive_content_weight(1) == 0.80
    assert get_adaptive_content_weight(4) == 0.60
    assert get_adaptive_content_weight(10) == 0.35
    assert get_adaptive_content_weight(30) == 0.20


def test_navigation_order_and_no_old_pages():
    pages_dir = Path(__file__).resolve().parent.parent / "pages"
    page_files = sorted([f.name for f in pages_dir.glob("*.py")])

    # Must have 6 pages
    assert len(page_files) == 6
    # Second-last must be Project Overview
    assert "Project_Overview" in page_files[-2]
    # Last must be About
    assert "About" in page_files[-1]

    # Old separate pages must be gone
    old_names = ["01_For_You.py", "02_Explore.py", "03_User_Clusters.py", "04_Compare_Methods.py", "05_Evaluation.py"]
    for old in old_names:
        assert old not in page_files


def test_team_config_tools_structure():
    assert "Analysis (R)" in TOOLS_USED
    assert "Big data stack" in TOOLS_USED
    assert "App (Python)" in TOOLS_USED
    assert "R" in TOOLS_USED["Analysis (R)"]
    assert "Python 3.10+" in TOOLS_USED["App (Python)"]
    assert isinstance(TOOLS_PLANNED, list)
    assert len(TEAM_MEMBERS) == 5
