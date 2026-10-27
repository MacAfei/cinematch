"""
Unit tests for Cold-Start Handler.
Verifies:
- 0-rating cold-start path returns results and informative notice
- 1-4 ratings cold-start path returns results and adaptive notice
- Under-rated movie cold-start detection
"""

import pytest
import pandas as pd
import numpy as np

from src.features import compute_movie_statistics, build_genre_matrix
from src.content import ContentRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.cold_start import ColdStartHandler


@pytest.fixture
def cold_start_setup():
    movies_df = pd.DataFrame([
        {"movieId": 1, "title": "Movie A (2000)", "clean_title": "Movie A", "year": 2000, "genres": "Action", "genre_list": ["Action"]},
        {"movieId": 2, "title": "Movie B (2001)", "clean_title": "Movie B", "year": 2001, "genres": "Action|Drama", "genre_list": ["Action", "Drama"]},
        {"movieId": 3, "title": "Movie C (2002)", "clean_title": "Movie C", "year": 2002, "genres": "Drama", "genre_list": ["Drama"]},
        {"movieId": 4, "title": "Rare Movie (2003)", "clean_title": "Rare Movie", "year": 2003, "genres": "Comedy", "genre_list": ["Comedy"]},
    ])

    ratings_df = pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 1},
        {"userId": 1, "movieId": 2, "rating": 4.0, "timestamp": 2},
        {"userId": 2, "movieId": 1, "rating": 4.5, "timestamp": 3},
        {"userId": 2, "movieId": 2, "rating": 5.0, "timestamp": 4},
        {"userId": 3, "movieId": 3, "rating": 3.0, "timestamp": 5},
    ])

    stats = compute_movie_statistics(movies_df, ratings_df)
    genre_mat, mlb, m2i, i2m = build_genre_matrix(movies_df)

    c_rec = ContentRecommender(movies_df, genre_mat, m2i, i2m, stats)
    co_rec = CollaborativeRecommender(movies_df, ratings_df, stats, min_co_raters=1, shrinkage=1.0)
    h_rec = HybridRecommender(c_rec, co_rec)
    handler = ColdStartHandler(movies_df, stats)

    return handler, h_rec, co_rec, movies_df


def test_zero_ratings_cold_start_returns_results(cold_start_setup):
    handler, _, _, _ = cold_start_setup
    recs, notice = handler.recommend_zero_ratings(n=3)
    assert len(recs) > 0
    assert "movieId" in recs.columns
    assert "Cold Start" in notice


def test_few_ratings_cold_start_returns_adaptive_notice(cold_start_setup):
    handler, h_rec, _, _ = cold_start_setup
    rated_pairs = [(1, 5.0), (2, 4.0)]
    recs, notice = handler.recommend_few_ratings(rated_pairs, h_rec, n=2)
    assert len(recs) > 0
    assert "movieId" in recs.columns
    assert "Adaptive Cold Start" in notice
    # Exclude already rated movies
    assert 1 not in recs["movieId"].values
    assert 2 not in recs["movieId"].values


def test_movie_cold_start_detection(cold_start_setup):
    handler, _, co_rec, _ = cold_start_setup
    # Movie 4 (Rare Movie) has 0 ratings in ratings_df
    is_cold, msg = handler.check_movie_cold_start(movie_id=4, collab_recommender=co_rec, min_co_raters=5)
    assert is_cold is True
    assert msg is not None
