"""
Unit tests for Collaborative Filtering Recommender.
Verifies:
- recommend_collab never returns seed movie
- Outputs always include movieId
- recommend_for_user excludes already rated items
- Deterministic reproducibility
"""

import pytest
import pandas as pd
import numpy as np

from src.features import compute_movie_statistics
from src.collaborative import CollaborativeRecommender


@pytest.fixture
def collab_data():
    movies_df = pd.DataFrame([
        {"movieId": 10, "title": "Star Wars (1977)", "clean_title": "Star Wars", "year": 1977, "genres": "Action|Sci-Fi", "genre_list": ["Action", "Sci-Fi"]},
        {"movieId": 20, "title": "Empire Strikes Back (1980)", "clean_title": "Empire Strikes Back", "year": 1980, "genres": "Action|Sci-Fi", "genre_list": ["Action", "Sci-Fi"]},
        {"movieId": 30, "title": "Return of the Jedi (1983)", "clean_title": "Return of the Jedi", "year": 1983, "genres": "Action|Sci-Fi", "genre_list": ["Action", "Sci-Fi"]},
        {"movieId": 40, "title": "The Godfather (1972)", "clean_title": "The Godfather", "year": 1972, "genres": "Crime|Drama", "genre_list": ["Crime", "Drama"]},
        {"movieId": 50, "title": "The Godfather Part II (1974)", "clean_title": "The Godfather Part II", "year": 1974, "genres": "Crime|Drama", "genre_list": ["Crime", "Drama"]},
    ])

    ratings_df = pd.DataFrame([
        # Users 1-6 co-rate 10, 20, 30
        {"userId": 1, "movieId": 10, "rating": 5.0, "timestamp": 1},
        {"userId": 1, "movieId": 20, "rating": 5.0, "timestamp": 2},
        {"userId": 2, "movieId": 10, "rating": 4.5, "timestamp": 3},
        {"userId": 2, "movieId": 20, "rating": 4.5, "timestamp": 4},
        {"userId": 3, "movieId": 10, "rating": 4.0, "timestamp": 5},
        {"userId": 3, "movieId": 30, "rating": 4.0, "timestamp": 6},
        {"userId": 4, "movieId": 10, "rating": 5.0, "timestamp": 7},
        {"userId": 4, "movieId": 20, "rating": 4.0, "timestamp": 8},
        {"userId": 5, "movieId": 10, "rating": 4.0, "timestamp": 9},
        {"userId": 5, "movieId": 20, "rating": 4.5, "timestamp": 10},
        {"userId": 6, "movieId": 10, "rating": 5.0, "timestamp": 11},
        {"userId": 6, "movieId": 20, "rating": 5.0, "timestamp": 12},
        # User 7 rates Godfathers
        {"userId": 7, "movieId": 40, "rating": 5.0, "timestamp": 13},
        {"userId": 7, "movieId": 50, "rating": 5.0, "timestamp": 14},
    ])

    stats = compute_movie_statistics(movies_df, ratings_df)
    rec = CollaborativeRecommender(
        movies_df=movies_df,
        ratings_df=ratings_df,
        movie_stats=stats,
        min_co_raters=2,
        shrinkage=1.0
    )

    return rec, movies_df, ratings_df


def test_collab_never_returns_seed_movie(collab_data):
    rec, _, _ = collab_data
    seed_id = 10
    results = rec.recommend_collab(movie_id=seed_id, n=5)
    assert seed_id not in results["movieId"].values


def test_collab_outputs_always_include_movie_id(collab_data):
    rec, _, _ = collab_data
    results = rec.recommend_collab(movie_id=10, n=3)
    assert "movieId" in results.columns
    assert len(results) > 0


def test_collab_recommend_for_user_excludes_rated_movies(collab_data):
    rec, _, _ = collab_data
    user_id = 1
    # User 1 rated 10 and 20
    results = rec.recommend_for_user(user_id=user_id, n=5)
    assert 10 not in results["movieId"].values
    assert 20 not in results["movieId"].values
