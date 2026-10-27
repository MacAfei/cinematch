"""
Unit tests for Content-Based Recommender.
Verifies:
- Content recommender never returns the seed movie
- Outputs always include movieId
- Deterministic tie-breaking behavior
"""

import pytest
import pandas as pd
import numpy as np
from scipy import sparse

from src.features import compute_movie_statistics, build_genre_matrix
from src.content import ContentRecommender


@pytest.fixture
def sample_data():
    movies_df = pd.DataFrame([
        {"movieId": 1, "title": "Toy Story (1995)", "clean_title": "Toy Story", "year": 1995, "genres": "Adventure|Animation|Children", "genre_list": ["Adventure", "Animation", "Children"]},
        {"movieId": 2, "title": "Jumanji (1995)", "clean_title": "Jumanji", "year": 1995, "genres": "Adventure|Children|Fantasy", "genre_list": ["Adventure", "Children", "Fantasy"]},
        {"movieId": 3, "title": "Grumpier Old Men (1995)", "clean_title": "Grumpier Old Men", "year": 1995, "genres": "Comedy|Romance", "genre_list": ["Comedy", "Romance"]},
        {"movieId": 4, "title": "Bug's Life, A (1998)", "clean_title": "Bug's Life, A", "year": 1998, "genres": "Adventure|Animation|Children", "genre_list": ["Adventure", "Animation", "Children"]},
        {"movieId": 5, "title": "Monsters, Inc. (2001)", "clean_title": "Monsters, Inc.", "year": 2001, "genres": "Adventure|Animation|Children", "genre_list": ["Adventure", "Animation", "Children"]},
    ])

    ratings_df = pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 100},
        {"userId": 1, "movieId": 4, "rating": 4.5, "timestamp": 101},
        {"userId": 2, "movieId": 1, "rating": 4.0, "timestamp": 102},
        {"userId": 2, "movieId": 5, "rating": 3.5, "timestamp": 103},
        {"userId": 3, "movieId": 2, "rating": 4.0, "timestamp": 104},
        {"userId": 3, "movieId": 3, "rating": 2.0, "timestamp": 105},
    ])

    stats = compute_movie_statistics(movies_df, ratings_df)
    genre_mat, mlb, m2i, i2m = build_genre_matrix(movies_df)

    rec = ContentRecommender(
        movies_df=movies_df,
        genre_matrix=genre_mat,
        movie_to_idx=m2i,
        idx_to_movie=i2m,
        movie_stats=stats
    )

    return rec, movies_df, ratings_df


def test_content_recommender_never_returns_seed_movie(sample_data):
    rec, _, _ = sample_data
    seed_id = 1
    results = rec.recommend_content(movie_id=seed_id, n=10)
    assert seed_id not in results["movieId"].values, "Seed movie must not be present in recommendations."


def test_outputs_always_include_movie_id(sample_data):
    rec, _, _ = sample_data
    results = rec.recommend_content(movie_id=1, n=3)
    assert "movieId" in results.columns, "Output dataframe must contain movieId."
    assert len(results) > 0, "Output must contain recommendations."


def test_content_tie_breaking_is_deterministic(sample_data):
    rec, _, _ = sample_data
    # Run multiple times to verify exact reproducibility
    res1 = rec.recommend_content(movie_id=1, n=4)
    res2 = rec.recommend_content(movie_id=1, n=4)
    pd.testing.assert_frame_equal(res1, res2)
