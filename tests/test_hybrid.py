"""
Unit tests for Hybrid Recommender.
Verifies:
- Hybrid scores are strictly bounded in [0, 1]
- Outputs always include movieId
- w=1.0 ranking matches content ranking
- w=0.0 ranking matches collaborative ranking
"""

import pytest
import pandas as pd
import numpy as np

from src.features import compute_movie_statistics, build_genre_matrix
from src.content import ContentRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender, min_max_normalize


@pytest.fixture
def hybrid_setup():
    movies_df = pd.DataFrame([
        {"movieId": 1, "title": "Toy Story (1995)", "clean_title": "Toy Story", "year": 1995, "genres": "Adventure|Animation|Children", "genre_list": ["Adventure", "Animation", "Children"]},
        {"movieId": 2, "title": "Jumanji (1995)", "clean_title": "Jumanji", "year": 1995, "genres": "Adventure|Children|Fantasy", "genre_list": ["Adventure", "Children", "Fantasy"]},
        {"movieId": 3, "title": "Lion King (1994)", "clean_title": "Lion King", "year": 1994, "genres": "Adventure|Animation|Children", "genre_list": ["Adventure", "Animation", "Children"]},
        {"movieId": 4, "title": "Bug's Life, A (1998)", "clean_title": "Bug's Life, A", "year": 1998, "genres": "Adventure|Animation|Children", "genre_list": ["Adventure", "Animation", "Children"]},
        {"movieId": 5, "title": "Heat (1995)", "clean_title": "Heat", "year": 1995, "genres": "Action|Crime|Thriller", "genre_list": ["Action", "Crime", "Thriller"]},
    ])

    ratings_df = pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 1},
        {"userId": 1, "movieId": 2, "rating": 5.0, "timestamp": 2},
        {"userId": 2, "movieId": 1, "rating": 4.5, "timestamp": 3},
        {"userId": 2, "movieId": 2, "rating": 4.0, "timestamp": 4},
        {"userId": 3, "movieId": 1, "rating": 4.0, "timestamp": 5},
        {"userId": 3, "movieId": 2, "rating": 4.5, "timestamp": 6},
        {"userId": 4, "movieId": 1, "rating": 5.0, "timestamp": 7},
        {"userId": 4, "movieId": 2, "rating": 5.0, "timestamp": 8},
        {"userId": 5, "movieId": 1, "rating": 4.0, "timestamp": 9},
        {"userId": 5, "movieId": 3, "rating": 4.0, "timestamp": 10},
        {"userId": 6, "movieId": 5, "rating": 5.0, "timestamp": 11},
    ])

    stats = compute_movie_statistics(movies_df, ratings_df)
    genre_mat, mlb, m2i, i2m = build_genre_matrix(movies_df)

    c_rec = ContentRecommender(movies_df, genre_mat, m2i, i2m, stats)
    co_rec = CollaborativeRecommender(movies_df, ratings_df, stats, min_co_raters=2, shrinkage=1.0)
    h_rec = HybridRecommender(c_rec, co_rec)

    return h_rec, c_rec, co_rec


def test_hybrid_scores_are_in_range_zero_to_one(hybrid_setup):
    h_rec, _, _ = hybrid_setup
    for w in [0.0, 0.2, 0.5, 0.8, 1.0]:
        res = h_rec.recommend_hybrid(movie_id=1, w=w, n=10)
        assert len(res) > 0
        assert "movieId" in res.columns
        assert (res["hybrid_score"] >= -1e-6).all() and (res["hybrid_score"] <= 1.000001).all()
        assert (res["content_norm"] >= -1e-6).all() and (res["content_norm"] <= 1.000001).all()
        assert (res["collab_norm"] >= -1e-6).all() and (res["collab_norm"] <= 1.000001).all()


def test_w_one_matches_content_ranking(hybrid_setup):
    h_rec, c_rec, _ = hybrid_setup
    seed = 1
    content_res = c_rec.recommend_content(movie_id=seed, n=4)
    hybrid_res = h_rec.recommend_hybrid(movie_id=seed, w=1.0, n=4)

    assert content_res["movieId"].tolist() == hybrid_res["movieId"].tolist()


def test_w_zero_matches_collab_ranking(hybrid_setup):
    h_rec, _, co_rec = hybrid_setup
    seed = 1
    collab_res = co_rec.recommend_collab(movie_id=seed, n=4)
    hybrid_res = h_rec.recommend_hybrid(movie_id=seed, w=0.0, n=4)

    assert collab_res["movieId"].tolist() == hybrid_res["movieId"].tolist()


def test_min_max_normalization_edge_cases():
    # All zeros
    zeros = np.zeros(5)
    norm_zeros = min_max_normalize(zeros)
    assert np.allclose(norm_zeros, 0.0)

    # Identical positive numbers
    flat = np.array([4.2, 4.2, 4.2])
    norm_flat = min_max_normalize(flat)
    assert np.allclose(norm_flat, 1.0)
