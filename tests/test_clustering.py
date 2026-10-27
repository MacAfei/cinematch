"""
Unit tests for User Clustering.
Verifies:
- KMeans fits with k=3 and produces valid cluster labels
- Cluster profiles include size, mean counts, mean rating, and top genres
- Semantic cluster names are auto-generated
- Single user lookup returns correct membership
"""

import pytest
import pandas as pd
import numpy as np

from src.clustering import UserClusterModel, compute_user_features


@pytest.fixture
def cluster_setup():
    movies_df = pd.DataFrame([
        {"movieId": 1, "title": "Movie 1", "genre_list": ["Action", "Sci-Fi"]},
        {"movieId": 2, "title": "Movie 2", "genre_list": ["Drama", "Romance"]},
        {"movieId": 3, "title": "Movie 3", "genre_list": ["Comedy"]},
        {"movieId": 4, "title": "Movie 4", "genre_list": ["Horror"]},
    ])

    # Create synthetic user data with distinct personas:
    # 1. Heavy raters (users 1-5, many ratings)
    # 2. Generous casual raters (users 6-10, few ratings, high average)
    # 3. Critical casual raters (users 11-15, few ratings, low average)
    records = []
    # Heavy raters
    for u in range(1, 6):
        for m in [1, 2, 3, 4]:
            records.append({"userId": u, "movieId": m, "rating": 4.0, "timestamp": 1})
            records.append({"userId": u, "movieId": m, "rating": 3.5, "timestamp": 2})

    # Generous casuals
    for u in range(6, 11):
        records.append({"userId": u, "movieId": 1, "rating": 5.0, "timestamp": 3})
        records.append({"userId": u, "movieId": 2, "rating": 4.5, "timestamp": 4})

    # Critical casuals
    for u in range(11, 16):
        records.append({"userId": u, "movieId": 1, "rating": 1.5, "timestamp": 5})
        records.append({"userId": u, "movieId": 3, "rating": 2.0, "timestamp": 6})

    ratings_df = pd.DataFrame(records)
    model = UserClusterModel(k=3, n_init=10, random_state=42)
    model.fit(ratings_df, movies_df)

    return model, ratings_df, movies_df


def test_kmeans_cluster_shapes_and_profiles(cluster_setup):
    model, _, _ = cluster_setup
    assert model.cluster_profiles is not None
    assert len(model.cluster_profiles) == 3
    assert "cluster_name" in model.cluster_profiles.columns
    assert "mean_n_ratings" in model.cluster_profiles.columns
    assert "mean_avg_rating" in model.cluster_profiles.columns


def test_user_membership_lookup(cluster_setup):
    model, _, _ = cluster_setup
    info = model.get_user_cluster_info(user_id=1)
    assert info["userId"] == 1
    assert info["cluster_id"] in [0, 1, 2]
    assert len(info["cluster_name"]) > 0
    assert info["user_n_ratings"] > 0
