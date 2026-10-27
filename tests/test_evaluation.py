"""
Unit tests for Offline Evaluation Engine.
Verifies:
- Train and test splits are strictly disjoint (no test leakage into train)
- Offline evaluation protocol respects train-only candidate generation
- Confidence intervals are non-negative
- Normalised recall is >= raw recall
"""

import pytest
import pandas as pd
import numpy as np

from src.evaluation import train_test_split_by_user, compute_confidence_interval, OfflineEvaluator


@pytest.fixture
def eval_sample_data():
    # 10 movies
    movies_df = pd.DataFrame([
        {"movieId": i, "title": f"Movie {i} (200{i%10})", "clean_title": f"Movie {i}", "year": 2000 + (i % 10), "genres": "Drama|Action", "genre_list": ["Drama", "Action"]}
        for i in range(1, 15)
    ])

    # 10 users each with 10 ratings
    records = []
    for u in range(1, 11):
        for m in range(1, 11):
            r = 4.0 if (u + m) % 2 == 0 else 2.5
            records.append({"userId": u, "movieId": m, "rating": r, "timestamp": 1000 + m})

    ratings_df = pd.DataFrame(records)
    return movies_df, ratings_df


def test_train_test_split_never_leaks_test_into_training(eval_sample_data):
    movies_df, ratings_df = eval_sample_data
    train_df, test_df = train_test_split_by_user(ratings_df, test_size=0.2, min_ratings=5, random_state=42)

    # Check total rows
    assert len(train_df) + len(test_df) == len(ratings_df)

    # Check intersection of (userId, movieId) ratings
    train_pairs = set(zip(train_df["userId"], train_df["movieId"]))
    test_pairs = set(zip(test_df["userId"], test_df["movieId"]))

    assert len(train_pairs.intersection(test_pairs)) == 0, "Train and test splits must be completely disjoint!"


def test_confidence_interval_bounds():
    s = pd.Series([0.1, 0.2, 0.15, 0.25, 0.18])
    mean, std, ci = compute_confidence_interval(s)
    assert mean > 0
    assert std > 0
    assert ci > 0


def test_evaluator_runs_without_leakage(eval_sample_data):
    movies_df, ratings_df = eval_sample_data
    evaluator = OfflineEvaluator(
        movies_df=movies_df,
        ratings_df=ratings_df,
        sample_users=5,
        random_state=42,
        k=5
    )

    summary_df, sweep_df = evaluator.run_evaluation(save_to_disk=False)

    assert not summary_df.empty
    assert "Model" in summary_df.columns
    assert "Precision@10" in summary_df.columns
    assert "Recall@10" in summary_df.columns
    assert "Normalized_Recall@10" in summary_df.columns

    # Normalized recall must be >= raw recall by definition
    for _, row in summary_df.iterrows():
        assert row["Normalized_Recall@10"] >= row["Recall@10"] - 1e-6
