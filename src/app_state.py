"""
Shared Application State and Caching Services for Streamlit.
Caches heavy computation with st.cache_data and st.cache_resource,
and persists artifacts to the artifacts/ directory so subsequent runs start instantly.
"""

from pathlib import Path
import pickle
import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
    DEFAULT_HYBRID_WEIGHT,
    KMEANS_K,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE
)
from src.data_loader import load_raw_data
from src.features import compute_movie_statistics, build_genre_matrix
from src.content import ContentRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.cold_start import ColdStartHandler
from src.clustering import UserClusterModel


@st.cache_data(show_spinner="Loading MovieLens Dataset...")
def get_cached_raw_data():
    """Loads raw dataframes with Streamlit caching."""
    movies_df, ratings_df, links_df, tags_df = load_raw_data()
    return movies_df, ratings_df, links_df, tags_df


@st.cache_resource(show_spinner="Initializing Recommendation Engines & Artifacts...")
def get_app_engine():
    """
    Initializes and caches recommendation engines and clustering model.
    Saves/loads precomputed artifacts from artifacts/ directory.
    """
    movies_df, ratings_df, links_df, tags_df = get_cached_raw_data()

    # Ensure working directories exist at startup
    from src.config import DATA_DIR, ARTIFACTS_DIR, OUTPUTS_DIR, BASE_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "db").mkdir(parents=True, exist_ok=True)

    # Precomputed movie statistics (stored as CSV for zero extra dependencies)
    stats_cache_path = ARTIFACTS_DIR / "movie_stats.csv"
    if stats_cache_path.exists():
        try:
            movie_stats = pd.read_csv(stats_cache_path)
        except Exception:
            movie_stats = compute_movie_statistics(movies_df, ratings_df)
            movie_stats.to_csv(stats_cache_path, index=False)
    else:
        movie_stats = compute_movie_statistics(movies_df, ratings_df)
        try:
            movie_stats.to_csv(stats_cache_path, index=False)
        except Exception:
            pass

    # Ensure movies_df has rating statistics columns attached
    if "avg_rating" not in movies_df.columns:
        stats_sub = movie_stats[["movieId", "avg_rating", "n_ratings", "bayesian_rating"]].drop_duplicates("movieId")
        movies_df = movies_df.merge(stats_sub, on="movieId", how="left")
        movies_df["avg_rating"] = movies_df["avg_rating"].fillna(3.0)
        movies_df["n_ratings"] = movies_df["n_ratings"].fillna(0).astype(int)
        movies_df["bayesian_rating"] = movies_df["bayesian_rating"].fillna(3.0)

    # Build genre matrix and enriched multimodal matrix
    genre_matrix, mlb, movie_to_idx, idx_to_movie = build_genre_matrix(movies_df)

    from src.features import build_enriched_content_matrix
    from scipy import sparse
    enriched_cache_path = ARTIFACTS_DIR / "enriched_content_matrix.npz"
    if enriched_cache_path.exists():
        try:
            enriched_matrix = sparse.load_npz(enriched_cache_path)
        except Exception:
            enriched_matrix, _, _ = build_enriched_content_matrix(movies_df, tags_df)
            sparse.save_npz(enriched_cache_path, enriched_matrix)
    else:
        enriched_matrix, _, _ = build_enriched_content_matrix(movies_df, tags_df)
        try:
            sparse.save_npz(enriched_cache_path, enriched_matrix)
        except Exception:
            pass

    # Initialize Content Recommender with enriched matrix
    content_recommender = ContentRecommender(
        movies_df=movies_df,
        genre_matrix=enriched_matrix,
        movie_to_idx=movie_to_idx,
        idx_to_movie=idx_to_movie,
        movie_stats=movie_stats
    )

    # Initialize Collaborative Recommender
    collab_recommender = CollaborativeRecommender(
        movies_df=movies_df,
        ratings_df=ratings_df,
        movie_stats=movie_stats
    )

    # Initialize SQLite database and default profiles
    from src.db import init_db, seed_default_profiles_if_empty
    init_db()
    seed_default_profiles_if_empty()

    # Initialize Hybrid Recommender
    hybrid_recommender = HybridRecommender(
        content_recommender=content_recommender,
        collab_recommender=collab_recommender
    )

    # Initialize Cold Start Handler
    cold_start_handler = ColdStartHandler(
        movies_df=movies_df,
        movie_stats=movie_stats
    )

    # Initialize or load Clustering Model
    cluster_cache_path = ARTIFACTS_DIR / "user_clusters.pkl"
    cluster_model = UserClusterModel(k=KMEANS_K, n_init=KMEANS_N_INIT, random_state=KMEANS_RANDOM_STATE)
    if cluster_cache_path.exists():
        try:
            with open(cluster_cache_path, "rb") as f:
                cluster_model = pickle.load(f)
        except Exception:
            cluster_model.fit(ratings_df, movies_df)
            with open(cluster_cache_path, "wb") as f:
                pickle.dump(cluster_model, f)
    else:
        cluster_model.fit(ratings_df, movies_df)
        try:
            with open(cluster_cache_path, "wb") as f:
                pickle.dump(cluster_model, f)
        except Exception:
            pass

    return {
        "movies_df": movies_df,
        "ratings_df": ratings_df,
        "links_df": links_df,
        "tags_df": tags_df,
        "movie_stats": movie_stats,
        "content_recommender": content_recommender,
        "collab_recommender": collab_recommender,
        "hybrid_recommender": hybrid_recommender,
        "cold_start_handler": cold_start_handler,
        "cluster_model": cluster_model,
        "enriched_matrix": enriched_matrix,
        "genre_matrix": genre_matrix
    }
