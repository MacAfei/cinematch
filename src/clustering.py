"""
User Clustering Module using K-Means (k=3) on User Rating Habits.
Features:
- StandardScaler on [number of movies rated, average rating]
- KMeans(k=3, n_init=25, random_state=42)
- Auto-generated semantic cluster names derived from empirical cluster centroids
- Top 3 preferred genres per cluster
- Elbow inertia curve and Silhouette scores for k = 2..8 to mathematically justify k=3
- User-specific cluster membership lookup and comparison
"""

from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.config import (
    KMEANS_K,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    ARTIFACTS_DIR
)


def compute_user_features(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-user activity features:
    - n_ratings: number of movies rated
    - avg_rating: average rating given by the user
    """
    user_features = ratings_df.groupby("userId").agg(
        n_ratings=("rating", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()
    return user_features


def generate_cluster_name(
    mean_count: float,
    mean_rating: float,
    overall_mean_count: float,
    overall_mean_rating: float,
    max_cluster_count: float
) -> str:
    """
    Auto-generates an intuitive, readable cluster persona based on empirical centroids.
    """
    # If this cluster's activity is at or near the top count
    if mean_count >= max_cluster_count * 0.7 or mean_count > overall_mean_count * 1.8:
        if mean_rating >= overall_mean_rating:
            return "Heavy Raters (Passionate Cinephiles)"
        else:
            return "Heavy Raters (Critical Reviewers)"

    # Casual / moderate users
    if mean_rating >= overall_mean_rating:
        return "Generous Casual Raters"
    else:
        return "Critical Casual Raters"


class UserClusterModel:
    """
    K-Means user clustering model and profiling engine.
    """

    def __init__(
        self,
        k: int = KMEANS_K,
        n_init: int = KMEANS_N_INIT,
        random_state: int = KMEANS_RANDOM_STATE
    ):
        self.k = k
        self.n_init = n_init
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.k, n_init=self.n_init, random_state=self.random_state)
        self.user_data: Optional[pd.DataFrame] = None
        self.cluster_profiles: Optional[pd.DataFrame] = None
        self.cluster_names: Dict[int, str] = {}

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> "UserClusterModel":
        """
        Fits StandardScaler and KMeans(k=3) on per-user activity features.
        Computes cluster profiles, top genres, and semantic persona names.
        """
        user_df = compute_user_features(ratings_df)
        X = user_df[["n_ratings", "avg_rating"]].to_numpy()
        X_scaled = self.scaler.fit_transform(X)

        labels = self.kmeans.fit_predict(X_scaled)
        user_df["cluster"] = labels

        overall_mean_count = user_df["n_ratings"].mean()
        overall_mean_rating = user_df["avg_rating"].mean()

        # Find top genres per cluster
        # Merge user ratings with movie genres
        ratings_with_genres = ratings_df.merge(
            movies_df[["movieId", "genre_list"]],
            on="movieId",
            how="inner"
        ).merge(
            user_df[["userId", "cluster"]],
            on="userId",
            how="inner"
        )

        cluster_top_genres = {}
        for c in range(self.k):
            sub_ratings = ratings_with_genres[ratings_with_genres["cluster"] == c]
            # Consider ratings >= 3.0 to reflect positive taste
            pos_ratings = sub_ratings[sub_ratings["rating"] >= 3.0]
            genre_series = pos_ratings["genre_list"].explode()
            top_g = genre_series.value_counts().head(3).index.tolist()
            cluster_top_genres[c] = ", ".join(top_g) if top_g else "Drama, Comedy, Action"

        # Compute cluster centroids and metadata
        summary_rows = []
        cluster_means = user_df.groupby("cluster")[["n_ratings", "avg_rating"]].mean()
        max_cluster_count = cluster_means["n_ratings"].max()

        # Sort clusters by activity for consistent naming & presentation
        for c in range(self.k):
            c_users = user_df[user_df["cluster"] == c]
            size = len(c_users)
            pct = (size / len(user_df)) * 100.0
            m_count = c_users["n_ratings"].mean()
            m_rating = c_users["avg_rating"].mean()

            c_name = generate_cluster_name(
                mean_count=m_count,
                mean_rating=m_rating,
                overall_mean_count=overall_mean_count,
                overall_mean_rating=overall_mean_rating,
                max_cluster_count=max_cluster_count
            )
            self.cluster_names[c] = c_name

            summary_rows.append({
                "cluster": c,
                "cluster_name": c_name,
                "user_count": size,
                "percentage": f"{pct:.1f}%",
                "mean_n_ratings": round(m_count, 1),
                "mean_avg_rating": round(m_rating, 2),
                "top_3_genres": cluster_top_genres.get(c, "")
            })

        self.cluster_profiles = pd.DataFrame(summary_rows).sort_values("mean_n_ratings").reset_index(drop=True)
        user_df["cluster_name"] = user_df["cluster"].map(self.cluster_names)
        self.user_data = user_df

        return self

    def get_user_cluster_info(self, user_id: int) -> Dict[str, Any]:
        """
        Retrieves cluster membership and comparative metrics for a specific user ID.
        """
        if self.user_data is None or user_id not in self.user_data["userId"].values:
            raise ValueError(f"User ID {user_id} not found in clustering model.")

        u_row = self.user_data.loc[self.user_data["userId"] == user_id].iloc[0]
        c_id = int(u_row["cluster"])
        c_name = u_row["cluster_name"]
        u_count = int(u_row["n_ratings"])
        u_rating = float(u_row["avg_rating"])

        profile_row = self.cluster_profiles.loc[self.cluster_profiles["cluster"] == c_id].iloc[0]

        return {
            "userId": user_id,
            "cluster_id": c_id,
            "cluster_name": c_name,
            "user_n_ratings": u_count,
            "user_avg_rating": round(u_rating, 2),
            "cluster_mean_ratings": profile_row["mean_n_ratings"],
            "cluster_mean_avg_rating": profile_row["mean_avg_rating"],
            "cluster_size": profile_row["user_count"],
            "cluster_top_genres": profile_row["top_3_genres"]
        }

    def compute_evaluation_curves(self, ratings_df: pd.DataFrame, max_k: int = 8) -> pd.DataFrame:
        """
        Computes inertia (elbow curve) and silhouette scores for k in [2, max_k]
        to empirically justify k=3.
        """
        user_df = compute_user_features(ratings_df)
        X = user_df[["n_ratings", "avg_rating"]].to_numpy()
        X_scaled = StandardScaler().fit_transform(X)

        results = []
        for k_val in range(2, max_k + 1):
            km = KMeans(n_clusters=k_val, n_init=15, random_state=self.random_state)
            labels = km.fit_predict(X_scaled)
            inertia = km.inertia_
            sil = silhouette_score(X_scaled, labels)
            results.append({
                "k": k_val,
                "inertia": round(inertia, 2),
                "silhouette_score": round(sil, 4)
            })

        return pd.DataFrame(results)
