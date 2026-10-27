"""
Item-Based Collaborative Filtering Recommender using Sparse Matrices.
Features:
- Mean-centered user ratings (adjusted cosine similarity)
- Minimum co-rater threshold (configurable, default 5)
- Shrinkage regularization to prevent low-support pairs from dominating
- High-performance sparse matrix operations (works on laptop and scales to large datasets)
- recommend_collab(movie_id, n): ranks by item similarity score
- recommend_for_user(user_id, n): user rating history weighted aggregation
- Exactly same collaborative score used standalone and inside hybrid recommender
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import sparse

from src.config import (
    MIN_CO_RATERS,
    SHRINKAGE_FACTOR,
    ARTIFACTS_DIR
)
from src.features import break_ties_sort


class CollaborativeRecommender:
    """
    Item-Based Collaborative Filtering Engine using sparse adjusted cosine similarity.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        movie_stats: pd.DataFrame,
        min_co_raters: int = MIN_CO_RATERS,
        shrinkage: float = SHRINKAGE_FACTOR
    ):
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()
        self.movie_stats = movie_stats.copy()
        self.min_co_raters = min_co_raters
        self.shrinkage = shrinkage

        # Ensure movie stats merged
        if "bayesian_rating" not in self.movies_df.columns:
            stats_subset = self.movie_stats[["movieId", "n_ratings", "avg_rating", "bayesian_rating"]].drop_duplicates("movieId")
            self.movies_df = self.movies_df.merge(stats_subset, on="movieId", how="left")
            self.movies_df["n_ratings"] = self.movies_df["n_ratings"].fillna(0).astype(int)
            self.movies_df["avg_rating"] = self.movies_df["avg_rating"].fillna(3.0)
            self.movies_df["bayesian_rating"] = self.movies_df["bayesian_rating"].fillna(3.0)

        self._build_sparse_matrices()

    def _build_sparse_matrices(self) -> None:
        """
        Builds user-item sparse matrices and mean-centers ratings per user.
        """
        # Map unique userIds and movieIds to contiguous matrix indices
        unique_users = np.sort(self.ratings_df["userId"].unique())
        unique_movies = self.movies_df["movieId"].unique()

        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for idx, uid in enumerate(unique_users)}

        self.movie_to_idx = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.idx_to_movie = {idx: mid for idx, mid in enumerate(unique_movies)}

        n_users = len(unique_users)
        n_items = len(unique_movies)

        # Filter ratings to catalog movies
        valid_ratings = self.ratings_df[self.ratings_df["movieId"].isin(self.movie_to_idx)].copy()
        user_indices = valid_ratings["userId"].map(self.user_to_idx).to_numpy(dtype=np.int32)
        movie_indices = valid_ratings["movieId"].map(self.movie_to_idx).to_numpy(dtype=np.int32)
        ratings_values = valid_ratings["rating"].to_numpy(dtype=np.float32)

        # Raw user-item sparse matrix (n_users x n_items)
        self.R = sparse.csr_matrix(
            (ratings_values, (user_indices, movie_indices)),
            shape=(n_users, n_items),
            dtype=np.float32
        )

        # Binary indicator matrix for co-rater counts (B)
        self.B = sparse.csr_matrix(
            (np.ones_like(ratings_values, dtype=np.float32), (user_indices, movie_indices)),
            shape=(n_users, n_items),
            dtype=np.float32
        )

        # Compute user means for mean-centering (adjusted cosine)
        user_sums = np.array(self.R.sum(axis=1)).flatten()
        user_counts = np.array(self.B.sum(axis=1)).flatten()
        # Avoid division by zero
        user_means = np.divide(user_sums, user_counts, out=np.zeros_like(user_sums), where=user_counts > 0)

        # Create mean-centered ratings matrix S
        centered_values = ratings_values - user_means[user_indices]
        self.S = sparse.csr_matrix(
            (centered_values, (user_indices, movie_indices)),
            shape=(n_users, n_items),
            dtype=np.float32
        )

        # Column-normalize S (for items)
        # item norms across users: sqrt(sum(s_ui^2))
        S_csc = self.S.tocsc()
        col_squared_sums = np.array(S_csc.power(2).sum(axis=0)).flatten()
        col_norms = np.sqrt(col_squared_sums)
        col_norms[col_norms == 0] = 1.0  # Avoid zero division

        # Normalized centered matrix
        # Multiply CSC columns by 1 / col_norms
        inv_norms = sparse.diags(1.0 / col_norms)
        self.S_norm_csc = (S_csc @ inv_norms).tocsc()
        self.B_csc = self.B.tocsc()

    def get_item_similarity_vector(self, movie_id: int) -> np.ndarray:
        """
        Computes the collaborative similarity vector for a given movie_id
        against all movies in the catalog.
        Applies:
        - Adjusted cosine similarity
        - Min co-raters threshold
        - Shrinkage regularization: sim * (co_raters / (co_raters + shrinkage))
        - Non-negative clipping
        - Zeroes out self-similarity
        Returns:
            np.ndarray of shape (n_items,) with similarity scores in [0, 1].
        """
        if movie_id not in self.movie_to_idx:
            # Cold start item with no ratings in matrix
            return np.zeros(len(self.movie_to_idx), dtype=np.float32)

        target_idx = self.movie_to_idx[movie_id]

        # Extract target item column
        col_norm = self.S_norm_csc[:, target_idx]  # shape: (n_users, 1)
        col_bin = self.B_csc[:, target_idx]        # shape: (n_users, 1)

        # Matrix-vector multiplications
        # raw_sim = S_norm.T @ col_norm -> shape: (n_items, 1)
        raw_sim = np.array((self.S_norm_csc.T @ col_norm).todense()).flatten()
        # co_raters = B.T @ col_bin -> shape: (n_items, 1)
        co_raters = np.array((self.B_csc.T @ col_bin).todense()).flatten()

        # Apply minimum co-rater threshold
        valid_co = co_raters >= self.min_co_raters

        # Apply shrinkage regularization
        shrinkage_weights = np.divide(
            co_raters,
            co_raters + self.shrinkage,
            out=np.zeros_like(co_raters),
            where=(co_raters + self.shrinkage) > 0
        )

        sim_scores = raw_sim * shrinkage_weights
        # Zero out below threshold and negative similarities
        sim_scores[~valid_co] = 0.0
        sim_scores = np.maximum(0.0, sim_scores)

        # Zero out self-similarity
        sim_scores[target_idx] = 0.0

        return sim_scores.astype(np.float32)

    def recommend_collab(
        self,
        movie_id: int,
        n: int = 10,
        selected_genres: Optional[List[str]] = None,
        year_range: Optional[tuple[int, int]] = None
    ) -> pd.DataFrame:
        """
        Recommends top-n movies similar to `movie_id` based on collaborative similarity score.
        Ranked strictly by similarity score, NOT by average rating alone.
        Deterministic tie-breaking:
        1. collab_score (descending)
        2. bayesian_rating (descending)
        3. n_ratings (descending)
        4. year_diff (ascending)
        5. movieId (ascending)
        """
        if movie_id not in self.movie_to_idx:
            raise ValueError(f"Movie ID {movie_id} has no collaborative rating data.")

        sim_scores = self.get_item_similarity_vector(movie_id)

        candidates = self.movies_df.copy()
        candidates["collab_score"] = sim_scores

        # Exclude seed movie
        candidates = candidates[candidates["movieId"] != movie_id].copy()

        # Optional filters
        if selected_genres:
            candidates = candidates[candidates["genre_list"].apply(
                lambda glist: any(g in glist for g in selected_genres) if glist else False
            )]

        if year_range is not None:
            min_y, max_y = year_range
            candidates = candidates[
                candidates["year"].apply(lambda y: pd.isna(y) or (min_y <= y <= max_y))
            ]

        # Break ties deterministically
        seed_year = self.movies_df.loc[self.movies_df["movieId"] == movie_id, "year"].values
        query_year = int(seed_year[0]) if len(seed_year) > 0 and pd.notna(seed_year[0]) else None

        sorted_candidates = break_ties_sort(
            candidates,
            query_year=query_year,
            primary_score_col="collab_score",
            ascending=False
        )

        top_candidates = sorted_candidates.head(n).copy()

        output_cols = [
            "movieId",
            "title",
            "clean_title",
            "genres",
            "year",
            "collab_score",
            "avg_rating",
            "n_ratings",
            "bayesian_rating"
        ]
        available_cols = [c for c in output_cols if c in top_candidates.columns]
        return top_candidates[available_cols].reset_index(drop=True)

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 10,
        selected_genres: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generates item-based collaborative recommendations for an existing user
        using their rating history.
        Score for unrated item j = sum_{i in rated} (sim(i, j) * rating_i) / (sum sim(i, j) + 1e-6)
        Excludes movies the user has already rated.
        """
        user_ratings_df = self.ratings_df[self.ratings_df["userId"] == user_id]
        if user_ratings_df.empty:
            raise ValueError(f"User ID {user_id} has no ratings in the dataset.")

        rated_movie_ids = set(user_ratings_df["movieId"].tolist())
        user_movie_ratings = dict(zip(user_ratings_df["movieId"], user_ratings_df["rating"]))

        # Aggregate weighted scores across all movies rated by this user
        n_items = len(self.movie_to_idx)
        score_numerator = np.zeros(n_items, dtype=np.float32)
        score_denominator = np.zeros(n_items, dtype=np.float32)

        for mid, r_val in user_movie_ratings.items():
            if mid in self.movie_to_idx:
                sim_vec = self.get_item_similarity_vector(mid)
                score_numerator += sim_vec * float(r_val)
                score_denominator += sim_vec

        pred_scores = np.divide(
            score_numerator,
            score_denominator,
            out=np.zeros_like(score_numerator),
            where=score_denominator > 1e-5
        )

        candidates = self.movies_df.copy()
        candidates["collab_score"] = pred_scores

        # Exclude movies already rated by user
        candidates = candidates[~candidates["movieId"].isin(rated_movie_ids)].copy()

        if selected_genres:
            candidates = candidates[candidates["genre_list"].apply(
                lambda glist: any(g in glist for g in selected_genres) if glist else False
            )]

        sorted_candidates = break_ties_sort(
            candidates,
            query_year=None,
            primary_score_col="collab_score",
            ascending=False
        )

        top_candidates = sorted_candidates.head(n).copy()
        output_cols = [
            "movieId",
            "title",
            "clean_title",
            "genres",
            "year",
            "collab_score",
            "avg_rating",
            "n_ratings",
            "bayesian_rating"
        ]
        available_cols = [c for c in output_cols if c in top_candidates.columns]
        return top_candidates[available_cols].reset_index(drop=True)
