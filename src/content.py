"""
Content-Based Recommendation Engine using Binary Genre Features and Cosine Similarity.
Deterministic tie-breaking:
1. Bayesian / weighted average rating
2. Number of ratings
3. Release-year closeness to seed movie
4. movieId (guarantees exact reproducibility)
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.features import break_ties_sort


class ContentRecommender:
    """
    Pure Python Content-Based Recommender.
    Operates on sparse genre feature matrix with cosine similarity.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        genre_matrix: sparse.csr_matrix,
        movie_to_idx: Dict[int, int],
        idx_to_movie: Dict[int, int],
        movie_stats: pd.DataFrame
    ):
        self.movies_df = movies_df.copy()
        self.genre_matrix = genre_matrix
        self.movie_to_idx = movie_to_idx
        self.idx_to_movie = idx_to_movie
        self.movie_stats = movie_stats.copy()

        # Merge stats onto movies_df if not already present
        if "bayesian_rating" not in self.movies_df.columns:
            stats_subset = self.movie_stats[["movieId", "n_ratings", "avg_rating", "bayesian_rating"]].drop_duplicates("movieId")
            self.movies_df = self.movies_df.merge(stats_subset, on="movieId", how="left")
            self.movies_df["n_ratings"] = self.movies_df["n_ratings"].fillna(0).astype(int)
            self.movies_df["avg_rating"] = self.movies_df["avg_rating"].fillna(3.0)
            self.movies_df["bayesian_rating"] = self.movies_df["bayesian_rating"].fillna(3.0)

        # Cache movie metadata dict for fast lookups
        self._movie_lookup = self.movies_df.set_index("movieId").to_dict(orient="index")

    def recommend_content(
        self,
        movie_id: int,
        n: int = 10,
        selected_genres: Optional[List[str]] = None,
        year_range: Optional[tuple[int, int]] = None
    ) -> pd.DataFrame:
        """
        Recommends top-n movies similar to `movie_id` based on genre cosine similarity.
        Never returns the seed movie.
        Returns:
            DataFrame with columns:
            ['movieId', 'title', 'genres', 'year', 'content_score', 'avg_rating', 'n_ratings', 'bayesian_rating']
        """
        if movie_id not in self.movie_to_idx:
            raise ValueError(f"Movie ID {movie_id} not found in catalog.")

        query_idx = self.movie_to_idx[movie_id]
        query_vec = self.genre_matrix[query_idx]
        seed_meta = self._movie_lookup.get(movie_id, {})
        query_year = seed_meta.get("year", None)

        # Compute cosine similarity against all movies (sparse x sparse)
        # Result shape: (1, n_movies)
        sim_scores = cosine_similarity(query_vec, self.genre_matrix).flatten()

        # Build candidate DataFrame
        candidates = self.movies_df.copy()
        candidates["content_score"] = sim_scores

        # Exclude the seed movie itself
        candidates = candidates[candidates["movieId"] != movie_id].copy()

        # Optional filtering by user criteria
        if selected_genres:
            # Movie must contain at least one of the selected genres
            candidates = candidates[candidates["genre_list"].apply(
                lambda glist: any(g in glist for g in selected_genres) if glist else False
            )]

        if year_range is not None:
            min_y, max_y = year_range
            candidates = candidates[
                candidates["year"].apply(lambda y: pd.isna(y) or (min_y <= y <= max_y))
            ]

        # Break ties deterministically:
        # 1. content_score (descending)
        # 2. bayesian_rating (descending)
        # 3. n_ratings (descending)
        # 4. year_diff (ascending)
        # 5. movieId (ascending)
        sorted_candidates = break_ties_sort(
            candidates,
            query_year=query_year,
            primary_score_col="content_score",
            ascending=False
        )

        top_candidates = sorted_candidates.head(n).copy()

        output_cols = [
            "movieId",
            "title",
            "clean_title",
            "genres",
            "year",
            "content_score",
            "avg_rating",
            "n_ratings",
            "bayesian_rating"
        ]
        available_cols = [c for c in output_cols if c in top_candidates.columns]
        return top_candidates[available_cols].reset_index(drop=True)
