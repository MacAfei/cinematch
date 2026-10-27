"""
Cold-Start Handler Module.
Implements principled fallbacks and visible transparency for:
1. New users with 0 ratings (Bayesian popularity baseline, optionally genre-filtered)
2. New users with 1-4 ratings (adaptive content-heavy hybrid with auto-reduced collab weight)
3. Under-rated movies (metadata content fallback with explicit UI notice)
"""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.features import break_ties_sort
from src.content import ContentRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender


class ColdStartHandler:
    """
    Coordinates cold-start recommendation logic and generates transparency notices.
    """

    def __init__(self, movies_df: pd.DataFrame, movie_stats: pd.DataFrame):
        self.movies_df = movies_df.copy()
        self.movie_stats = movie_stats.copy()

        if "bayesian_rating" not in self.movies_df.columns:
            stats_subset = self.movie_stats[["movieId", "n_ratings", "avg_rating", "bayesian_rating"]].drop_duplicates("movieId")
            self.movies_df = self.movies_df.merge(stats_subset, on="movieId", how="left")
            self.movies_df["n_ratings"] = self.movies_df["n_ratings"].fillna(0).astype(int)
            self.movies_df["avg_rating"] = self.movies_df["avg_rating"].fillna(3.0)
            self.movies_df["bayesian_rating"] = self.movies_df["bayesian_rating"].fillna(3.0)

    def recommend_zero_ratings(
        self,
        n: int = 10,
        selected_genres: Optional[List[str]] = None,
        year_range: Optional[tuple[int, int]] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Cold start for a brand new user with 0 ratings.
        Ranks by Bayesian weighted rating and popularity.
        Returns:
            Tuple of (DataFrame, notice_message)
        """
        candidates = self.movies_df.copy()

        if selected_genres:
            candidates = candidates[candidates["genre_list"].apply(
                lambda glist: any(g in glist for g in selected_genres) if glist else False
            )]

        if year_range is not None:
            min_y, max_y = year_range
            candidates = candidates[
                candidates["year"].apply(lambda y: pd.isna(y) or (min_y <= y <= max_y))
            ]

        # Break ties deterministically:
        # 1. bayesian_rating (descending)
        # 2. n_ratings (descending)
        sorted_candidates = break_ties_sort(
            candidates,
            query_year=None,
            primary_score_col="bayesian_rating",
            ascending=False
        )

        top_candidates = sorted_candidates.head(n).copy()
        top_candidates["why_recommended"] = (
            "Popular cinematic essential with universally acclaim and strong community ratings."
        )

        notice = (
            "✨ **Cold Start Mode Active (0 Ratings)**: You have not rated any movies yet. "
            "Displaying universally acclaimed, top-rated cinematic classics based on Bayesian community ratings."
        )

        output_cols = [
            "movieId",
            "title",
            "genres",
            "year",
            "bayesian_rating",
            "avg_rating",
            "n_ratings",
            "why_recommended"
        ]
        available_cols = [c for c in output_cols if c in top_candidates.columns]
        return top_candidates[available_cols].reset_index(drop=True), notice

    def recommend_few_ratings(
        self,
        rated_pairs: List[Tuple[int, float]],
        hybrid_recommender: HybridRecommender,
        n: int = 10,
        selected_genres: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Cold start for users with 1 to 4 ratings.
        Applies content-heavy hybrid weighting to prevent noisy collaborative drift.
        Returns:
            Tuple of (DataFrame, notice_message)
        """
        num_ratings = len(rated_pairs)
        # Scale content weight: 1 rating -> 0.85, 2 ratings -> 0.80, 3 -> 0.75, 4 -> 0.70
        content_weight = max(0.65, 0.90 - (num_ratings * 0.05))
        collab_weight = 1.0 - content_weight

        recs = hybrid_recommender.recommend_for_profile(
            rated_pairs=rated_pairs,
            w=content_weight,
            n=n,
            selected_genres=selected_genres
        )

        notice = (
            f"ℹ️ **Adaptive Cold Start ({num_ratings} rating{'s' if num_ratings > 1 else ''})**: "
            f"Because you have fewer than 5 ratings, collaborative signals are sparse. "
            f"The engine has dynamically adjusted to a content-heavy hybrid balance "
            f"({content_weight*100:.0f}% genre metadata, {collab_weight*100:.0f}% collaborative taste)."
        )

        return recs, notice

    def check_movie_cold_start(
        self,
        movie_id: int,
        collab_recommender: CollaborativeRecommender,
        min_co_raters: int = 5
    ) -> Tuple[bool, Optional[str]]:
        """
        Checks whether a query movie has sufficient collaborative rating history.
        If under-rated, returns (True, warning_message).
        """
        row = self.movies_df.loc[self.movies_df["movieId"] == movie_id]
        if row.empty:
            return True, "Movie not found in catalog."

        n_ratings = int(row.iloc[0].get("n_ratings", 0))
        title = row.iloc[0].get("clean_title", "Selected movie")

        if movie_id not in collab_recommender.movie_to_idx or n_ratings < min_co_raters:
            msg = (
                f"⚠️ **Low Collaborative Support**: *{title}* has only {n_ratings} community rating(s) "
                f"(threshold: {min_co_raters}). Collaborative recommendations may be sparse; "
                f"falling back to content-based similarity is recommended."
            )
            return True, msg

        return False, None
