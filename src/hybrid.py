"""
Hybrid Recommendation Engine combining Content-Based and Collaborative Filtering.
Features:
- Min-max normalization for both content and collaborative score distributions
- Configurable weighting w: hybrid_score = w * content_norm + (1 - w) * collab_norm
- Deterministic tie-breaking
- Human-readable "why recommended" natural language explanations
- "My Profile" taste profile mode for new and existing users
- Seamless fallback and cold-start support
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DEFAULT_HYBRID_WEIGHT
from src.features import break_ties_sort
from src.content import ContentRecommender
from src.collaborative import CollaborativeRecommender


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalizes scores to [0.0, 1.0].
    Guards against divide-by-zero when all values are identical.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    s_min = np.nanmin(scores_arr) if len(scores_arr) > 0 else 0.0
    s_max = np.nanmax(scores_arr) if len(scores_arr) > 0 else 0.0

    if np.isnan(s_min) or np.isnan(s_max):
        return np.zeros_like(scores_arr)

    spread = s_max - s_min
    if spread > 1e-9:
        normalized = (scores_arr - s_min) / spread
        return np.clip(normalized, 0.0, 1.0)
    
    # If all non-zero and identical, map to 1.0; if all 0, return 0.0
    if s_max > 0:
        return np.ones_like(scores_arr)
    return np.zeros_like(scores_arr)


def generate_explanation(
    seed_title: str,
    seed_genres: List[str],
    rec_title: str,
    rec_genres: List[str],
    content_norm: float,
    collab_norm: float
) -> str:
    """
    Generates a concise, transparent human-readable explanation for the recommendation.
    """
    shared = [g for g in seed_genres if g in rec_genres]
    genre_phrase = f"Shares {', '.join(shared[:3])}" if shared else "Genre crossover"

    if collab_norm >= 0.6:
        collab_phrase = f"frequently loved by viewers who enjoyed {seed_title}"
    elif collab_norm >= 0.25:
        collab_phrase = f"shared audience overlap with {seed_title}"
    else:
        collab_phrase = "aligned thematic match"

    return f"{genre_phrase} with {seed_title}; {collab_phrase}."


class HybridRecommender:
    """
    Combines Content-Based and Collaborative scores with min-max normalization.
    """

    def __init__(
        self,
        content_recommender: ContentRecommender,
        collab_recommender: CollaborativeRecommender
    ):
        self.content_recommender = content_recommender
        self.collab_recommender = collab_recommender
        self.movies_df = content_recommender.movies_df.copy()

    def recommend_hybrid(
        self,
        movie_id: int,
        w: float = DEFAULT_HYBRID_WEIGHT,
        n: int = 10,
        selected_genres: Optional[List[str]] = None,
        year_range: Optional[tuple[int, int]] = None
    ) -> pd.DataFrame:
        """
        Computes hybrid recommendations for a seed movie_id.
        hybrid_score = w * content_norm + (1 - w) * collab_norm
        Returns:
            DataFrame with columns:
            ['movieId', 'title', 'genres', 'year', 'content_norm', 'collab_norm',
             'hybrid_score', 'why_recommended', 'avg_rating', 'n_ratings', 'bayesian_rating']
        """
        if movie_id not in self.content_recommender.movie_to_idx:
            raise ValueError(f"Movie ID {movie_id} not found in catalog.")

        # Candidate pool is all movies excluding the seed movie
        candidates = self.movies_df[self.movies_df["movieId"] != movie_id].copy()

        # Apply user filters if any
        if selected_genres:
            candidates = candidates[candidates["genre_list"].apply(
                lambda glist: any(g in glist for g in selected_genres) if glist else False
            )]

        if year_range is not None:
            min_y, max_y = year_range
            candidates = candidates[
                candidates["year"].apply(lambda y: pd.isna(y) or (min_y <= y <= max_y))
            ]

        if candidates.empty:
            return pd.DataFrame()

        # 1. Content scores
        query_idx = self.content_recommender.movie_to_idx[movie_id]
        query_vec = self.content_recommender.genre_matrix[query_idx]
        candidate_indices = [self.content_recommender.movie_to_idx[mid] for mid in candidates["movieId"]]
        candidate_genre_matrix = self.content_recommender.genre_matrix[candidate_indices]
        content_raw = cosine_similarity(query_vec, candidate_genre_matrix).flatten()

        # 2. Collaborative scores (MUST be identical to standalone collab score)
        full_collab_sim = self.collab_recommender.get_item_similarity_vector(movie_id)
        collab_indices = [self.collab_recommender.movie_to_idx[mid] for mid in candidates["movieId"]]
        collab_raw = full_collab_sim[collab_indices]

        # 3. Min-max normalization
        content_norm = min_max_normalize(content_raw)
        collab_norm = min_max_normalize(collab_raw)

        # 4. Hybrid combination
        hybrid_scores = w * content_norm + (1.0 - w) * collab_norm

        candidates["content_norm"] = content_norm
        candidates["collab_norm"] = collab_norm
        candidates["hybrid_score"] = hybrid_scores

        # Seed metadata for explanation
        seed_row = self.movies_df.loc[self.movies_df["movieId"] == movie_id].iloc[0]
        seed_title = seed_row["clean_title"]
        seed_genres = seed_row["genre_list"]
        seed_year = seed_row["year"]

        # Deterministic tie-breaking:
        # primary_score_col = "hybrid_score"
        sorted_candidates = break_ties_sort(
            candidates,
            query_year=int(seed_year) if pd.notna(seed_year) else None,
            primary_score_col="hybrid_score",
            ascending=False
        )

        top_candidates = sorted_candidates.head(n).copy()

        # Generate "why recommended" explanations
        explanations = [
            generate_explanation(
                seed_title=seed_title,
                seed_genres=seed_genres,
                rec_title=row["clean_title"],
                rec_genres=row["genre_list"],
                content_norm=row["content_norm"],
                collab_norm=row["collab_norm"]
            )
            for _, row in top_candidates.iterrows()
        ]
        top_candidates["why_recommended"] = explanations

        output_cols = [
            "movieId",
            "title",
            "clean_title",
            "genres",
            "year",
            "content_norm",
            "collab_norm",
            "hybrid_score",
            "why_recommended",
            "avg_rating",
            "n_ratings",
            "bayesian_rating"
        ]
        available_cols = [c for c in output_cols if c in top_candidates.columns]
        return top_candidates[available_cols].reset_index(drop=True)

    def recommend_for_profile(
        self,
        rated_pairs: List[Tuple[int, float]],
        w: float = DEFAULT_HYBRID_WEIGHT,
        n: int = 10,
        selected_genres: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Builds a personalized taste profile from user-rated movies and generates
        hybrid recommendations.
        - Content profile: rating-weighted average genre vector
        - Collab profile: rating-weighted aggregation of item-item similarity vectors
        - Excludes already-rated movies
        """
        if not rated_pairs:
            raise ValueError("No rated movies provided for profile.")

        rated_ids = {mid for mid, _ in rated_pairs}
        candidates = self.movies_df[~self.movies_df["movieId"].isin(rated_ids)].copy()

        if selected_genres:
            candidates = candidates[candidates["genre_list"].apply(
                lambda glist: any(g in glist for g in selected_genres) if glist else False
            )]

        if candidates.empty:
            return pd.DataFrame()

        # 1. Build content profile vector
        # Weights: (rating - 2.5) so that low ratings do not pull genres positively
        total_weight = 0.0
        n_genres = self.content_recommender.genre_matrix.shape[1]
        profile_vec = np.zeros((1, n_genres), dtype=np.float64)

        for mid, r in rated_pairs:
            if mid in self.content_recommender.movie_to_idx:
                idx = self.content_recommender.movie_to_idx[mid]
                weight = max(0.1, float(r))
                profile_vec += weight * self.content_recommender.genre_matrix[idx].toarray()
                total_weight += weight

        if total_weight > 0:
            profile_vec /= total_weight

        candidate_indices = [self.content_recommender.movie_to_idx[mid] for mid in candidates["movieId"]]
        candidate_genre_matrix = self.content_recommender.genre_matrix[candidate_indices]
        content_raw = cosine_similarity(profile_vec, candidate_genre_matrix).flatten()

        # 2. Build collaborative profile vector
        n_items = len(self.collab_recommender.movie_to_idx)
        collab_sum = np.zeros(n_items, dtype=np.float64)
        collab_weight = 0.0

        for mid, r in rated_pairs:
            if mid in self.collab_recommender.movie_to_idx:
                weight = float(r)
                sim_vec = self.collab_recommender.get_item_similarity_vector(mid)
                collab_sum += weight * sim_vec
                collab_weight += weight

        if collab_weight > 0:
            collab_sum /= collab_weight

        collab_indices = [self.collab_recommender.movie_to_idx[mid] for mid in candidates["movieId"]]
        collab_raw = collab_sum[collab_indices]

        # 3. Min-max normalize
        content_norm = min_max_normalize(content_raw)
        collab_norm = min_max_normalize(collab_raw)

        # 4. Hybrid score
        hybrid_scores = w * content_norm + (1.0 - w) * collab_norm

        candidates["content_norm"] = content_norm
        candidates["collab_norm"] = collab_norm
        candidates["hybrid_score"] = hybrid_scores

        # Top genres from rated movies for explanation
        user_top_genres = []
        for mid, _ in rated_pairs:
            row = self.movies_df.loc[self.movies_df["movieId"] == mid]
            if not row.empty:
                user_top_genres.extend(row.iloc[0]["genre_list"])
        top_genres_unique = list(dict.fromkeys(user_top_genres))[:3]
        top_genre_str = ", ".join(top_genres_unique) if top_genres_unique else "Film"

        sorted_candidates = break_ties_sort(
            candidates,
            query_year=None,
            primary_score_col="hybrid_score",
            ascending=False
        )
        top_candidates = sorted_candidates.head(n).copy()

        top_candidates["why_recommended"] = [
            f"Tailored to your preference for {top_genre_str}; recommended by viewers with matching taste."
            for _ in range(len(top_candidates))
        ]

        output_cols = [
            "movieId",
            "title",
            "clean_title",
            "genres",
            "year",
            "content_norm",
            "collab_norm",
            "hybrid_score",
            "why_recommended",
            "avg_rating",
            "n_ratings",
            "bayesian_rating"
        ]
        available_cols = [c for c in output_cols if c in top_candidates.columns]
        return top_candidates[available_cols].reset_index(drop=True)
