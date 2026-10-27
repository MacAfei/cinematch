"""
Feature engineering module for Movie Recommendation System.
Handles:
- Genre binary matrix construction (MultiLabelBinarizer, sparse representation)
- Bayesian / weighted average rating computation (deterministic tie-breaking)
- Index mapping helpers (movieId <-> matrix index)
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import BAYESIAN_M_QUANTILE


def compute_movie_statistics(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    m_override: Optional[float] = None
) -> pd.DataFrame:
    """
    Computes per-movie rating statistics and the Bayesian weighted average rating (IMDb formula).
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        v = number of ratings for the movie
        m = minimum rating threshold (default: 25th percentile of movie counts or 5, whichever is greater)
        R = average rating of the movie
        C = mean rating across all movies in the dataset
    """
    stats = ratings_df.groupby("movieId").agg(
        n_ratings=("rating", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()

    # Merge with full movies dataframe so every movie has a record
    full_stats = movies_df[["movieId", "title", "year"]].merge(stats, on="movieId", how="left")
    full_stats["n_ratings"] = full_stats["n_ratings"].fillna(0).astype(int)
    
    # Global mean rating across all ratings
    global_mean_c = ratings_df["rating"].mean() if len(ratings_df) > 0 else 3.0
    full_stats["avg_rating"] = full_stats["avg_rating"].fillna(global_mean_c)

    # Determine m threshold
    if m_override is not None:
        m = float(m_override)
    else:
        # 25th percentile of rated movies, bounded at minimum 3
        rated_counts = full_stats.loc[full_stats["n_ratings"] > 0, "n_ratings"]
        if len(rated_counts) > 0:
            m = max(3.0, float(np.quantile(rated_counts, BAYESIAN_M_QUANTILE)))
        else:
            m = 5.0

    v = full_stats["n_ratings"]
    r = full_stats["avg_rating"]
    full_stats["bayesian_rating"] = (v / (v + m)) * r + (m / (v + m)) * global_mean_c

    return full_stats


def build_genre_matrix(
    movies_df: pd.DataFrame
) -> Tuple[sparse.csr_matrix, MultiLabelBinarizer, Dict[int, int], Dict[int, int]]:
    """
    Builds a sparse binary genre matrix using MultiLabelBinarizer.
    Returns:
        - genre_matrix: scipy.sparse.csr_matrix of shape (n_movies, n_genres)
        - mlb: MultiLabelBinarizer fitted object
        - movie_to_idx: dictionary mapping movieId -> row index in matrix
        - idx_to_movie: dictionary mapping row index -> movieId
    """
    mlb = MultiLabelBinarizer(sparse_output=True)
    # Ensure genre_list exists and contains lists of genres
    genre_lists = movies_df["genre_list"].tolist()
    genre_matrix = mlb.fit_transform(genre_lists).tocsr()

    movie_ids = movies_df["movieId"].tolist()
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    idx_to_movie = {idx: mid for idx, mid in enumerate(movie_ids)}

    return genre_matrix, mlb, movie_to_idx, idx_to_movie


def build_enriched_content_matrix(
    movies_df: pd.DataFrame,
    tags_df: Optional[pd.DataFrame] = None
) -> Tuple[sparse.csr_matrix, Dict[int, int], Dict[int, int]]:
    """
    Builds an enriched multimodal content feature matrix combining:
    1. Multi-hot binary genres (weight: 1.0)
    2. User tags TF-IDF (weight: 0.8)
    3. Title words TF-IDF (weight: 0.4)
    4. Release decade one-hot (weight: 0.3)
    Rows are L2-normalized for cosine similarity.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.preprocessing import normalize

    movie_ids = movies_df["movieId"].tolist()
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    idx_to_movie = {idx: mid for idx, mid in enumerate(movie_ids)}

    # 1. Genres binary matrix (w = 1.0)
    mlb = MultiLabelBinarizer(sparse_output=True)
    genre_mat = mlb.fit_transform(movies_df["genre_list"].tolist()).tocsr().astype(np.float32)

    # 2. Tags TF-IDF (w = 0.8)
    if tags_df is not None and not tags_df.empty:
        tag_series = tags_df.groupby("movieId")["tag"].apply(
            lambda t_list: " ".join(str(t).lower().replace("-", " ") for t in t_list)
        ).to_dict()
        movie_tags_text = [tag_series.get(mid, "") for mid in movie_ids]
        tag_vec = TfidfVectorizer(max_features=1200, stop_words="english", min_df=1, ngram_range=(1, 2))
        tag_mat = tag_vec.fit_transform(movie_tags_text).tocsr().astype(np.float32) * 0.8
    else:
        tag_mat = sparse.csr_matrix((len(movie_ids), 0), dtype=np.float32)

    # 3. Title keywords TF-IDF (w = 0.4)
    title_texts = movies_df["clean_title"].fillna("").astype(str).tolist()
    title_vec = TfidfVectorizer(max_features=600, stop_words="english", min_df=1)
    title_mat = title_vec.fit_transform(title_texts).tocsr().astype(np.float32) * 0.4

    # 4. Release decade (w = 0.3)
    def to_decade(y):
        if pd.isna(y) or y < 1900 or y > 2035:
            return "decade_unknown"
        return f"decade_{int(y // 10 * 10)}s"

    decades = [to_decade(y) for y in movies_df["year"]]
    decade_vec = CountVectorizer(binary=True)
    decade_mat = decade_vec.fit_transform(decades).tocsr().astype(np.float32) * 0.3

    # Combine blocks
    blocks = [genre_mat, title_mat, decade_mat]
    if tag_mat.shape[1] > 0:
        blocks.append(tag_mat)

    combined = sparse.hstack(blocks, format="csr")
    # L2 normalize rows
    combined_norm = normalize(combined, norm="l2", axis=1)

    return combined_norm, movie_to_idx, idx_to_movie


def apply_diversity_rerank(
    candidates_df: pd.DataFrame,
    feature_matrix: sparse.csr_matrix,
    movie_to_idx: Dict[int, int],
    primary_score_col: str = "score",
    n: int = 10,
    diversity_lambda: float = 0.75,
    popularity_penalty: float = 0.04
) -> pd.DataFrame:
    """
    Applies Maximal Marginal Relevance (MMR) and popularity-bias damping
    to prevent repetitive franchise dominance while preserving relevance.
    Guarantees no duplicate movies are selected.
    """
    if len(candidates_df) <= n or primary_score_col not in candidates_df.columns:
        return candidates_df.head(n).reset_index(drop=True)

    candidates = candidates_df.copy().reset_index(drop=True)
    n_candidates = len(candidates)

    scores = candidates[primary_score_col].to_numpy(dtype=np.float64)
    s_min, s_max = np.nanmin(scores), np.nanmax(scores)
    norm_scores = (scores - s_min) / (s_max - s_min + 1e-9)

    # Popularity penalty
    n_ratings = candidates.get("n_ratings", pd.Series(np.zeros(n_candidates))).to_numpy(dtype=np.float64)
    max_ratings = np.nanmax(n_ratings) if len(n_ratings) > 0 and np.nanmax(n_ratings) > 0 else 1.0
    pop_factors = np.log1p(n_ratings) / np.log1p(max_ratings)

    # Get feature vectors for candidates
    cand_indices = [movie_to_idx.get(mid, 0) for mid in candidates["movieId"]]
    cand_vectors = feature_matrix[cand_indices]

    selected_indices: List[int] = []
    unselected = set(range(n_candidates))

    # Pick top scoring candidate first
    first_pick = int(np.argmax(norm_scores))
    selected_indices.append(first_pick)
    unselected.remove(first_pick)

    # Precompute pairwise similarities among candidates on-the-fly or incrementally
    while len(selected_indices) < n and unselected:
        best_idx = -1
        best_mmr = -1e9

        # Similarity to selected set
        selected_vecs = cand_vectors[selected_indices]

        for u in unselected:
            u_vec = cand_vectors[u]
            # Max similarity to already selected items
            sims = (selected_vecs @ u_vec.T).toarray().flatten()
            max_sim = np.max(sims) if len(sims) > 0 else 0.0

            # MMR formula
            mmr_val = (
                diversity_lambda * norm_scores[u]
                - (1.0 - diversity_lambda) * max_sim
                - popularity_penalty * pop_factors[u]
            )

            if mmr_val > best_mmr:
                best_mmr = mmr_val
                best_idx = u

        if best_idx != -1:
            selected_indices.append(best_idx)
            unselected.remove(best_idx)
        else:
            break

    return candidates.iloc[selected_indices].reset_index(drop=True)


def break_ties_sort(
    candidates_df: pd.DataFrame,
    query_year: Optional[int] = None,
    primary_score_col: str = "score",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Deterministically sorts recommendation candidates:
    1. primary_score_col (descending by default)
    2. bayesian_rating (descending)
    3. n_ratings (descending)
    4. year_diff (ascending, closest release year to query movie)
    5. movieId (ascending, guarantees 100% stable deterministic order)
    """
    df = candidates_df.copy()

    # Calculate year distance if query_year is provided
    if query_year is not None and "year" in df.columns:
        df["year_diff"] = df["year"].apply(
            lambda y: abs(y - query_year) if pd.notna(y) else 9999
        )
    else:
        df["year_diff"] = 0

    sort_cols = [primary_score_col, "bayesian_rating", "n_ratings", "year_diff", "movieId"]
    sort_ascending = [ascending, False, False, True, True]

    # Filter to available columns
    available_sort_cols = [c for c in sort_cols if c in df.columns]
    available_ascending = [sort_ascending[sort_cols.index(c)] for c in available_sort_cols]

    sorted_df = df.sort_values(by=available_sort_cols, ascending=available_ascending)
    if "year_diff" in sorted_df.columns:
        sorted_df = sorted_df.drop(columns=["year_diff"])

    return sorted_df.reset_index(drop=True)
