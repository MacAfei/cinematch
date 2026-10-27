"""
Offline Evaluation Engine.
Strict Protocol:
- Per-user train/test holdout (80% train, 20% test, fixed random_state)
- Relevant items: held-out test ratings >= 4.0
- Recommendations built strictly from TRAIN data only (no data leakage)
- Already-rated train movies excluded from recommendation candidate pool
- Metrics at K=10: Precision@10, Recall@10, Normalised Recall@10
- Mean, standard deviation, and 95% Confidence Interval
- Hybrid weight sweep (w = 0.0 to 1.0 in steps of 0.1)
- Outputs saved to outputs/evaluation_summary.csv and outputs/weight_sweep.csv
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    EVAL_TEST_SIZE,
    EVAL_MIN_USER_RATINGS,
    EVAL_RELEVANT_THRESHOLD,
    EVAL_SAMPLE_USERS,
    EVAL_RANDOM_STATE,
    EVAL_TOP_K,
    OUTPUTS_DIR
)
from src.features import compute_movie_statistics, build_genre_matrix, break_ties_sort
from src.content import ContentRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import min_max_normalize


def train_test_split_by_user(
    ratings_df: pd.DataFrame,
    test_size: float = EVAL_TEST_SIZE,
    min_ratings: int = EVAL_MIN_USER_RATINGS,
    random_state: int = EVAL_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs per-user holdout split.
    For each user with at least min_ratings, holds out test_size (20%) of ratings.
    Guarantees no user leakage and deterministic reproducibility.
    """
    rng = np.random.RandomState(random_state)
    user_counts = ratings_df["userId"].value_counts()
    eligible_users = user_counts[user_counts >= min_ratings].index

    train_indices = []
    test_indices = []

    # Group ratings by user
    grouped = ratings_df.groupby("userId", sort=False)

    for uid, group in grouped:
        indices = group.index.to_numpy()
        if uid in eligible_users:
            n_total = len(indices)
            n_test = max(1, int(np.round(n_total * test_size)))
            shuffled = rng.permutation(indices)
            test_idx = shuffled[:n_test]
            train_idx = shuffled[n_test:]
            test_indices.extend(test_idx)
            train_indices.extend(train_idx)
        else:
            # Users with too few ratings kept entirely in train
            train_indices.extend(indices)

    train_df = ratings_df.loc[train_indices].copy().reset_index(drop=True)
    test_df = ratings_df.loc[test_indices].copy().reset_index(drop=True)

    return train_df, test_df


def compute_confidence_interval(series: pd.Series) -> Tuple[float, float, float]:
    """
    Returns (mean, std, 95% CI margin: 1.96 * std / sqrt(n))
    """
    n = len(series)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if n > 1 else 0.0
    ci_margin = float(1.96 * (std / np.sqrt(n))) if n > 1 else 0.0
    return round(mean, 4), round(std, 4), round(ci_margin, 4)


class OfflineEvaluator:
    """
    Runs offline evaluation and parameter sweeps across recommendation algorithms.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        sample_users: int = EVAL_SAMPLE_USERS,
        random_state: int = EVAL_RANDOM_STATE,
        k: int = EVAL_TOP_K
    ):
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()
        self.sample_users = sample_users
        self.random_state = random_state
        self.k = k

        # Step 1: Split data
        self.train_df, self.test_df = train_test_split_by_user(
            self.ratings_df,
            test_size=EVAL_TEST_SIZE,
            min_ratings=EVAL_MIN_USER_RATINGS,
            random_state=self.random_state
        )

        # Step 2: Build models strictly from train_df
        self.train_stats = compute_movie_statistics(self.movies_df, self.train_df)
        self.genre_matrix, self.mlb, self.movie_to_idx, self.idx_to_movie = build_genre_matrix(self.movies_df)

        self.content_rec = ContentRecommender(
            movies_df=self.movies_df,
            genre_matrix=self.genre_matrix,
            movie_to_idx=self.movie_to_idx,
            idx_to_movie=self.idx_to_movie,
            movie_stats=self.train_stats
        )

        self.collab_rec = CollaborativeRecommender(
            movies_df=self.movies_df,
            ratings_df=self.train_df,
            movie_stats=self.train_stats
        )

        # Step 3: Identify test evaluation cohort (users with relevant items)
        rel_test = self.test_df[self.test_df["rating"] >= EVAL_RELEVANT_THRESHOLD]
        users_with_rel = rel_test["userId"].unique()

        rng = np.random.RandomState(self.random_state)
        if len(users_with_rel) > self.sample_users:
            self.eval_users = rng.choice(users_with_rel, size=self.sample_users, replace=False)
        else:
            self.eval_users = users_with_rel

        # Pre-cache user train and test sets
        self.user_train_movies = self.train_df.groupby("userId")["movieId"].apply(set).to_dict()
        self.user_train_ratings = {
            uid: dict(zip(group["movieId"], group["rating"]))
            for uid, group in self.train_df.groupby("userId")
        }
        self.user_test_relevant = rel_test.groupby("userId")["movieId"].apply(set).to_dict()

        # Popularity baseline ranking
        pop_candidates = self.train_stats.sort_values(
            by=["bayesian_rating", "n_ratings", "movieId"],
            ascending=[False, False, True]
        )
        self.popularity_ordered_ids = pop_candidates["movieId"].to_numpy()

    def _recommend_for_user_models(
        self,
        user_id: int,
        weights: List[float]
    ) -> Dict[str, List[int]]:
        """
        Generates top-K recommendations for all models for a single test user.
        Excludes movies the user rated in train_df.
        Returns dict: {model_name: [movie_ids]}
        """
        train_rated = self.user_train_movies.get(user_id, set())
        train_ratings_map = self.user_train_ratings.get(user_id, {})
        n_items = len(self.movies_df)

        # 1. Popularity Baseline
        pop_recs = [mid for mid in self.popularity_ordered_ids if mid not in train_rated][:self.k]

        # 2. Content profile scores
        n_genres = self.genre_matrix.shape[1]
        user_genre_vec = np.zeros((1, n_genres), dtype=np.float64)
        total_w = 0.0
        for mid, r in train_ratings_map.items():
            if mid in self.movie_to_idx:
                w = max(0.1, float(r))
                user_genre_vec += w * self.genre_matrix[self.movie_to_idx[mid]].toarray()
                total_w += w

        if total_w > 0:
            user_genre_vec /= total_w

        content_raw = cosine_similarity(user_genre_vec, self.genre_matrix).flatten()

        # 3. Collaborative profile scores
        collab_sum = np.zeros(n_items, dtype=np.float64)
        collab_w = 0.0
        for mid, r in train_ratings_map.items():
            if mid in self.collab_rec.movie_to_idx:
                w = float(r)
                sim_vec = self.collab_rec.get_item_similarity_vector(mid)
                collab_sum += w * sim_vec
                collab_w += w

        if collab_w > 0:
            collab_sum /= collab_w

        # Mask out already rated train movies
        mask = np.ones(n_items, dtype=bool)
        for mid in train_rated:
            if mid in self.movie_to_idx:
                mask[self.movie_to_idx[mid]] = False

        content_raw_masked = content_raw.copy()
        content_raw_masked[~mask] = -1.0

        collab_raw_masked = collab_sum.copy()
        collab_raw_masked[~mask] = -1.0

        # Model: Content-based top-K
        content_top_idx = np.argsort(-content_raw_masked)[:self.k]
        content_recs = [self.idx_to_movie[idx] for idx in content_top_idx]

        # Model: Collaborative top-K
        collab_top_idx = np.argsort(-collab_raw_masked)[:self.k]
        collab_recs = [self.idx_to_movie[idx] for idx in collab_top_idx]

        # Min-max normalization for hybrid combinations
        valid_indices = np.where(mask)[0]
        c_norm = np.zeros(n_items, dtype=np.float64)
        co_norm = np.zeros(n_items, dtype=np.float64)

        if len(valid_indices) > 0:
            c_norm[valid_indices] = min_max_normalize(content_raw[valid_indices])
            co_norm[valid_indices] = min_max_normalize(collab_sum[valid_indices])

        c_norm[~mask] = -1.0
        co_norm[~mask] = -1.0

        results = {
            "Popularity Baseline": pop_recs,
            "Content-Based": content_recs,
            "Collaborative": collab_recs
        }

        # Compute recommendations for requested hybrid weights
        for w in weights:
            h_score = w * c_norm + (1.0 - w) * co_norm
            h_score[~mask] = -1.0
            h_top_idx = np.argsort(-h_score)[:self.k]
            h_recs = [self.idx_to_movie[idx] for idx in h_top_idx]
            model_key = f"Hybrid (w={w:.1f})"
            results[model_key] = h_recs

        return results

    def run_evaluation(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        save_to_disk: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs comprehensive evaluation on the test cohort:
        - Evaluates Popularity Baseline, Content-Based, Collaborative, Hybrid (default w=0.5)
        - Runs weight sweep for w in [0.0, 1.0] step 0.1
        - Calculates Precision@10, Recall@10, Normalised Recall@10 with std and 95% CI
        - Saves outputs to outputs/evaluation_summary.csv and outputs/weight_sweep.csv (if save_to_disk=True)
        """
        weights = [round(w, 1) for w in np.linspace(0.0, 1.0, 11)]
        model_names = ["Popularity Baseline", "Content-Based", "Collaborative"] + [f"Hybrid (w={w:.1f})" for w in weights]

        user_metrics: Dict[str, Dict[str, List[float]]] = {
            m: {"precision": [], "recall": [], "norm_recall": []}
            for m in model_names
        }

        total_users = len(self.eval_users)
        for i, uid in enumerate(self.eval_users):
            if progress_callback and (i % 10 == 0 or i == total_users - 1):
                progress = (i + 1) / total_users
                progress_callback(progress, f"Evaluating user {i+1} of {total_users}...")

            rel_items = self.user_test_relevant.get(uid, set())
            n_rel = len(rel_items)
            if n_rel == 0:
                continue

            recs_dict = self._recommend_for_user_models(uid, weights=weights)

            for m_name, recs in recs_dict.items():
                hits = len(set(recs).intersection(rel_items))
                prec = hits / float(self.k)
                rec = hits / float(n_rel)
                norm_rec = hits / float(min(self.k, n_rel))

                user_metrics[m_name]["precision"].append(prec)
                user_metrics[m_name]["recall"].append(rec)
                user_metrics[m_name]["norm_recall"].append(norm_rec)

        # Aggregate summary table for primary models
        primary_models = ["Popularity Baseline", "Content-Based", "Collaborative", "Hybrid (w=0.5)"]
        summary_rows = []

        for m_name in primary_models:
            p_mean, p_std, p_ci = compute_confidence_interval(pd.Series(user_metrics[m_name]["precision"]))
            r_mean, r_std, r_ci = compute_confidence_interval(pd.Series(user_metrics[m_name]["recall"]))
            nr_mean, nr_std, nr_ci = compute_confidence_interval(pd.Series(user_metrics[m_name]["norm_recall"]))

            display_name = "Hybrid (w=0.5)" if m_name == "Hybrid (w=0.5)" else m_name
            summary_rows.append({
                "Model": display_name,
                "Precision@10": p_mean,
                "Precision_Std": p_std,
                "Precision_CI95": p_ci,
                "Recall@10": r_mean,
                "Recall_Std": r_std,
                "Recall_CI95": r_ci,
                "Normalized_Recall@10": nr_mean,
                "Normalized_Recall_Std": nr_std,
                "Normalized_Recall_CI95": nr_ci,
                "Test_Users_Count": len(user_metrics[m_name]["precision"])
            })

        summary_df = pd.DataFrame(summary_rows)

        # Weight sweep summary table
        sweep_rows = []
        for w in weights:
            m_key = f"Hybrid (w={w:.1f})"
            p_mean, p_std, p_ci = compute_confidence_interval(pd.Series(user_metrics[m_key]["precision"]))
            r_mean, r_std, r_ci = compute_confidence_interval(pd.Series(user_metrics[m_key]["recall"]))
            nr_mean, nr_std, nr_ci = compute_confidence_interval(pd.Series(user_metrics[m_key]["norm_recall"]))

            sweep_rows.append({
                "Content_Weight_w": w,
                "Collab_Weight": round(1.0 - w, 1),
                "Precision@10": p_mean,
                "Precision_CI95": p_ci,
                "Recall@10": r_mean,
                "Recall_CI95": r_ci,
                "Normalized_Recall@10": nr_mean,
                "Normalized_Recall_CI95": nr_ci
            })

        sweep_df = pd.DataFrame(sweep_rows)

        # Save to outputs directory if requested
        if save_to_disk:
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            summary_path = OUTPUTS_DIR / "evaluation_summary.csv"
            sweep_path = OUTPUTS_DIR / "weight_sweep.csv"

            summary_df.to_csv(summary_path, index=False)
            sweep_df.to_csv(sweep_path, index=False)

        return summary_df, sweep_df


def load_evaluation_results() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads saved evaluation summary and weight sweep CSVs if they exist.
    """
    summary_path = OUTPUTS_DIR / "evaluation_summary.csv"
    sweep_path = OUTPUTS_DIR / "weight_sweep.csv"

    summary_df = pd.read_csv(summary_path) if summary_path.exists() else None
    sweep_df = pd.read_csv(sweep_path) if sweep_path.exists() else None

    return summary_df, sweep_df


def load_r_evaluation_results() -> Optional[pd.DataFrame]:
    """
    Loads R pipeline model evaluation results from outputs/model_evaluation.csv
    if present in the workspace.
    """
    r_path = OUTPUTS_DIR / "model_evaluation.csv"
    if r_path.exists():
        try:
            return pd.read_csv(r_path)
        except Exception:
            return None
    return None
