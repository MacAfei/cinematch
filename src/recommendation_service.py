"""
Recommendation Service Layer for CineMatch.
Provides Netflix-style horizontal row generation, adaptive hybrid weighting,
dynamic profile vector synthesis from local SQLite ratings, and diversity re-ranking.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DEFAULT_HYBRID_WEIGHT
from src.features import (
    break_ties_sort,
    apply_diversity_rerank,
    build_enriched_content_matrix
)
from src.hybrid import min_max_normalize
from src.db import (
    get_user_ratings,
    get_watchlist,
    get_recently_viewed,
    get_profile
)


def get_adaptive_content_weight(num_ratings: int, base_w: float = 0.2) -> float:
    """
    Dynamically scales content weight:
    - 0 ratings: 1.0 (pure cold start / popularity)
    - 1-2 ratings: 0.80 (content-heavy)
    - 3-4 ratings: 0.60
    - 5-15 ratings: 0.35
    - 16+ ratings: base_w (optimal peak at 0.2)
    """
    if num_ratings <= 0:
        return 1.0
    elif num_ratings <= 2:
        return 0.80
    elif num_ratings <= 4:
        return 0.60
    elif num_ratings <= 15:
        return 0.35
    return base_w


def recommend_for_active_profile(
    profile_id: int,
    engine: Dict[str, Any],
    n: int = 12,
    selected_genres: Optional[List[str]] = None,
    apply_diversity: bool = True
) -> pd.DataFrame:
    """
    Synthesizes recommendations for any active profile (new local user or MovieLens user).
    Builds profile vector from their SQLite ratings, scores unrated items via
    item-item collaborative similarity and multimodal content features,
    and excludes movies already rated or on the watchlist.
    """
    movies_df = engine["movies_df"]
    collab_rec = engine["collab_recommender"]
    content_rec = engine["content_recommender"]
    enriched_matrix = engine.get("enriched_matrix", content_rec.genre_matrix)

    profile = get_profile(profile_id)
    user_ratings_map = get_user_ratings(profile_id)

    # If it's a seeded MovieLens user and local SQLite has no new ratings, use their MovieLens ratings
    if not user_ratings_map and profile and profile.get("is_movielens") and profile.get("movielens_user_id"):
        ml_uid = int(profile["movielens_user_id"])
        ml_ratings_df = engine["ratings_df"][engine["ratings_df"]["userId"] == ml_uid]
        user_ratings_map = dict(zip(ml_ratings_df["movieId"], ml_ratings_df["rating"]))

    # Movies to exclude
    already_rated = set(user_ratings_map.keys())
    watchlist = set(get_watchlist(profile_id))
    exclude_ids = already_rated.union(watchlist)

    num_ratings = len(user_ratings_map)
    adaptive_w = get_adaptive_content_weight(num_ratings, base_w=0.2)

    # Favorite genres fallback if 0 ratings
    fav_genres = []
    if profile and profile.get("fav_genres"):
        fav_genres = [g.strip() for g in profile["fav_genres"].split(",") if g.strip()]
    if selected_genres:
        fav_genres.extend(selected_genres)
    fav_genres = list(dict.fromkeys(fav_genres))

    # Candidate pool
    candidates = movies_df[~movies_df["movieId"].isin(exclude_ids)].copy()
    if fav_genres and num_ratings == 0:
        candidates_fav = candidates[candidates["genre_list"].apply(
            lambda glist: any(g in glist for g in fav_genres) if glist else False
        )]
        if not candidates_fav.empty:
            candidates = candidates_fav

    if candidates.empty:
        return pd.DataFrame()

    n_items = len(collab_rec.movie_to_idx)
    candidate_mids = candidates["movieId"].tolist()
    candidate_c_indices = [content_rec.movie_to_idx.get(mid, 0) for mid in candidate_mids]
    candidate_co_indices = [collab_rec.movie_to_idx.get(mid, 0) for mid in candidate_mids]

    if num_ratings == 0:
        # Zero ratings: Bayesian-weighted popularity
        candidates["rec_score"] = candidates["bayesian_rating"]
        sorted_cand = break_ties_sort(candidates, primary_score_col="rec_score", ascending=False)
        top = sorted_cand.head(n).copy()
        top["why_recommended"] = "Critically acclaimed essential based on universal community ratings."
        return top.reset_index(drop=True)

    # 1. Content profile vector
    n_features = enriched_matrix.shape[1]
    profile_content_vec = np.zeros((1, n_features), dtype=np.float64)
    content_w_sum = 0.0

    for mid, r in user_ratings_map.items():
        if mid in content_rec.movie_to_idx:
            c_idx = content_rec.movie_to_idx[mid]
            weight = max(0.1, float(r))
            profile_content_vec += weight * enriched_matrix[c_idx].toarray()
            content_w_sum += weight

    if content_w_sum > 0:
        profile_content_vec /= content_w_sum

    content_sim_raw = cosine_similarity(profile_content_vec, enriched_matrix[candidate_c_indices]).flatten()

    # 2. Collaborative profile vector
    collab_sum = np.zeros(n_items, dtype=np.float64)
    collab_w_sum = 0.0

    for mid, r in user_ratings_map.items():
        if mid in collab_rec.movie_to_idx:
            weight = float(r)
            sim_vec = collab_rec.get_item_similarity_vector(mid)
            collab_sum += weight * sim_vec
            collab_w_sum += weight

    if collab_w_sum > 0:
        collab_sum /= collab_w_sum

    collab_sim_raw = collab_sum[candidate_co_indices]

    # 3. Min-max normalization
    c_norm = min_max_normalize(content_sim_raw)
    co_norm = min_max_normalize(collab_sim_raw)

    hybrid_scores = adaptive_w * c_norm + (1.0 - adaptive_w) * co_norm
    candidates["rec_score"] = hybrid_scores
    candidates["content_norm"] = c_norm
    candidates["collab_norm"] = co_norm

    sorted_candidates = break_ties_sort(candidates, primary_score_col="rec_score", ascending=False)

    if apply_diversity:
        top_candidates = apply_diversity_rerank(
            candidates_df=sorted_candidates,
            feature_matrix=enriched_matrix,
            movie_to_idx=content_rec.movie_to_idx,
            primary_score_col="rec_score",
            n=n,
            diversity_lambda=0.75,
            popularity_penalty=0.03
        )
    else:
        top_candidates = sorted_candidates.head(n).copy()

    # Why explanation
    explanations = []
    for _, row in top_candidates.iterrows():
        g_list = row["genre_list"][:2]
        g_str = ", ".join(g_list) if g_list else "Cinema"
        if row["collab_norm"] >= 0.5:
            why = f"Matches your taste profile in {g_str}; highly co-rated by viewers like you."
        else:
            why = f"Strong thematic alignment with your top preferences in {g_str}."
        explanations.append(why)
    top_candidates["why_recommended"] = explanations

    return top_candidates.reset_index(drop=True)


def get_home_rows(
    profile_id: int,
    engine: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Constructs the Netflix-style rows for the home page:
    1. Top picks for you (personalized hybrid)
    2. Because you liked <movie rated 4+> (or popular classic if none)
    3. More like <recently viewed movie>
    4. Popular right now (high Bayesian rating + rating count)
    5. Top in <favorite genre 1>
    6. Hidden gems (high rating, lower rating count)
    7. Continue from My List
    """
    movies_df = engine["movies_df"]
    collab_rec = engine["collab_recommender"]
    user_ratings = get_user_ratings(profile_id)
    profile = get_profile(profile_id)
    watchlist_ids = get_watchlist(profile_id)
    recent_views = get_recently_viewed(profile_id, limit=3)

    rows = []

    # 1. Continue from My List (if any)
    if watchlist_ids:
        wl_movies = movies_df[movies_df["movieId"].isin(watchlist_ids)].head(12)
        if not wl_movies.empty:
            rows.append({
                "title": "📌 Continue from My List",
                "tooltip": "Movies you saved to watch later",
                "movies": wl_movies.to_dict(orient="records")
            })

    # 2. Top picks for you
    top_picks = recommend_for_active_profile(profile_id, engine, n=12)
    if not top_picks.empty:
        rows.append({
            "title": "✨ Top Picks For You",
            "tooltip": "Personalized hybrid recommendations adapted to your ratings and taste profile",
            "movies": top_picks.to_dict(orient="records")
        })

    # 3. Because you liked <X>
    high_rated = [mid for mid, r in user_ratings.items() if r >= 4.0]
    seed_movie_id = None
    seed_title = "Toy Story"
    if high_rated:
        seed_movie_id = high_rated[-1]
        m_row = movies_df.loc[movies_df["movieId"] == seed_movie_id]
        if not m_row.empty:
            seed_title = m_row.iloc[0]["clean_title"]
    else:
        # Default seed
        seed_movie_id = 1

    try:
        because_liked = collab_rec.recommend_collab(movie_id=seed_movie_id, n=12)
        # Exclude rated
        because_liked = because_liked[~because_liked["movieId"].isin(set(user_ratings.keys()))].head(10)
        if not because_liked.empty:
            rows.append({
                "title": f"🎬 Because you liked {seed_title}",
                "tooltip": f"Item-based collaborative recommendations derived from audience ratings for {seed_title}",
                "movies": because_liked.to_dict(orient="records")
            })
    except Exception:
        pass

    # 4. More like <recently viewed>
    if recent_views and recent_views[0] != seed_movie_id:
        view_id = recent_views[0]
        v_row = movies_df.loc[movies_df["movieId"] == view_id]
        if not v_row.empty:
            v_title = v_row.iloc[0]["clean_title"]
            try:
                more_like = collab_rec.recommend_collab(movie_id=view_id, n=10)
                more_like = more_like[~more_like["movieId"].isin(set(user_ratings.keys()))].head(10)
                if not more_like.empty:
                    rows.append({
                        "title": f"👀 More Like {v_title}",
                        "tooltip": f"Movies with high audience overlap with {v_title}",
                        "movies": more_like.to_dict(orient="records")
                    })
            except Exception:
                pass

    # 5. Popular Right Now
    pop_candidates = movies_df.copy()
    if user_ratings:
        pop_candidates = pop_candidates[~pop_candidates["movieId"].isin(set(user_ratings.keys()))]
    pop_df = break_ties_sort(pop_candidates, primary_score_col="bayesian_rating", ascending=False).head(12)
    pop_df["why_recommended"] = "Consistently high community ratings and widespread acclaim."
    rows.append({
        "title": "🔥 Popular Right Now",
        "tooltip": "Highest Bayesian-weighted community ratings across the entire MovieLens catalog",
        "movies": pop_df.to_dict(orient="records")
    })

    # 6. Top in favorite genre
    user_fav_genres = []
    if profile and profile.get("fav_genres"):
        user_fav_genres = [g.strip() for g in profile["fav_genres"].split(",") if g.strip()]
    if not user_fav_genres:
        user_fav_genres = ["Sci-Fi", "Animation", "Drama"]

    for g in user_fav_genres[:2]:
        g_candidates = movies_df[movies_df["genre_list"].apply(lambda gl: g in gl if gl else False)].copy()
        if user_ratings:
            g_candidates = g_candidates[~g_candidates["movieId"].isin(set(user_ratings.keys()))]
        g_sorted = break_ties_sort(g_candidates, primary_score_col="bayesian_rating", ascending=False).head(10)
        if not g_sorted.empty:
            rows.append({
                "title": f"🍿 Best in {g}",
                "tooltip": f"Top Bayesian-rated films within the {g} genre",
                "movies": g_sorted.to_dict(orient="records")
            })

    # 7. Hidden Gems (high rating, lower count: 15 to 70 ratings)
    gems = movies_df[(movies_df["n_ratings"] >= 15) & (movies_df["n_ratings"] <= 70) & (movies_df["avg_rating"] >= 3.85)].copy()
    if user_ratings:
        gems = gems[~gems["movieId"].isin(set(user_ratings.keys()))]
    gems_sorted = break_ties_sort(gems, primary_score_col="avg_rating", ascending=False).head(10)
    if not gems_sorted.empty:
        gems_sorted["why_recommended"] = "Undiscovered masterpiece with exceptional ratings and a devoted following."
        rows.append({
            "title": "💎 Hidden Gems",
            "tooltip": "Critically loved films with smaller audience sizes that deserve more attention",
            "movies": gems_sorted.to_dict(orient="records")
        })

    return rows
