"""
Unit tests for Search & Filtering Engine in CineMatch.
Verifies:
- All filters applied together (query + genres + year + min rating)
- Match ANY vs Match ALL genre logic
- All 5 sort orders (Best match, Highest rated, Most rated, Newest, Oldest)
- Empty state detection when filters produce zero matches
- RapidFuzz typo tolerance and match scores
- Reset/clear state consistency
"""

import pytest
import pandas as pd
import numpy as np

from src.ui_components import filter_and_sort_movies, fuzzy_search_movie


@pytest.fixture
def search_catalog():
    movies_df = pd.DataFrame([
        {
            "movieId": 1,
            "title": "The Matrix (1999)",
            "clean_title": "The Matrix",
            "year": 1999,
            "genres": "Action|Sci-Fi",
            "genre_list": ["Action", "Sci-Fi"],
            "avg_rating": 4.5,
            "n_ratings": 280,
            "bayesian_rating": 4.45
        },
        {
            "movieId": 2,
            "title": "The Matrix Reloaded (2003)",
            "clean_title": "The Matrix Reloaded",
            "year": 2003,
            "genres": "Action|Sci-Fi",
            "genre_list": ["Action", "Sci-Fi"],
            "avg_rating": 3.8,
            "n_ratings": 150,
            "bayesian_rating": 3.75
        },
        {
            "movieId": 3,
            "title": "Toy Story (1995)",
            "clean_title": "Toy Story",
            "year": 1995,
            "genres": "Adventure|Animation|Children",
            "genre_list": ["Adventure", "Animation", "Children"],
            "avg_rating": 4.2,
            "n_ratings": 215,
            "bayesian_rating": 4.18
        },
        {
            "movieId": 4,
            "title": "Pulp Fiction (1994)",
            "clean_title": "Pulp Fiction",
            "year": 1994,
            "genres": "Comedy|Crime|Drama",
            "genre_list": ["Comedy", "Crime", "Drama"],
            "avg_rating": 4.8,
            "n_ratings": 310,
            "bayesian_rating": 4.75
        },
        {
            "movieId": 5,
            "title": "Interstellar (2014)",
            "clean_title": "Interstellar",
            "year": 2014,
            "genres": "Drama|Sci-Fi",
            "genre_list": ["Drama", "Sci-Fi"],
            "avg_rating": 4.4,
            "n_ratings": 220,
            "bayesian_rating": 4.35
        },
        {
            "movieId": 6,
            "title": "Inception (2010)",
            "clean_title": "Inception",
            "year": 2010,
            "genres": "Action|Drama|Sci-Fi",
            "genre_list": ["Action", "Drama", "Sci-Fi"],
            "avg_rating": 4.6,
            "n_ratings": 300,
            "bayesian_rating": 4.58
        }
    ])
    return movies_df


def test_filters_applied_together(search_catalog):
    # Search for Sci-Fi movies released between 1998 and 2005 with rating >= 4.0
    results = filter_and_sort_movies(
        search_catalog,
        query="matrix",
        selected_genres=["Sci-Fi"],
        genre_match_mode="Match ANY",
        year_range=(1998, 2005),
        min_rating=4.0
    )
    # The Matrix (1999, 4.5) passes; Reloaded (3.8) fails min_rating
    assert len(results) == 1
    assert results.iloc[0]["movieId"] == 1
    assert results.iloc[0]["clean_title"] == "The Matrix"


def test_genre_match_any_vs_all(search_catalog):
    # Movie 5 is Drama|Sci-Fi, Movie 6 is Action|Drama|Sci-Fi, Movie 1 is Action|Sci-Fi
    # Under Match ALL for ["Action", "Sci-Fi", "Drama"], only Movie 6 matches
    all_res = filter_and_sort_movies(
        search_catalog,
        selected_genres=["Action", "Sci-Fi", "Drama"],
        genre_match_mode="Match ALL"
    )
    assert len(all_res) == 1
    assert all_res.iloc[0]["movieId"] == 6  # Inception

    # Under Match ANY for ["Animation", "Crime"], Toy Story (Animation) and Pulp Fiction (Crime) match
    any_res = filter_and_sort_movies(
        search_catalog,
        selected_genres=["Animation", "Crime"],
        genre_match_mode="Match ANY"
    )
    assert len(any_res) == 2
    matched_ids = set(any_res["movieId"])
    assert matched_ids == {3, 4}


def test_sort_orders(search_catalog):
    # Highest rated
    hi_res = filter_and_sort_movies(search_catalog, sort_by="Highest rated")
    assert hi_res.iloc[0]["movieId"] == 4  # Pulp Fiction (4.8)

    # Most rated
    mr_res = filter_and_sort_movies(search_catalog, sort_by="Most rated")
    assert mr_res.iloc[0]["movieId"] == 4  # Pulp Fiction (310 ratings)
    assert mr_res.iloc[1]["movieId"] == 6  # Inception (300 ratings)

    # Newest
    new_res = filter_and_sort_movies(search_catalog, sort_by="Newest")
    assert new_res.iloc[0]["movieId"] == 5  # Interstellar (2014)

    # Oldest
    old_res = filter_and_sort_movies(search_catalog, sort_by="Oldest")
    assert old_res.iloc[0]["movieId"] == 4  # Pulp Fiction (1994)


def test_empty_state_detection(search_catalog):
    # Extreme filter that matches nothing: rating >= 5.0 and year <= 1990
    empty_res = filter_and_sort_movies(
        search_catalog,
        year_range=(1980, 1990),
        min_rating=4.9
    )
    assert empty_res.empty


def test_fuzzy_search_ranking_and_typos(search_catalog):
    # Typo: 'matrx'
    res = filter_and_sort_movies(search_catalog, query="matrx", sort_by="Best match")
    assert not res.empty
    assert "Matrix" in res.iloc[0]["clean_title"]


def test_clear_resets_catalog(search_catalog):
    # Passing empty/default arguments returns full catalog
    default_res = filter_and_sort_movies(search_catalog)
    assert len(default_res) == len(search_catalog)
