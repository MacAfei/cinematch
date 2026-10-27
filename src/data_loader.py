"""
Data loading and preprocessing module for MovieLens dataset.
Pure Python, no Streamlit dependencies.
Handles automatic downloading, extraction, caching, and filtering.
"""

import io
import os
import re
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

from src.config import (
    DATA_DIR,
    DATASET_NAME,
    DATASET_URL,
    DATASET_DIR,
    MIN_RATINGS_PER_MOVIE,
    MIN_RATINGS_PER_USER
)


def extract_year_from_title(title: str) -> Optional[int]:
    """
    Extracts 4-digit release year from movie title if present.
    Example: 'Toy Story (1995)' -> 1995.
    Returns None if no year is found.
    """
    if not isinstance(title, str):
        return None
    matches = re.findall(r'\((\d{4})\)', title)
    if matches:
        try:
            year = int(matches[-1])
            if 1880 <= year <= 2035:
                return year
        except ValueError:
            return None
    return None


def clean_movie_title(title: str) -> str:
    """
    Removes the trailing (YYYY) from title for cleaner display and fuzzy search.
    """
    if not isinstance(title, str):
        return ""
    cleaned = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    return cleaned if cleaned else title.strip()


def parse_genres(genres_str: str) -> list[str]:
    """
    Parses pipe-separated genres string into a list of genre names.
    Handles '(no genres listed)' by returning an empty list.
    """
    if not isinstance(genres_str, str) or not genres_str:
        return []
    if genres_str.strip() == "(no genres listed)":
        return []
    return [g.strip() for g in genres_str.split('|') if g.strip() and g.strip() != "(no genres listed)"]


def download_dataset_if_missing(dataset_dir: Path = DATASET_DIR, dataset_url: str = DATASET_URL) -> Path:
    """
    Checks if movies.csv and ratings.csv exist in dataset_dir.
    If not, downloads and extracts the MovieLens archive from dataset_url.
    Raises RuntimeError with clear manual instructions if download fails.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    movies_path = dataset_dir / "movies.csv"
    ratings_path = dataset_dir / "ratings.csv"

    if movies_path.exists() and ratings_path.exists():
        return dataset_dir

    print(f"Downloading dataset from {dataset_url}...")
    try:
        import requests
        try:
            import certifi
            verify_cert = certifi.where()
        except ImportError:
            verify_cert = True

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MovieRecommender/1.0"}
        
        try:
            resp = requests.get(dataset_url, headers=headers, timeout=60, verify=verify_cert)
            resp.raise_for_status()
            zip_content = resp.content
        except requests.exceptions.SSLError:
            # Fallback if enterprise or system cert issue occurs
            print("SSL verification failed, retrying with verify=False...")
            resp = requests.get(dataset_url, headers=headers, timeout=60, verify=False)
            resp.raise_for_status()
            zip_content = resp.content

        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            for member in z.namelist():
                filename = Path(member).name
                if filename in ["movies.csv", "ratings.csv", "links.csv", "tags.csv", "README.txt"]:
                    source = z.open(member)
                    target = open(dataset_dir / filename, "wb")
                    with source, target:
                        target.write(source.read())
        print(f"Dataset successfully downloaded and extracted to {dataset_dir}")
    except Exception as e:
        error_msg = (
            f"Failed to automatically download MovieLens dataset from {dataset_url}.\n"
            f"Error details: {str(e)}\n"
            f"MANUAL DOWNLOAD INSTRUCTIONS:\n"
            f"1. Download the zip archive manually from: {dataset_url}\n"
            f"2. Extract movies.csv, ratings.csv (and optionally links.csv, tags.csv) into:\n"
            f"   {dataset_dir.resolve()}\n"
            f"3. Re-run the application."
        )
        raise RuntimeError(error_msg) from e

    return dataset_dir


def load_raw_data(
    dataset_dir: Optional[Path] = None,
    min_movie_ratings: int = MIN_RATINGS_PER_MOVIE,
    min_user_ratings: int = MIN_RATINGS_PER_USER
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads movies.csv and ratings.csv (and optional links.csv and tags.csv).
    Preprocesses movies (year extraction, genre parsing) and applies configurable
    filtering thresholds to ratings.

    Returns:
        Tuple of (movies_df, ratings_df, links_df, tags_df)
    """
    target_dir = dataset_dir or DATASET_DIR
    download_dataset_if_missing(target_dir)

    movies_path = target_dir / "movies.csv"
    ratings_path = target_dir / "ratings.csv"
    links_path = target_dir / "links.csv"
    tags_path = target_dir / "tags.csv"

    if not movies_path.exists() or not ratings_path.exists():
        raise FileNotFoundError(f"Missing movies.csv or ratings.csv in {target_dir}")

    # Load movies
    movies_df = pd.read_csv(
        movies_path,
        dtype={"movieId": int, "title": str, "genres": str}
    )
    # Feature extraction
    movies_df["year"] = movies_df["title"].apply(extract_year_from_title)
    movies_df["clean_title"] = movies_df["title"].apply(clean_movie_title)
    movies_df["genre_list"] = movies_df["genres"].apply(parse_genres)

    # Load ratings
    ratings_df = pd.read_csv(
        ratings_path,
        dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int}
    )

    # Apply scaling filters if configured (for large datasets like ml-25m)
    if min_movie_ratings > 0:
        movie_counts = ratings_df["movieId"].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movies)]

    if min_user_ratings > 0:
        user_counts = ratings_df["userId"].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        ratings_df = ratings_df[ratings_df["userId"].isin(valid_users)]

    # Load optional links
    links_df = None
    if links_path.exists():
        try:
            links_df = pd.read_csv(links_path, dtype={"movieId": int, "imdbId": str, "tmdbId": str})
        except Exception:
            links_df = None

    # Load optional tags
    tags_df = None
    if tags_path.exists():
        try:
            tags_df = pd.read_csv(tags_path, dtype={"userId": int, "movieId": int, "tag": str, "timestamp": int})
        except Exception:
            tags_df = None

    return movies_df, ratings_df, links_df, tags_df
