"""
Configuration settings for the Movie Recommendation System.
Centralizes all paths, dataset configurations, algorithm hyperparameters, and evaluation settings.
"""

from pathlib import Path
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
# Default is 'ml-latest-small'. Can be switched to 'ml-25m' or 'ml-32m' for large scale.
DATASET_NAME = os.environ.get("MOVIELENS_DATASET", "ml-latest-small")
GROUPLENS_BASE_URL = "https://files.grouplens.org/datasets/movielens/"
DATASET_URL = f"{GROUPLENS_BASE_URL}{DATASET_NAME}.zip"
DATASET_DIR = DATA_DIR / DATASET_NAME

# Filtering thresholds for scaling to large datasets (e.g. ml-25m)
# For ml-latest-small, min ratings can be 0 or small.
MIN_RATINGS_PER_MOVIE = int(os.environ.get("MIN_RATINGS_PER_MOVIE", 0))
MIN_RATINGS_PER_USER = int(os.environ.get("MIN_RATINGS_PER_USER", 0))

# Recommendation Algorithm Parameters
# Content-based
CONTENT_MIN_YEAR = 1900
CONTENT_MAX_YEAR = 2030

# Collaborative Filtering (Item-based)
# Minimum co-raters required between two items to consider their similarity
MIN_CO_RATERS = int(os.environ.get("MIN_CO_RATERS", 5))
# Shrinkage factor to regularize similarity for item pairs with low support
SHRINKAGE_FACTOR = float(os.environ.get("SHRINKAGE_FACTOR", 10.0))

# Hybrid Recommender
DEFAULT_HYBRID_WEIGHT = 0.5  # Weight w for content-based score (1-w for collaborative)

# Cold-start thresholds
COLD_START_MIN_RATINGS = 5   # Users with < 5 ratings get content-heavy cold start
COLD_START_DEFAULT_N = 10
BAYESIAN_M_QUANTILE = 0.25   # Minimum ratings quantile for Bayesian average calculation

# Clustering
KMEANS_K = 3
KMEANS_N_INIT = 25
KMEANS_RANDOM_STATE = 42

# Offline Evaluation Parameters
EVAL_TEST_SIZE = 0.2
EVAL_MIN_USER_RATINGS = 5
EVAL_RELEVANT_THRESHOLD = 4.0
EVAL_SAMPLE_USERS = int(os.environ.get("EVAL_SAMPLE_USERS", 300))
EVAL_RANDOM_STATE = 42
EVAL_TOP_K = 10

# External APIs
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Project Metadata
PROJECT_TITLE = "Movie Recommendation System"
INSTITUTION = "Ilahia College of Engineering and Technology"
DEPARTMENT = "Dept. of Artificial Intelligence & Data Science"
TEAM_MEMBERS = [
    "Farsana Kabeer",
    "Afeefa Parvin Ashruf",
    "Muhammed Safwan",
    "Muhammed Anees MI",
    "Sharhabeel Mohammed Shameer K"
]
DATASET_CITATION = "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015)."
