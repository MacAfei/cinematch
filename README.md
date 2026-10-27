# 🎬 CineMatch — Premium Movie Recommendation Platform

> **Big Data Analytics Laboratory Deployable Project**  
> **Institution:** Ilahia College of Engineering and Technology, Dept. of Artificial Intelligence & Data Science  
> **Team Members:** Farsana Kabeer, Afeefa Parvin Ashruf, Muhammed Safwan, Muhammed Anees MI, Sharhabeel Mohammed Shameer K  
> **Course:** B.Tech Artificial Intelligence & Data Science (Big Data Analytics Lab)

---

## 📌 Executive Overview

**CineMatch** transforms the traditional lab dashboard into a modern, complete, dark-themed streaming recommendation product inspired by Netflix and Prime Video. Built on the MovieLens benchmark and integrated with TMDb metadata, CineMatch delivers personalized recommendations, instant interactive ratings, watchlist persistence, and an academic lab laboratory embedded directly within the application.

### Key Capabilities
- **Multi-Strategy Recommendation Engine**:
  - **Content-Based Filtering**: Enriched sparse matrix combining multi-hot genres, TF-IDF user tags (`tags.csv`), title tokenization, and release decade bins with cosine similarity and deterministic tie-breaking.
  - **Collaborative Filtering (Item-Based)**: User mean-centered sparse rating matrices (adjusted cosine similarity), co-rater support thresholds ($\ge 5$), and shrinkage regularization ($\lambda = 10$).
  - **Hybrid Engine**: Dynamic min-max normalization fusing content metadata and collaborative patterns with adaptive weighting ($w = 0.2$ default, scaling up to $0.8$ for new users) and natural language explanations.
  - **Maximal Marginal Relevance (MMR)**: Diversity re-ranking to avoid repetitive blockbuster franchises.
- **Full Streaming User Experience**:
  - **Profile Switcher & Onboarding**: Select existing MovieLens user profiles or create local guest profiles stored in a local SQLite database (`db/app.db`). New profiles complete a 5-movie interactive poster onboarding flow.
  - **Hero Banner & Horizontal Rows**: Featured top recommendation backdrop, plus "Top picks for you", "Because you liked...", "More like...", "Popular right now" (Bayesian IMDb rating), "Top in genre", and "Hidden gems".
  - **Movie Detail Modal**: Backdrop, poster, synopsis, director, top 5 cast members, 1–5 star rating with half stars, watchlist toggle, watched status, and 3-strategy recommendation tabs.
  - **Typo-Tolerant Search**: Real-time title fuzzy search powered by RapidFuzz with genre, release year, and minimum rating filters.
  - **Live Feedback Loop**: Instant profile vector updates upon rating movies, immediately updating downstream recommendations.
  - **Unified Project Overview & About Pages**: Complete academic walkthrough (workflow timeline, EDA charts, K-Means clustering, side-by-side comparison, evaluation benchmark) and team credits.

---

## 🗂️ Application Structure & Navigation

The navigation is structured so the streaming product experience takes center stage, with academic insights and team credits logically positioned at the end:

```
movie-recommender-app/
├── app.py                      # Main CineMatch streaming home & hero discovery
├── pages/                      # Multi-page application structure
│   ├── 01_Search.py            # Typo-tolerant fuzzy search & multi-attribute filter grid
│   ├── 02_Movie_Detail.py      # Detailed movie view (synopsis, cast, tabs, ratings)
│   ├── 03_My_List.py           # User watchlist manager with CSV export
│   ├── 04_My_Ratings.py        # Personal rating history manager with CSV export
│   ├── 05_Project_Overview.py  # (Second-Last) 6 interactive academic walkthrough tabs
│   └── 06_About.py             # (Last) Team member cards, institution, TOOLS_USED & citations
├── src/                        # Core Python engine & services
│   ├── __init__.py
│   ├── config.py               # Parameters, thresholds, and paths
│   ├── db.py                   # SQLite local storage (profiles, ratings, watchlist, history)
│   ├── tmdb_client.py          # TMDb API client with disk cache & SVG fallback generator
│   ├── recommendation_service.py # High-level home row generator & live feedback loop
│   ├── team_config.py          # Configurable team credits & verified TOOLS_USED lists
│   ├── data_loader.py          # Auto-downloader with SSL fallback and robust CSV parsers
│   ├── features.py             # Enriched TF-IDF content matrix & MMR diversity re-ranking
│   ├── content.py              # Sparse content-based cosine similarity engine
│   ├── collaborative.py        # Sparse item-item adjusted cosine with shrinkage
│   ├── hybrid.py               # Min-max normalization & transparent explanations
│   ├── cold_start.py           # Bayesian essentials & adaptive profile fallbacks
│   ├── clustering.py           # K-Means (k=3) user activity & score segmentation
│   ├── evaluation.py           # Leak-free offline train/test benchmark protocol
│   ├── app_state.py            # Streamlit caching & SQLite synchronization manager
│   └── ui_components.py        # Streaming CSS, poster cards, chips & dialog components
├── tests/                      # Automated pytest test suite (27 passing tests)
│   ├── test_content.py         # Content recommendations & tie-breaking tests
│   ├── test_collaborative.py   # Collaborative filtering & rated movie exclusion
│   ├── test_hybrid.py          # Score bounds, normalization, and weight verification
│   ├── test_cold_start.py      # Cold-start fallback paths
│   ├── test_clustering.py      # K-Means clustering validation
│   ├── test_evaluation.py      # Leak-free evaluation verification
│   └── test_new_product.py     # SQLite persistence, TMDb fallback, search typos, MMR, nav order
├── data/                       # MovieLens datasets (auto-downloaded, gitignored)
├── db/                         # Local SQLite persistence (app.db)
├── artifacts/                  # TMDb metadata cache and precomputed matrices
├── outputs/                    # Empirical evaluation summaries & parameter sweeps
├── .streamlit/
│   ├── config.toml             # Dark-Plum (#4B2542) & Coral (#E36F61) theme configuration
│   └── secrets.toml            # Local secrets (TMDb API key)
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Pinned Python package dependencies
└── README.md                   # Complete documentation
```

---

## 🎨 Theme & Design Aesthetics

CineMatch adheres to a sleek dark-mode streaming aesthetic:
- **Background**: Near-black cinematic canvas (`#0d0c13` to `#16131f`).
- **Primary Accent (Plum)**: Deep plum (`#4B2542` / `#6b325d`).
- **Secondary Accent (Coral)**: High-energy coral (`#E36F61` / `#ff7b6b`).
- **Cards & Rows**: Glassmorphic styling, smooth hover zoom animations, badges, and horizontal scrolling rows.

---

## 🖼️ TMDb Poster Integration & Setup

CineMatch fetches high-resolution posters, backdrops, cast, and director details from The Movie Database (TMDb) via `links.csv` (`tmdbId`).

### 1. Getting a Free TMDb API Key
1. Create a free account at [themoviedb.org](https://www.themoviedb.org/signup).
2. Go to **Settings > API** ([themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)).
3. Request an API key (choose "Developer").
4. Copy your **API Read Access Token** or **API Key (v3 auth)**.

### 2. Configuring the API Key
**Option A — Streamlit Secrets (Recommended):**
Create or edit `.streamlit/secrets.toml`:
```toml
TMDB_API_KEY = "your_tmdb_api_key_here"
```

**Option B — Environment Variable:**
```bash
# Windows PowerShell:
$env:TMDB_API_KEY="your_tmdb_api_key_here"

# Linux / macOS / Bash:
export TMDB_API_KEY="your_tmdb_api_key_here"
```

### 3. Graceful Fallback Mode
If no API key is provided or the device is offline:
- The app generates **procedural SVG gradient posters** featuring movie title initials and genre tags.
- A one-time informational banner appears at the top of the interface guiding the user on how to enable real posters.
- All recommendation, search, rating, and comparison functionality remains **100% operational**.

### 4. Disk Caching & Prefetching
- All TMDb responses are saved to `artifacts/tmdb_cache.json`. Every movie is fetched at most once.
- Posters for the home page recommendations are prefetched in background threads to guarantee zero UI latency.

---

## 🚀 Local Installation & Running

### 1. Clone Repository & Create Virtual Environment
```bash
# Navigate to the project root
cd "movie-recomentation system"

# Create a Python 3.10 virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Test Suite
Ensure all 27 unit tests pass:
```bash
python -m pytest -v
```

### 3. Launch CineMatch
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 📊 Recommendation Quality & Offline Evaluation Benchmark

The offline evaluation protocol runs on $N = 300$ held-out test users ($80\%$ train / $20\%$ test per user, fixed random seed `42`, test ratings $\ge 4.0$, evaluated at $K = 10$). Models are trained strictly on training data; train-rated items are excluded to prevent data leakage.

### Benchmark Results at $K = 10$:

| Model | Precision@10 | 95% CI | Recall@10 | 95% CI | Capacity-Norm Recall@10 | 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0220 | ± 0.0057 | 0.0171 | ± 0.0065 | 0.0273 | ± 0.0078 |
| **Content-Based** | 0.0053 | ± 0.0027 | 0.0052 | ± 0.0042 | 0.0078 | ± 0.0047 |
| **Collaborative** | **0.1530** | ± 0.0186 | **0.1309** | ± 0.0188 | **0.1991** | ± 0.0229 |
| **Hybrid ($w = 0.5$)** | 0.1483 | ± 0.0185 | 0.1317 | ± 0.0189 | 0.1961 | ± 0.0230 |

### Parameter Sensitivity & Analysis:
- **Optimal Hybrid Weight**: The offline weight sweep ($w \in [0.0, 1.0]$) peaked at **$w = 0.2$** with **Precision@10 = 0.1563 (± 0.0191)** and **Normalized Recall@10 = 0.2045 (± 0.0235)**.
- **Why Standalone Content Underperformed Offline**: Content-based filtering relying solely on high-level genres lacks quality discrimination (recommending unpopular or low-rated genre matches). CineMatch resolves this in the live product by enriching features with user tags (`tags.csv`), decade bins, Bayesian IMDb rating tie-breakers, and adaptive weighting.
- **Adaptive Hybrid Weighting in Product**:
  - Seasoned users ($>10$ ratings): $w = 0.2$ (prioritizing high-accuracy collaborative filtering).
  - New / Onboarding users ($1-4$ ratings): $w = 0.6 - 0.8$ (prioritizing content taste vectors until sufficient collaborative co-ratings accumulate).

---

## 🛠️ Technology Stack & Attribution

The technologies used in this project are strictly rendered from `src/team_config.py`:

### Verified Technologies (`TOOLS_USED`)
- **Analysis (R)**: R, RStudio, `dplyr`, `tidyr`, `ggplot2`, `proxy`
- **App Core (Python)**: Python 3.10, Streamlit, pandas, scikit-learn, SciPy, Plotly, RapidFuzz, SQLite3, GroupLens MovieLens
- **External Metadata**: The Movie Database (TMDb) API

### Future Scalability Roadmap (`TOOLS_PLANNED`)
- **Distributed Big Data Engine**: Apache Spark (MLlib alternating least squares collaborative filtering on Dataproc / EMR).
- **Data Warehousing & Querying**: Apache Hadoop (HDFS distributed storage), Apache Hive (SQL-like large-scale data queries).

> *Note on Spark/Hadoop/Hive: These technologies represent planned architectural extensions for multi-node production deployment on 100M+ rating datasets. They are cleanly separated from the active standalone demo in `src/team_config.py`.*

---

## ☁️ Deployment to Streamlit Community Cloud

Follow these steps to deploy CineMatch live on [share.streamlit.io](https://share.streamlit.io):

1. **Push Changes to GitHub**:
   Ensure all changes are pushed to your repository `MacAfei/cinematch` on branch `main`.

2. **Create New App on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and click **"Create app"** (or **"New app"**).
   - Choose:
     - **Repository**: `MacAfei/cinematch`
     - **Branch**: `main`
     - **Main file path**: `app.py`
     - **App URL**: `cinematch.streamlit.app` (or your chosen subdomain)

3. **Select Python 3.12 in Advanced Settings**:
   > ⚠️ **CRITICAL: Select Python 3.12**  
   > Click **Advanced settings** before deploying and select **Python 3.12** from the Python version dropdown.  
   > *(Note: Streamlit Community Cloud does not allow changing the Python version of an existing app after deployment; if an older version was selected, you must delete and recreate the app).*

4. **Configure Secrets (TMDb Posters)**:
   In the **Secrets** box under Advanced settings, enter:
   ```toml
   TMDB_API_KEY = "your_actual_tmdb_key_here"
   ```
   *(If you don't have a key yet, leave it empty or configure it later under App Settings ➔ Secrets; the app will run with attractive built-in SVG gradient posters).*

5. **Deploy**:
   Click **"Deploy!"**. CineMatch will install dependencies from `requirements.txt`, load the committed `ml-latest-small` dataset, initialize the local SQLite database, and launch your live streaming app!

---

## 📜 Citations

- **MovieLens Benchmark**:  
  F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: `https://doi.org/10.1145/2827872`.
- **Item-Based Collaborative Filtering**:  
  Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001. *Item-based collaborative filtering recommendation algorithms.* In Proceedings of the 10th international conference on World Wide Web (WWW '01). ACM, New York, NY, USA, 285–295.
- **TMDb Attribution**:  
  *This product uses the TMDB API but is not endorsed or certified by TMDB.*
