"""
Configuration for Team Members, Project Metadata, and Technologies Used.
Centralizes credits, department details, and tools for the About page.
"""

PROJECT_NAME = "CineMatch"
PROJECT_TAGLINE = "Intelligent Multi-Strategy Movie Recommendation System"
ACADEMIC_COURSE = "Big Data Analytics Laboratory"
INSTITUTION = "Ilahia College of Engineering and Technology"
DEPARTMENT = "Department of Artificial Intelligence & Data Science"

ABSTRACT = (
    "CineMatch is an interactive, production-grade movie recommendation platform that addresses "
    "choice overload by harmonizing content-based metadata, sparse item-based collaborative filtering, "
    "and dynamic hybrid weighting. Built as the deployable evolution of our Big Data Analytics Lab project, "
    "it empowers cinephiles to explore personalized suggestions, discover hidden cinematic gems, "
    "and inspect mathematical explanations behind every recommendation in real time."
)

# Team members: customizable fields (role, linkedin, photo)
TEAM_MEMBERS = [
    {
        "name": "Farsana Kabeer",
        "role": "Team Lead & ML Researcher",
        "linkedin": "https://linkedin.com",
        "avatar": "👩‍💻"
    },
    {
        "name": "Afeefa Parvin Ashruf",
        "role": "Data Engineer & App Developer",
        "linkedin": "https://linkedin.com",
        "avatar": "👩‍💻"
    },
    {
        "name": "Muhammed Safwan",
        "role": "Algorithms & Evaluation Specialist",
        "linkedin": "https://linkedin.com",
        "avatar": "👨‍💻"
    },
    {
        "name": "Muhammed Anees MI",
        "role": "EDA & Visualization Architect",
        "linkedin": "https://linkedin.com",
        "avatar": "👨‍💻"
    },
    {
        "name": "Sharhabeel Mohammed Shameer K",
        "role": "Clustering & Analytics Engineer",
        "linkedin": "https://linkedin.com",
        "avatar": "👨‍💻"
    }
]

# Technologies used in the project
# ONLY tools that were actually implemented are listed here.
TOOLS_USED = {
    "Analysis (R)": [
        "R",
        "RStudio",
        "dplyr",
        "tidyr",
        "ggplot2",
        "proxy"
    ],
    "Big data stack": [
        "MovieLens (100K Benchmark Dataset)",
        "SQLite (Local Persistent Relational DB)"
    ],
    "App (Python)": [
        "Python 3.10+",
        "pandas",
        "scikit-learn",
        "SciPy (Sparse Linear Algebra)",
        "Plotly",
        "Streamlit",
        "RapidFuzz (Fuzzy Search)",
        "Requests"
    ]
}

# Scalability / Future Work: empty by default per user guidelines.
# To add tools here if utilized in extended deployments:
# e.g., ["Apache Spark", "Hadoop HDFS", "Apache Hive"]
TOOLS_PLANNED = [
    # "Apache Spark (Distributed Collaborative Filtering on Dataproc / EMR)",
    # "Hadoop HDFS (Distributed Distributed File System for Big Data Datasets)",
    # "Apache Hive (Data Warehousing & SQL Query Engine)"
]

DATASET_CITATION = (
    "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. "
    "ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015)."
)

TMDB_ATTRIBUTION = (
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)

GITHUB_REPO_URL = "https://github.com/MacAfei/movie-recomentation-system"
