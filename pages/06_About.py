"""
CineMatch - About & Credits (Last Page).
Renders project abstract, team member cards, methodology overview,
dynamically grouped tools from team_config.py, citations, and TMDb attribution.
"""

import streamlit as st

from src.team_config import (
    PROJECT_NAME,
    PROJECT_TAGLINE,
    ACADEMIC_COURSE,
    INSTITUTION,
    DEPARTMENT,
    ABSTRACT,
    TEAM_MEMBERS,
    TOOLS_USED,
    TOOLS_PLANNED,
    DATASET_CITATION,
    TMDB_ATTRIBUTION,
    GITHUB_REPO_URL
)
from src.tmdb_client import get_tmdb_api_key
from src.ui_components import inject_custom_css, render_top_navbar

st.set_page_config(
    page_title="About | CineMatch",
    page_icon="ℹ️",
    layout="wide"
)

inject_custom_css()
render_top_navbar()

# Section 1: Title, Tagline, Abstract
st.title(f"ℹ️ About {PROJECT_NAME}")
st.markdown(f"#### *{PROJECT_TAGLINE}*")
st.markdown(
    f"""
    <div style="background:#181222; border-left:4px solid #E36F61; border-radius:10px; padding:18px 22px; margin:16px 0 24px 0;">
        <h4 style="color:#FAF5EC; margin:0 0 8px 0;">📄 Project Abstract</h4>
        <p style="color:#D4C3D0; font-size:1.0rem; line-height:1.6; margin:0;">
            {ABSTRACT}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Section 2: Developed By (Team Cards)
st.subheader("👥 Developed By")
st.markdown(f"**{DEPARTMENT}**  \n*{INSTITUTION} — {ACADEMIC_COURSE}*")

cols = st.columns(len(TEAM_MEMBERS))
for i, member in enumerate(TEAM_MEMBERS):
    with cols[i]:
        st.markdown(
            f"""
            <div style="background:#181222; border:1px solid #372348; border-radius:12px; padding:16px 12px; text-align:center; height:100%; box-shadow:0 4px 12px rgba(0,0,0,0.3);">
                <div style="font-size:2.4rem; margin-bottom:6px;">{member.get('avatar', '🎓')}</div>
                <div style="font-weight:700; color:#FAF5EC; font-size:0.95rem; margin-bottom:4px;">{member['name']}</div>
                <div style="font-size:0.75rem; color:#E36F61; font-weight:600; margin-bottom:8px;">{member.get('role', 'Researcher')}</div>
                <a href="{member.get('linkedin', '#')}" target="_blank" style="text-decoration:none; font-size:0.75rem; color:#A896A6;">🔗 Profile</a>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# Section 3: How it was built
st.subheader("🛠️ How It Was Built")
st.markdown("""
The project integrates comprehensive exploratory data analysis, empirical unsupervised clustering, and three distinct recommendation paradigms:
1. **Exploratory Analysis (EDA)**: Investigated rating distributions, user behavior, and catalog sparsity in both R and Python.
2. **K-Means Clustering ($k = 3$)**: Discovered distinct behavioral personas (Heavy Reviewers, Generous Casuals, Critical Casuals), mathematically justified using Elbow and Silhouette metrics.
3. **Tri-Strategy Recommender**: Formulated sparse binary content-based matching, mean-centered collaborative filtering with shrinkage, and min-max normalized hybrid fusion.
4. **Offline Evaluation**: Benchmark validation across 300 held-out test users, proving peak ranking precision at $w = 0.2$.

👉 *For full mathematical derivations, workflow diagrams, and interactive charts, visit the **Project Overview** page.*
""")

st.markdown("---")

# Section 4: Technologies Used (rendered strictly from team_config.py)
st.subheader("💻 Technologies Used")
st.markdown("Configured and rendered from `src/team_config.py`:")

tech_cols = st.columns(len(TOOLS_USED))
for idx, (group_name, tools_list) in enumerate(TOOLS_USED.items()):
    with tech_cols[idx]:
        st.markdown(f"#### {group_name}")
        for t in tools_list:
            st.markdown(f"- `{t}`")

if TOOLS_PLANNED:
    st.markdown("#### Scalability / Future Work")
    for t in TOOLS_PLANNED:
        st.markdown(f"- ⏳ `{t}`")
else:
    with st.expander("⏳ Scalability / Future Work (Planned Technologies)", expanded=False):
        st.markdown(
            "To scale this engine to enterprise Big Data workloads (e.g. MovieLens 32M+ ratings), "
            "distributed processing engines (such as **Apache Spark**, **Hadoop HDFS**, and **Apache Hive**) "
            "can be activated by moving them into `TOOLS_PLANNED` in `src/team_config.py`."
        )

st.markdown("---")

# Section 5: Citations, Attribution, GitHub Repo
c_col1, c_col2 = st.columns(2)
with c_col1:
    st.subheader("📚 Dataset & Citation")
    st.markdown(f"*{DATASET_CITATION}*")
    st.markdown("[GroupLens Research Datasets](https://grouplens.org/datasets/movielens/)")

with c_col2:
    st.subheader("🎬 Attribution & Links")
    if get_tmdb_api_key():
        st.markdown(f"*{TMDB_ATTRIBUTION}*")
    st.markdown(f"🔗 **Repository:** [{GITHUB_REPO_URL}]({GITHUB_REPO_URL})")
