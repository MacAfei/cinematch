"""
CineMatch - Project Overview (Second-Last Page).
Merges all academic analytics and lab findings into a single, cohesive 6-tab interface:
1. What is this project (problem, goal, 3 paradigms, live dataset facts)
2. Pipeline / Steps (visual end-to-end workflow cards)
3. Visualizations (EDA with data-driven insights)
4. User Clustering (K-Means k=3, scatter plot, validation curves)
5. Compare Methods (side-by-side strategy comparison with overlap badges)
6. Evaluation (Precision@10, Recall@10, Normalized Recall, CI error bars, sweep, R comparison)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    DATASET_NAME,
    MIN_RATINGS_PER_MOVIE,
    MIN_RATINGS_PER_USER,
    MIN_CO_RATERS,
    SHRINKAGE_FACTOR,
    OUTPUTS_DIR
)
from src.app_state import get_app_engine
from src.evaluation import (
    OfflineEvaluator,
    load_evaluation_results,
    load_r_evaluation_results
)
from src.ui_components import (
    inject_custom_css,
    render_top_navbar,
    get_movie_poster_url
)

st.set_page_config(
    page_title="Project Overview | CineMatch",
    page_icon="📑",
    layout="wide"
)

inject_custom_css()

# Theme constants
COLOR_CORAL = "#E36F61"
COLOR_DARK_PLUM = "#4B2542"
COLOR_SECONDARY = "#8B4F7B"
CLUSTER_COLORS = ["#E36F61", "#8B4F7B", "#5C7D9D"]

try:
    engine = get_app_engine()
except Exception as e:
    st.error(f"Error loading application data: {str(e)}")
    st.stop()

movies_df = engine["movies_df"]
ratings_df = engine["ratings_df"]
links_df = engine["links_df"]
movie_stats = engine["movie_stats"]
content_rec = engine["content_recommender"]
collab_rec = engine["collab_recommender"]
hybrid_rec = engine["hybrid_recommender"]
cluster_model = engine["cluster_model"]

render_top_navbar()

st.title("📑 Big Data Analytics Lab: Project Overview")
st.markdown(
    "Comprehensive academic walkthrough of our **Movie Recommendation System**, detailing the mathematical paradigms, "
    "data engineering pipeline, exploratory analysis, unsupervised clustering, comparative evaluation, and production deployment."
)

# 6 Sticky Tabs
tabs = st.tabs([
    "💡 What is this Project",
    "🔄 Pipeline / Steps",
    "📊 Visualizations (EDA)",
    "👥 User Clustering",
    "⚖️ Compare Methods",
    "📈 Evaluation Benchmark"
])

# =========================================================================
# TAB 1: WHAT IS THIS PROJECT
# =========================================================================
with tabs[0]:
    st.subheader("1. Problem Statement & Objectives")
    st.markdown("""
    Modern video-on-demand platforms present audiences with tens of thousands of choices. 
    Without personalized algorithmic curation, users experience **cognitive choice overload**, spending excessive time 
    browsing instead of enjoying cinema.
    
    The goal of this project is to implement, evaluate, and deploy an end-to-end multi-strategy recommendation engine 
    originating in **R** (Big Data Analytics Lab) and evolved into high-performance Python and Streamlit.
    """)

    st.markdown("### The Three Recommendation Paradigms")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            f"""
            <div style="background:#181222; border:1px solid #372348; border-radius:12px; padding:18px; height:100%;">
                <div style="font-size:2rem; margin-bottom:6px;">🏷️</div>
                <h4 style="color:#FAF5EC; margin:0 0 6px 0;">Content-Based</h4>
                <p style="font-size:0.85rem; color:#A896A6; line-height:1.4;">
                    Evaluates intrinsic item metadata (genres, user tags, release era). Recommends titles with maximum cosine similarity to items you previously enjoyed.
                </p>
                <span class="genre-badge">Cold-Start Resilient</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with p2:
        st.markdown(
            f"""
            <div style="background:#181222; border:1px solid #372348; border-radius:12px; padding:18px; height:100%;">
                <div style="font-size:2rem; margin-bottom:6px;">👥</div>
                <h4 style="color:#FAF5EC; margin:0 0 6px 0;">Collaborative Filtering</h4>
                <p style="font-size:0.85rem; color:#A896A6; line-height:1.4;">
                    Operates on sparse user-item interaction matrices using user mean-centered adjusted cosine similarity with shrinkage regularization.
                </p>
                <span class="genre-badge">High Serendipity</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with p3:
        st.markdown(
            f"""
            <div style="background:#181222; border:1px solid #372348; border-radius:12px; padding:18px; height:100%;">
                <div style="font-size:2rem; margin-bottom:6px;">✨</div>
                <h4 style="color:#FAF5EC; margin:0 0 6px 0;">Hybrid System</h4>
                <p style="font-size:0.85rem; color:#A896A6; line-height:1.4;">
                    Applies dynamic min-max normalization fusing content metadata and collaborative co-rating patterns with adaptive user-weighting.
                </p>
                <span class="genre-badge">Best Real-World Precision</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Live Dataset Facts")
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.metric("Dataset Benchmark", DATASET_NAME)
    with d2:
        st.metric("Total Titles", f"{len(movies_df):,}")
    with d3:
        st.metric("Total Ratings", f"{len(ratings_df):,}")
    with d4:
        st.metric("Unique Users", f"{ratings_df['userId'].nunique():,}")
    with d5:
        unique_genres = len({g for sub in movies_df["genre_list"] for g in sub})
        st.metric("Unique Genres", f"{unique_genres}")

    st.caption(f"Active Filters: Min Movie Ratings = {MIN_RATINGS_PER_MOVIE}, Min User Ratings = {MIN_RATINGS_PER_USER}, Min Co-Raters = {MIN_CO_RATERS}, Shrinkage Factor = {SHRINKAGE_FACTOR}.")

# =========================================================================
# TAB 2: PIPELINE / STEPS
# =========================================================================
with tabs[1]:
    st.subheader("2. End-to-End System Workflow")
    st.markdown("The 7 systematic phases executed across data engineering, analysis, modeling, evaluation, and application deployment:")

    steps = [
        {"num": "1", "title": "Data Ingestion & Parsing", "tool": "Python (urllib / requests / zipfile)", "desc": "Automatic download of MovieLens archive, title release-year extraction via regex, genre parsing, and links mapping.", "out": "Cleaned DataFrames (9,742 movies, 100,836 ratings)"},
        {"num": "2", "title": "Feature Engineering", "tool": "scikit-learn & SciPy Sparse", "desc": "Multi-hot genre binarization, TF-IDF tag representations, release decade encoding, and IMDb Bayesian weighted rating formula.", "out": "Enriched multimodal sparse matrix (L2-normalized)"},
        {"num": "3", "title": "Exploratory Data Analysis (EDA)", "tool": "R (ggplot2) & Python (Plotly)", "desc": "Rating distributions, heavy user activity profiles, genre popularity, and temporal volume trends.", "out": "6 Interactive exploratory visualizations"},
        {"num": "4", "title": "Unsupervised User Clustering", "tool": "StandardScaler & KMeans (k=3)", "desc": "Clustering users by activity count and average score; Elbow inertia and Silhouette validation ($k=2..8$).", "out": "3 behavioral personas (Heavy Raters, Generous Casuals, Critical Casuals)"},
        {"num": "5", "title": "Recommendation Engine Development", "tool": "Sparse Linear Algebra", "desc": "Content cosine similarity, user mean-centered adjusted cosine collaborative filtering with shrinkage, and min-max hybrid fusion.", "out": "3 core recommender algorithms + adaptive cold start"},
        {"num": "6", "title": "Leak-Free Offline Evaluation", "tool": "Per-User Holdout (80/20 Split)", "desc": "Evaluation on 300 test users (relevant $\ge 4.0$) measuring Precision@10, Recall@10, Normalized Recall, and 95% CI error bounds.", "out": "Benchmark CSV summaries and weight sweeps"},
        {"num": "7", "title": "Product Deployment", "tool": "Streamlit & SQLite", "desc": "Netflix-style streaming UI, SQLite local persistence, TMDb poster caching, and diversity re-ranking.", "out": "Production-ready CineMatch web application"}
    ]

    for s in steps:
        st.markdown(
            f"""
            <div style="background:#181222; border-left:4px solid #E36F61; border-radius:8px; padding:14px 18px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color:#FAF5EC; font-size:1.05rem;">Step {s['num']}: {s['title']}</strong>
                    <span style="font-size:0.75rem; color:#FF9082; background:rgba(227,111,97,0.15); padding:3px 8px; border-radius:10px;">{s['tool']}</span>
                </div>
                <p style="color:#D4C3D0; font-size:0.88rem; margin:6px 0 4px 0;">{s['desc']}</p>
                <div style="font-size:0.8rem; color:#A896A6;"><strong>Key Artifact:</strong> {s['out']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================================
# TAB 3: VISUALIZATIONS (EDA)
# =========================================================================
with tabs[2]:
    st.subheader("3. Exploratory Data Analysis (EDA)")

    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        st.markdown("#### Rating Distribution")
        r_counts = ratings_df["rating"].value_counts().sort_index().reset_index()
        r_counts.columns = ["Rating", "Count"]
        r_counts["Pct"] = (r_counts["Count"] / len(ratings_df)) * 100

        fig_r = px.bar(r_counts, x="Rating", y="Count", text=r_counts["Pct"].apply(lambda p: f"{p:.1f}%"), color_discrete_sequence=[COLOR_CORAL])
        fig_r.update_traces(textposition="outside")
        fig_r.update_layout(plot_bgcolor="#181222", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FAF5EC"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_r, use_container_width=True)
        st.info("💡 **Insight:** The distribution peaks at 4.0 (26.6%) and 3.0 (20.0%). Half-star ratings occur less frequently, indicating that users predominantly review movies they enjoyed.")

    with eda_c2:
        st.markdown("#### User Average Rating Distribution")
        user_means = ratings_df.groupby("userId")["rating"].mean().reset_index()
        fig_u = px.histogram(user_means, x="rating", nbins=25, color_discrete_sequence=[COLOR_SECONDARY])
        fig_u.add_vline(x=user_means["rating"].mean(), line_dash="dash", line_color=COLOR_CORAL, annotation_text=f"Mean: {user_means['rating'].mean():.2f}")
        fig_u.update_layout(plot_bgcolor="#181222", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FAF5EC"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_u, use_container_width=True)
        st.info(f"💡 **Insight:** Overall user mean rating is {user_means['rating'].mean():.2f}★ with slight negative skew. Most raters are generous, while a small minority grade strictly below 3.0★.")

    st.markdown("---")
    eda_c3, eda_c4 = st.columns(2)
    with eda_c3:
        st.markdown("#### Top 10 Most-Rated Movies")
        top10_m = movie_stats.sort_values(by="n_ratings", ascending=True).tail(10)
        fig_top = px.bar(top10_m, y="title", x="n_ratings", orientation="h", color_discrete_sequence=[COLOR_CORAL])
        fig_top.update_layout(plot_bgcolor="#181222", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FAF5EC"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_top, use_container_width=True)
        st.info("💡 **Insight:** Forrest Gump, Shawshank Redemption, and Pulp Fiction lead total volume, creating popularity hubs that collaborative filtering must regularize against.")

    with eda_c4:
        st.markdown("#### Genre Frequency Across Catalog")
        g_series = movies_df["genre_list"].explode()
        g_df = g_series.value_counts().reset_index()
        g_df.columns = ["Genre", "Count"]
        fig_g = px.bar(g_df, x="Genre", y="Count", color_discrete_sequence=[COLOR_DARK_PLUM])
        fig_g.update_layout(plot_bgcolor="#181222", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FAF5EC"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_g, use_container_width=True)
        st.info("💡 **Insight:** Drama (4,361 titles) and Comedy (3,756 titles) dominate the catalog, highlighting why genre-only features alone produce low discriminatory precision.")

# =========================================================================
# TAB 4: USER CLUSTERING
# =========================================================================
with tabs[3]:
    st.subheader("4. Unsupervised K-Means User Segmentation ($k = 3$)")
    user_data = cluster_model.user_data
    profiles_c = cluster_model.cluster_profiles

    st.markdown("#### Cluster Profiles Summary")
    st.dataframe(profiles_c, use_container_width=True)

    st.markdown("#### Interactive Cluster Scatter Plot")
    fig_cl = px.scatter(
        user_data,
        x="n_ratings",
        y="avg_rating",
        color="cluster_name",
        hover_name="userId",
        color_discrete_sequence=CLUSTER_COLORS,
        labels={"n_ratings": "Number of Ratings Given", "avg_rating": "Average Rating", "cluster_name": "Segment"}
    )
    fig_cl.update_layout(plot_bgcolor="#181222", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FAF5EC"), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_cl, use_container_width=True)

    st.markdown("#### Mathematical Justification for $k = 3$")
    j1, j2 = st.columns(2)
    eval_curves = cluster_model.compute_evaluation_curves(ratings_df, max_k=8)
    with j1:
        fig_elb = px.line(eval_curves, x="k", y="inertia", markers=True, color_discrete_sequence=[COLOR_CORAL], title="Elbow Method (Inertia vs. k)")
        fig_elb.update_layout(plot_bgcolor="#181222", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FAF5EC"))
        st.plotly_chart(fig_elb, use_container_width=True)
    with j2:
        fig_s = px.line(eval_curves, x="k", y="silhouette_score", markers=True, color_discrete_sequence=[COLOR_DARK_PLUM], title="Silhouette Score vs. k")
        fig_s.update_layout(plot_bgcolor="#181222", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FAF5EC"))
        st.plotly_chart(fig_s, use_container_width=True)

    st.info("💡 **Why $k = 3$:** The rate of inertia decline shows a marked elbow bend at $k = 3$ while retaining a high silhouette coefficient, cleanly separating high-volume cinephiles from generous and critical casual viewers.")

# =========================================================================
# TAB 5: COMPARE METHODS
# =========================================================================
with tabs[4]:
    st.subheader("5. Side-by-Side Model Comparison")
    st.markdown("Pick any seed movie to observe how Content-Based, Collaborative, and Hybrid recommendation outputs converge and diverge.")

    comp_title = st.selectbox("Select Seed Movie:", options=movies_df["title"].tolist(), index=0, key="overview_comp_select")
    comp_id = int(movies_df.loc[movies_df["title"] == comp_title, "movieId"].iloc[0])

    c_df = content_rec.recommend_content(movie_id=comp_id, n=10)
    co_df = collab_rec.recommend_collab(movie_id=comp_id, n=10)
    h_df = hybrid_rec.recommend_hybrid(movie_id=comp_id, w=0.2, n=10)

    s_c = set(c_df["movieId"])
    s_co = set(co_df["movieId"])
    s_h = set(h_df["movieId"])
    in_all = s_c.intersection(s_co).intersection(s_h)

    m_o1, m_o2, m_o3 = st.columns(3)
    with m_o1:
        st.metric("Total Unique Titles", f"{len(s_c.union(s_co).union(s_h))}")
    with m_o2:
        st.metric("Consensus in All 3 Lists", f"{len(in_all)}")
    with m_o3:
        st.metric("Collab ∩ Hybrid", f"{len(s_co.intersection(s_h))}")

    def _display_cols(df, score_col):
        t_col = "clean_title" if "clean_title" in df.columns else "title"
        cols = [c for c in [t_col, "year", score_col] if c in df.columns]
        return df[cols]

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("#### 1. Content-Based")
        st.caption("Matches metadata (genres, tags, decade). Best when items are newly added.")
        st.dataframe(_display_cols(c_df, "content_score"), use_container_width=True)
    with col_m2:
        st.markdown("#### 2. Collaborative")
        st.caption("Matches audience co-rating habits. Best for discovering unexpected crossover gems.")
        st.dataframe(_display_cols(co_df, "collab_score"), use_container_width=True)
    with col_m3:
        st.markdown("#### 3. Hybrid (w=0.2)")
        st.caption("Regularized balance. Combines collaborative strength with metadata guidance.")
        st.dataframe(_display_cols(h_df, "hybrid_score"), use_container_width=True)

# =========================================================================
# TAB 6: EVALUATION BENCHMARK
# =========================================================================
with tabs[5]:
    st.subheader("6. Offline Benchmark Protocol & Live Results")
    st.markdown("""
    Offline evaluation protocol:
    - **Split:** 80% train / 20% test per user (fixed `random_state=42`).
    - **Relevance:** Held-out test ratings $\ge 4.0$.
    - **No Data Leakage:** Models trained solely on train data; train-rated items strictly excluded.
    """)

    summary_df, sweep_df = load_evaluation_results()

    if summary_df is not None:
        st.markdown("#### Empirical Benchmark Results at K=10")
        st.dataframe(summary_df, use_container_width=True)

        ev_c1, ev_c2 = st.columns(2)
        with ev_c1:
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(name="Precision@10", x=summary_df["Model"], y=summary_df["Precision@10"], error_y=dict(type="data", array=summary_df["Precision_CI95"], visible=True), marker_color=COLOR_CORAL))
            fig_b.add_trace(go.Bar(name="Recall@10", x=summary_df["Model"], y=summary_df["Recall@10"], error_y=dict(type="data", array=summary_df["Recall_CI95"], visible=True), marker_color=COLOR_DARK_PLUM))
            fig_b.add_trace(go.Bar(name="Norm Recall@10", x=summary_df["Model"], y=summary_df["Normalized_Recall@10"], error_y=dict(type="data", array=summary_df["Normalized_Recall_CI95"], visible=True), marker_color=COLOR_SECONDARY))
            fig_b.update_layout(barmode="group", plot_bgcolor="#181222", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FAF5EC"), legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_b, use_container_width=True)

        with ev_c2:
            fig_sw = px.line(sweep_df, x="Content_Weight_w", y=["Precision@10", "Recall@10", "Normalized_Recall@10"], markers=True, color_discrete_sequence=[COLOR_CORAL, COLOR_DARK_PLUM, COLOR_SECONDARY], title="Hybrid Weight Sweep Sensitivity")
            fig_sw.update_layout(plot_bgcolor="#181222", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FAF5EC"), legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_sw, use_container_width=True)

        st.info("💡 **Key Finding:** The weight sweep demonstrates an empirical peak at $w = 0.2$ (Precision@10 = 0.1563). Incorporating 10%–20% metadata regularizes collaborative filtering without diluting personalized audience preferences.")

    # Optional R Results
    r_results = load_r_evaluation_results()
    if r_results is not None:
        st.markdown("#### Side-by-Side Comparison: Python Engine vs. R Pipeline")
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            st.markdown("**Python App Results (Current Run)**")
            st.dataframe(summary_df[["Model", "Precision@10", "Recall@10"]], use_container_width=True)
        with r_c2:
            st.markdown("**R Pipeline Lab Results (`model_evaluation.csv`)**")
            st.dataframe(r_results, use_container_width=True)
    else:
        st.caption("ℹ️ To display original R pipeline lab results, save them as `outputs/model_evaluation.csv`.")

    # Re-run evaluation button
    if st.button("🚀 Re-Run Evaluation Protocol Live", key="overview_rerun_eval_btn"):
        prog = st.progress(0.0)
        status = st.empty()
        def cb(pct, msg):
            prog.progress(pct)
            status.text(msg)
        evaluator = OfflineEvaluator(movies_df=movies_df, ratings_df=ratings_df, sample_users=300)
        s_df, sw_df = evaluator.run_evaluation(progress_callback=cb, save_to_disk=True)
        st.success("Evaluation complete and outputs updated!")
        st.rerun()
