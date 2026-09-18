import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="EduPulse",
    page_icon="📊",
    layout="wide"
)


# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

/* =========================
   GLOBAL
   ========================= */

.stApp {
    background-color: #F3F1F8;
}

section[data-testid="stSidebar"] {
    background-color: #7B6BC5;
}

section[data-testid="stSidebar"] * {
    color: white;
}

h1, h2, h3 {
    color: #292832 !important;
}

p {
    color: #6F6D78;
}


/* =========================
   KPI CARDS
   ========================= */

.metric-card {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #E2DEF2;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.metric-title {
    color: #6F6D78;
    font-size: 14px;
}

.metric-value {
    color: #292832;
    font-size: 28px;
    font-weight: 700;
}

/* =========================
   PAGINATION BUTTON
   ========================= */

.stButton > button {
    background-color: #FFFFFF !important;
    color: #6F5BC3 !important;
    border: 1px solid #E2DEF2 !important;
    border-radius: 8px !important;
}

.stButton > button:hover {
    background-color: #EEEBFA !important;
    color: #6F5BC3 !important;
}

/* =========================
   STUDENT PREDICTION INPUTS
   ========================= */

/* White number input fields with dark text */
div[data-testid="stNumberInput"] > div {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
}

div[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #292832 !important;
    -webkit-text-fill-color: #292832 !important;
    border-color: #E2DEF2 !important;
}

div[data-testid="stNumberInput"] input:focus {
    background-color: #FFFFFF !important;
    color: #292832 !important;
    -webkit-text-fill-color: #292832 !important;
    border-color: #806FC9 !important;
}

div[data-testid="stNumberInput"] button {
    background-color: #FFFFFF !important;
    color: #292832 !important;
}

div[data-testid="stNumberInput"] button svg {
    color: #292832 !important;
    fill: #292832 !important;
}

/* Prediction result and recommendation text */
div[data-testid="stAlert"] {
    color: #292832 !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] li,
div[data-testid="stAlert"] span {
    color: #292832 !important;
}

.prediction-recommendation {
    color: #292832 !important;
}

.prediction-recommendation li {
    color: #292832 !important;
    margin-bottom: 6px;
}

/* =========================
   STUDENT PREDICTION BUTTON
   ========================= */

.stButton > button[kind="primary"] {
    background-color: #6F5BC3 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    border: 1px solid #6F5BC3 !important;
    border-radius: 8px !important;
}

.stButton > button[kind="primary"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #806FC9 !important;
    color: #FFFFFF !important;
    border-color: #806FC9 !important;
}


/* =========================
   SEARCH STUDENT ID
   ========================= */

div[data-testid="stTextInput"] > div {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
}

div[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    color: #292832 !important;
    -webkit-text-fill-color: #292832 !important;
    border-color: #E2DEF2 !important;
}

div[data-testid="stTextInput"] input:focus {
    background-color: #FFFFFF !important;
    color: #292832 !important;
    -webkit-text-fill-color: #292832 !important;
    border-color: #806FC9 !important;
    box-shadow: 0 0 0 1px #806FC9 !important;
}

div[data-testid="stTextInput"] label {
    color: #6F6D78 !important;
}

/* =========================
   RISK FILTER - STREAMLIT 1.61
   ========================= */

/* Label */
div[data-testid="stSelectbox"] label {
    color: #6F6D78 !important;
}

/* SEMUA BAGIAN SELECTBOX TERTUTUP */
div[data-testid="stSelectbox"] {
    background-color: transparent !important;
}

/* Wrapper utama */
div[data-testid="stSelectbox"] > div {
    background-color: #FFFFFF !important;
}

/* Semua elemen internal selectbox */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSelectbox"] [role="combobox"],
div[data-testid="stSelectbox"] button,
div[data-testid="stSelectbox"] input {
    background-color: #FFFFFF !important;
    color: #292832 !important;
    -webkit-text-fill-color: #292832 !important;
    border-color: #E2DEF2 !important;
}

/* Teks "All" */
div[data-testid="stSelectbox"] [role="combobox"] *,
div[data-testid="stSelectbox"] button *,
div[data-testid="stSelectbox"] input * {
    background-color: transparent !important;
    color: #292832 !important;
    -webkit-text-fill-color: #292832 !important;
}

/* Border selectbox */
div[data-testid="stSelectbox"] [role="combobox"] {
    border: 1px solid #E2DEF2 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

/* Saat diklik */
div[data-testid="stSelectbox"] [role="combobox"]:focus,
div[data-testid="stSelectbox"] [role="combobox"]:focus-within {
    border-color: #806FC9 !important;
    box-shadow: 0 0 0 1px #806FC9 !important;
}

/* Panah */
div[data-testid="stSelectbox"] svg {
    color: #292832 !important;
    fill: #292832 !important;
}

/* =========================
   DROPDOWN
   ========================= */

div[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"],
div[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
    color: #292832 !important;
}

div[data-testid="stSelectboxVirtualDropdown"] [role="option"],
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="menu"] [role="option"] {
    background-color: #FFFFFF !important;
    color: #292832 !important;
}

div[data-testid="stSelectboxVirtualDropdown"] [role="option"] *,
div[data-baseweb="popover"] [role="option"] *,
div[data-baseweb="menu"] [role="option"] * {
    background-color: transparent !important;
    color: #292832 !important;
}

/* Hover */
div[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="menu"] [role="option"]:hover {
    background-color: #EEEBFA !important;
    color: #292832 !important;
}


/* =========================
   GENERAL
   ========================= */

hr {
    border-color: #E2DEF2 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD DATA
# =========================

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Final_Marks_Data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "edupulse_model_final.joblib"
)

df = pd.read_csv(DATA_PATH)

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]


# =========================
# PREPARE DATA
# =========================

# Standardize column names
df.columns = [
    "student_id",
    "attendance",
    "quiz_1_score",
    "quiz_2_score",
    "assignment_score",
    "daily_study_hours",
    "final_exam_score"
]


# Convert scores to percentage
df["quiz_1_score_pct"] = (
    df["quiz_1_score"] / 40
) * 100

df["quiz_2_score_pct"] = (
    df["quiz_2_score"] / 40
) * 100

df["assignment_pct"] = (
    df["assignment_score"] / 10
) * 100


# =========================
# PREDICTION
# =========================

features = [
    "attendance",
    "quiz_1_score_pct",
    "quiz_2_score_pct",
    "assignment_pct",
    "daily_study_hours"
]

df["predicted_score"] = model.predict(
    df[features]
)


# =========================
# RISK CLASSIFICATION
# =========================

def classify_risk(score):

    if score < 70:
        return "High Risk"

    elif score < 80:
        return "Medium Risk"

    else:
        return "Low Risk"


df["risk"] = df["predicted_score"].apply(
    classify_risk
)


# =========================
# SIDEBAR / NAVIGATION
# =========================
with st.sidebar:
    st.markdown("# 📊 EduPulse")
    st.markdown("### Student Early Warning System")
    st.divider()
    st.markdown("**Dashboard**")
    st.markdown(
        "Pantau performa akademik siswa dan identifikasi siswa "
        "yang memerlukan perhatian lebih awal."
    )

# =========================
# HEADER
# =========================
st.title("Student Performance Dashboard")
st.markdown(
    "Pantau performa akademik siswa dan identifikasi siswa yang "
    "mungkin memerlukan early intervention berdasarkan hasil prediksi Final Exam."
)


# =========================
# KPI CARDS
# =========================
total_students = len(df)
average_final = df["final_exam_score"].mean()
high_risk = (df["risk"] == "High Risk").sum()
low_risk = (df["risk"] == "Low Risk").sum()
medium_risk = (df["risk"] == "Medium Risk").sum()
students_at_risk = high_risk + medium_risk

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class="metric-card"><div class="metric-title">Total Students</div><div class="metric-value">{total_students:,}</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card"><div class="metric-title">Average Final Exam</div><div class="metric-value">{average_final:.1f}</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card"><div class="metric-title">Students At Risk</div><div class="metric-value">{students_at_risk:,}</div></div>""", unsafe_allow_html=True)

st.write("")

# =========================
# PERFORMANCE KPI CARDS
# =========================
performance = df[
    [
        "attendance",
        "quiz_1_score_pct",
        "quiz_2_score_pct",
        "assignment_pct"
    ]
].mean()

performance_col1, performance_col2, performance_col3, performance_col4 = st.columns(4)

with performance_col1:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-title">Average Attendance</div><div class="metric-value">{performance['attendance']:.1f}%</div></div>""",
        unsafe_allow_html=True
    )

with performance_col2:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-title">Average Quiz 1</div><div class="metric-value">{performance['quiz_1_score_pct']:.1f}</div></div>""",
        unsafe_allow_html=True
    )

with performance_col3:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-title">Average Quiz 2</div><div class="metric-value">{performance['quiz_2_score_pct']:.1f}</div></div>""",
        unsafe_allow_html=True
    )

with performance_col4:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-title">Average Assignment</div><div class="metric-value">{performance['assignment_pct']:.1f}</div></div>""",
        unsafe_allow_html=True
    )

st.write("")

# =========================
# RISK SUMMARY CARDS
# =========================

risk_col1, risk_col2, risk_col3 = st.columns(3)

with risk_col1:
    st.markdown(
        f"""
        <div style="background:#FDECEF; border:1px solid #D45A6D; border-radius:12px; padding:18px 20px;">
            <div style="color:#D45A6D; font-size:14px; font-weight:600;">High Risk Students</div>
            <div style="color:#292832; font-size:28px; font-weight:700; margin-top:6px;">{high_risk:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with risk_col2:
    st.markdown(
        f"""
        <div style="background:#FFF7E3; border:1px solid #D0A447; border-radius:12px; padding:18px 20px;">
            <div style="color:#B18420; font-size:14px; font-weight:600;">Medium Risk Students</div>
            <div style="color:#292832; font-size:28px; font-weight:700; margin-top:6px;">{medium_risk:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with risk_col3:
    st.markdown(
        f"""
        <div style="background:#EAF7F0; border:1px solid #58A77E; border-radius:12px; padding:18px 20px;">
            <div style="color:#3E8A65; font-size:14px; font-weight:600;">Low Risk Students</div>
            <div style="color:#292832; font-size:28px; font-weight:700; margin-top:6px;">{low_risk:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# =========================
# TABS
# =========================
tab_dashboard, tab_monitoring, tab_prediction = st.tabs([
    "Dashboard",
    "Student Risk Monitoring",
    "Student Prediction"
])

# =========================
# DASHBOARD TAB
# =========================
with tab_dashboard:

    # =========================
    # RISK FILTER
    # =========================
    st.markdown("### Risk Level")
    analysis_filter = st.selectbox(
        "Filter Risk Level",
        ["All", "High Risk", "Medium Risk", "Low Risk"],
        index=0,
        key="dashboard_risk_filter"
    )

    if analysis_filter == "All":
        analysis_df = df.copy()
    else:
        analysis_df = df[df["risk"] == analysis_filter].copy()

    st.caption(
        f"Menampilkan {len(analysis_df):,} siswa dengan filter: {analysis_filter}."
    )

    # =========================
    # DISTRIBUTION
    # =========================
    st.markdown("## Distribution")

    distribution_features = [
        ("Risk Distribution", "risk", "Risk Level", "risk"),
        ("Final Exam Distribution", "final_exam_score", "Final Exam Score", "hist"),
        ("Attendance Distribution", "attendance", "Attendance (%)", "hist"),
        ("Quiz 1 Distribution", "quiz_1_score_pct", "Quiz 1 Score (%)", "hist"),
        ("Quiz 2 Distribution", "quiz_2_score_pct", "Quiz 2 Score (%)", "hist"),
        ("Assignment Distribution", "assignment_pct", "Assignment Score (%)", "hist"),
        ("Daily Study Hours Distribution", "daily_study_hours", "Daily Study Hours", "hist")
    ]

    for i in range(0, len(distribution_features), 2):
        dist_col1, dist_col2 = st.columns(2)

        for col, item in zip(
            [dist_col1, dist_col2],
            distribution_features[i:i+2]
        ):
            title, column, x_title, chart_type = item

            with col:
                st.subheader(title)

                if chart_type == "risk":
                    risk_counts = (
                        analysis_df["risk"]
                        .value_counts()
                        .reindex(["High Risk", "Medium Risk", "Low Risk"])
                        .fillna(0)
                        .reset_index()
                    )
                    risk_counts.columns = ["Risk", "Students"]

                    fig_dist = px.pie(
                        risk_counts,
                        names="Risk",
                        values="Students",
                        hole=0.55,
                        template="plotly_white",
                        color="Risk",
                        color_discrete_map={
                            "High Risk": "#D45A6D",
                            "Medium Risk": "#D0A447",
                            "Low Risk": "#58A77E"
                        }
                    )
                    fig_dist.update_layout(
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        font=dict(color="#292832"),
                        legend=dict(font=dict(color="#292832")),
                        margin=dict(t=10, b=10, l=10, r=10),
                        showlegend=True
                    )

                else:
                    fig_dist = px.histogram(
                        analysis_df,
                        x=column,
                        nbins=20,
                        template="plotly_white",
                        color_discrete_sequence=["#806FC9"]
                    )
                    fig_dist.update_layout(
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        font=dict(color="#292832"),
                        xaxis=dict(
                            title=x_title,
                            title_font=dict(color="#292832"),
                            tickfont=dict(color="#292832"),
                            gridcolor="#E2DEF2"
                        ),
                        yaxis=dict(
                            title="Number of Students",
                            title_font=dict(color="#292832"),
                            tickfont=dict(color="#292832"),
                            gridcolor="#E2DEF2"
                        ),
                        margin=dict(t=10, b=10, l=10, r=10)
                    )

                st.plotly_chart(
                    fig_dist,
                    use_container_width=True,
                    key=f"dashboard_distribution_{column}"
                )

    # =========================
    # DISTRIBUTION INSIGHT
    # =========================

    avg_final = analysis_df["final_exam_score"].mean()
    avg_attendance = analysis_df["attendance"].mean()
    avg_quiz_1 = analysis_df["quiz_1_score_pct"].mean()
    avg_quiz_2 = analysis_df["quiz_2_score_pct"].mean()
    avg_assignment = analysis_df["assignment_pct"].mean()
    avg_study_hours = analysis_df["daily_study_hours"].mean()

    st.markdown("### 💡 Insight")

    if analysis_filter == "All":

        risk_distribution = (
            analysis_df["risk"]
            .value_counts(normalize=True)
            .reindex(["High Risk", "Medium Risk", "Low Risk"])
            .fillna(0)
        )

        dominant_risk = risk_distribution.idxmax()
        dominant_risk_pct = risk_distribution.max() * 100

        st.info(
            f"Kategori risiko yang paling dominan adalah **{dominant_risk}** "
            f"({dominant_risk_pct:.1f}%). Secara keseluruhan, rata-rata "
            f"Final Exam adalah **{avg_final:.1f}**, attendance **{avg_attendance:.1f}%**, "
            f"Quiz 1 **{avg_quiz_1:.1f}**, Quiz 2 **{avg_quiz_2:.1f}**, "
            f"assignment **{avg_assignment:.1f}**, dan daily study hours "
            f"**{avg_study_hours:.1f} jam/hari**."
        )

    else:

        st.info(
            f"Pada filter **{analysis_filter}**, terdapat **{len(analysis_df):,} siswa** "
            f"dengan rata-rata Final Exam **{avg_final:.1f}**. "
            f"Rata-rata attendance sebesar **{avg_attendance:.1f}%**, "
            f"Quiz 1 **{avg_quiz_1:.1f}**, Quiz 2 **{avg_quiz_2:.1f}**, "
            f"assignment **{avg_assignment:.1f}**, dan daily study hours "
            f"**{avg_study_hours:.1f} jam/hari**."
        )

    # =========================
    # CORRELATION
    # =========================
    st.markdown("## Correlation")

    correlation_columns = [
        "attendance",
        "quiz_1_score_pct",
        "quiz_2_score_pct",
        "assignment_pct",
        "daily_study_hours",
        "final_exam_score"
    ]

    correlation_labels = [
        "Attendance",
        "Quiz 1",
        "Quiz 2",
        "Assignment",
        "Study Hours",
        "Final Exam"
    ]

    corr_matrix = analysis_df[correlation_columns].corr()
    corr_matrix.index = correlation_labels
    corr_matrix.columns = correlation_labels

    corr_col1, corr_col2 = st.columns(2)

    with corr_col1:
        st.markdown("**Correlation Matrix**")
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            template="plotly_white",
            color_continuous_scale="Purples",
            zmin=-1,
            zmax=1
        )
        fig_heatmap.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#292832"),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
            key="dashboard_correlation_matrix"
        )

    with corr_col2:
        st.markdown("**Correlation with Final Exam**")
        correlation_df_analysis = pd.DataFrame({
            "Metric": correlation_labels[:-1],
            "Correlation": [
                corr_matrix.loc[label, "Final Exam"]
                for label in correlation_labels[:-1]
            ]
        })
        correlation_df_analysis["Correlation"] = (
            correlation_df_analysis["Correlation"].round(3)
        )

        fig_corr = px.bar(
            correlation_df_analysis.sort_values("Correlation"),
            x="Correlation",
            y="Metric",
            orientation="h",
            template="plotly_white",
            text="Correlation",
            color_discrete_sequence=["#806FC9"]
        )
        fig_corr.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#292832"),
            xaxis=dict(
                title="Correlation Coefficient",
                title_font=dict(color="#292832"),
                tickfont=dict(color="#292832"),
                gridcolor="#E2DEF2",
                range=[0, 1]
            ),
            yaxis=dict(title="", tickfont=dict(color="#292832")),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(
            fig_corr,
            use_container_width=True,
            key="dashboard_correlation_final_exam"
        )

    # =========================
    # INSIGHT
    # =========================

    strongest = correlation_df_analysis.loc[
        correlation_df_analysis["Correlation"].idxmax()
    ]

    direction = "positif" if strongest["Correlation"] > 0 else "negatif"

    st.markdown("### 💡 Insight")

    st.info(
        f"Indikator dengan hubungan linear paling kuat terhadap **Final Exam** "
        f"pada data yang ditampilkan adalah **{strongest['Metric']}**, "
        f"dengan koefisien korelasi **{strongest['Correlation']:.2f}** "
        f"({direction}). "
        f"Hubungan positif menunjukkan bahwa nilai indikator yang lebih tinggi "
        f"cenderung diikuti oleh nilai Final Exam yang lebih tinggi, sedangkan "
        f"hubungan negatif menunjukkan kecenderungan sebaliknya."
    )

    # =========================
    # RELATIONSHIPS WITH FINAL EXAM
    # =========================
    st.markdown("## Relationship with Final Exam")

    relationship_features = [
        ("Attendance vs Final Exam", "attendance", "Attendance (%)"),
        ("Quiz 1 vs Final Exam", "quiz_1_score_pct", "Quiz 1 Score (%)"),
        ("Quiz 2 vs Final Exam", "quiz_2_score_pct", "Quiz 2 Score (%)"),
        ("Assignment vs Final Exam", "assignment_pct", "Assignment Score (%)"),
        ("Daily Study Hours vs Final Exam", "daily_study_hours", "Daily Study Hours")
    ]

    for i in range(0, len(relationship_features), 2):
        rel_col1, rel_col2 = st.columns(2)
        for col, item in zip(
            [rel_col1, rel_col2],
            relationship_features[i:i+2]
        ):
            title, x_column, x_title = item
            with col:
                st.markdown(f"**{title}**")
                fig_relationship = px.scatter(
                    analysis_df,
                    x=x_column,
                    y="final_exam_score",
                    template="plotly_white",
                    opacity=0.65,
                    labels={
                        x_column: x_title,
                        "final_exam_score": "Final Exam Score"
                    }
                )
                fig_relationship.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#292832"),
                    xaxis=dict(
                        title=x_title,
                        title_font=dict(color="#292832"),
                        tickfont=dict(color="#292832"),
                        gridcolor="#E2DEF2"
                    ),
                    yaxis=dict(
                        title="Final Exam Score",
                        title_font=dict(color="#292832"),
                        tickfont=dict(color="#292832"),
                        gridcolor="#E2DEF2"
                    ),
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(
                    fig_relationship,
                    use_container_width=True,
                    key=f"dashboard_relationship_{x_column}"
                )

    # =========================
    # RELATIONSHIP INSIGHT
    # =========================

    relationship_columns = [
        "attendance",
        "quiz_1_score_pct",
        "quiz_2_score_pct",
        "assignment_pct",
        "daily_study_hours"
    ]

    relationship_labels = {
        "attendance": "Attendance",
        "quiz_1_score_pct": "Quiz 1",
        "quiz_2_score_pct": "Quiz 2",
        "assignment_pct": "Assignment",
        "daily_study_hours": "Daily Study Hours"
    }

    relationship_corr = (
        analysis_df[
            relationship_columns + ["final_exam_score"]
        ]
        .corr(method="spearman")["final_exam_score"]
        .drop("final_exam_score")
    )

    strongest_relationship = relationship_corr.abs().idxmax()

    strongest_relationship_value = relationship_corr.loc[
        strongest_relationship
    ]

    relationship_direction = (
        "positif"
        if strongest_relationship_value > 0
        else "negatif"
    )

    relationship_interpretation = (
        "nilai indikator yang lebih tinggi cenderung diikuti "
        "oleh nilai Final Exam yang lebih tinggi"
        if strongest_relationship_value > 0
        else
        "nilai indikator yang lebih tinggi cenderung diikuti "
        "oleh nilai Final Exam yang lebih rendah"
    )

    st.markdown("### 💡 Insight")

    st.info(
        f"Di antara indikator yang ditampilkan, "
        f"**{relationship_labels[strongest_relationship]}** "
        f"menunjukkan pola hubungan paling kuat dengan **Final Exam**, "
        f"dengan korelasi Spearman **{strongest_relationship_value:.2f}** "
        f"({relationship_direction}). "
        f"Artinya, pada data yang ditampilkan, "
        f"{relationship_interpretation}. "
        f"Temuan ini menunjukkan pola hubungan pada data dan bukan "
        f"hubungan sebab-akibat."
    )    


    # =========================
    # PERFORMANCE OVERVIEW
    # =========================
    st.markdown("## Performance Overview")

    performance = analysis_df[
        [
            "attendance",
            "quiz_1_score_pct",
            "quiz_2_score_pct",
            "assignment_pct",
            "final_exam_score"
        ]
    ].mean()

    performance_df = performance.reset_index()
    performance_df.columns = ["Metric", "Average"]
    performance_df["Average"] = performance_df["Average"].round(1)
    performance_df["Metric"] = [
        "Attendance",
        "Quiz 1",
        "Quiz 2",
        "Assignment",
        "Final Exam"
    ]

    fig_performance = px.bar(
        performance_df,
        x="Metric",
        y="Average",
        template="plotly_white",
        text="Average",
        color_discrete_sequence=["#806FC9"]
    )

    fig_performance.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#292832"),
        xaxis=dict(
            title="",
            tickfont=dict(color="#292832"),
            gridcolor="#E2DEF2"
        ),
        yaxis=dict(
            title="Average",
            title_font=dict(color="#292832"),
            tickfont=dict(color="#292832"),
            gridcolor="#E2DEF2"
        ),
        margin=dict(t=20, b=20, l=20, r=20)
    )

    st.plotly_chart(
        fig_performance,
        use_container_width=True,
        key="dashboard_performance_overview"
    )

# =========================
# MONITORING TAB
# =========================
with tab_monitoring:
    st.markdown("## Student Risk Monitoring")

    # Filter risk level
    risk_filter = st.selectbox(
        "Filter Risk Level",
        ["All", "High Risk", "Medium Risk", "Low Risk"]
    )

    # Filter data berdasarkan risk
    if risk_filter == "All":
        risk_students = df.copy()
    else:
        risk_students = df[df["risk"] == risk_filter].copy()

    # Search Student ID
    search_student = st.text_input(
        "Search Student ID",
        placeholder="Enter Student ID..."
    )

    if search_student.strip():
        risk_students = risk_students[
            risk_students["student_id"]
            .astype(str)
            .str.contains(
                search_student.strip(),
                case=False,
                na=False
            )
        ]

    # Kolom yang ditampilkan
    display_columns = [
        "student_id",
        "final_exam_score",
        "predicted_score",
        "risk"
    ]

    # Bulatkan predicted score
    risk_students["predicted_score"] = (
        risk_students["predicted_score"].round(2)
    )

    # =========================
    # PAGINATION
    # =========================

    page_size = 10

    total_students_displayed = len(risk_students)

    if total_students_displayed == 0:
        st.info("No students found for this risk level.")
    else:

        total_pages = (
            (total_students_displayed - 1) // page_size
        ) + 1

        # Simpan halaman aktif di session state
        if "risk_page" not in st.session_state:
            st.session_state.risk_page = 1

        # Reset ke halaman 1 ketika filter berubah
        if (
            "previous_risk_filter" not in st.session_state
            or st.session_state.previous_risk_filter != risk_filter
        ):
            st.session_state.risk_page = 1
            st.session_state.previous_risk_filter = risk_filter

        # Pastikan halaman tidak melebihi total halaman
        if st.session_state.risk_page > total_pages:
            st.session_state.risk_page = total_pages

        current_page = st.session_state.risk_page

        # Tentukan data yang ditampilkan
        start_index = (current_page - 1) * page_size
        end_index = start_index + page_size

        risk_students_page = risk_students.iloc[
            start_index:end_index
        ]

       # =========================
        # TABLE
        # =========================

        def style_risk(value):
            if value == "High Risk":
                return "color: #D45A6D; font-weight: 600;"
            elif value == "Medium Risk":
                return "color: #D0A447; font-weight: 600;"
            elif value == "Low Risk":
                return "color: #58A77E; font-weight: 600;"
            return ""


        styled_table = (
            risk_students_page[display_columns]
            .style
            .format({
                "predicted_score": "{:.2f}"
            })
            .set_properties(**{
                "background-color": "#FFFFFF",
                "color": "#292832",
                "border-color": "#E2DEF2"
            })
            .map(
                style_risk,
                subset=["risk"]
            )
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#F3F1F8"),
                        ("color", "#292832"),
                        ("font-weight", "600"),
                        ("border-color", "#E2DEF2")
                    ]
                },
                {
                    "selector": "td",
                    "props": [
                        ("border-color", "#E2DEF2")
                    ]
                }
            ])
        )

        st.table(styled_table)

        # =========================
        # PAGINATION BUTTONS
        # =========================

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if current_page > 1:
                if st.button("← Previous", use_container_width=True):
                    st.session_state.risk_page -= 1
                    st.rerun()

        with col2:
            st.markdown(
                f"""
                <div style="
                    text-align: center;
                    color: #292832;
                    font-size: 14px;
                    padding-top: 8px;
                ">
                    Page <b>{current_page}</b> of <b>{total_pages}</b>
                    &nbsp; • &nbsp;
                    Showing <b>{start_index + 1}</b>–
                    <b>{min(end_index, total_students_displayed)}</b>
                    of <b>{total_students_displayed}</b> students
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            if current_page < total_pages:
                if st.button("Next →", use_container_width=True):
                    st.session_state.risk_page += 1
                    st.rerun()


    st.write("")


# =========================
# PREDICTION TAB
# =========================
with tab_prediction:
    st.markdown("---")

    st.subheader("Student Prediction")
    st.markdown(
        "Enter student academic indicators to predict the Final Exam score "
        "and identify the student's risk level."
    )

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            input_attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0
            )

            input_quiz_1 = st.number_input(
                "Quiz 1 Score (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

        with col2:
            input_quiz_2 = st.number_input(
                "Quiz 2 Score (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

            input_assignment = st.number_input(
                "Assignment Score (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

        with col3:
            input_study_hours = st.number_input(
                "Daily Study Hours",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.5
            )

            st.write("")
            predict_button = st.button(
                "Predict Student",
                use_container_width=True,
                type="primary"
            )

        if predict_button:

            input_data = pd.DataFrame([{
                "attendance": input_attendance,
                "quiz_1_score_pct": input_quiz_1,
                "quiz_2_score_pct": input_quiz_2,
                "assignment_pct": input_assignment,
                "daily_study_hours": input_study_hours
            }])

            predicted_score = model.predict(input_data)[0]

            risk = classify_risk(predicted_score)

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Predicted Final Exam",
                    f"{predicted_score:.2f}"
                )

            with result_col2:
                st.metric(
                    "Risk Level",
                    risk
                )

            if risk == "High Risk":
                st.error(
                    "Student is classified as High Risk and may need early intervention."
                )
                st.markdown("### Recommendation")
                st.markdown(
                    """
                    <div class="prediction-recommendation">
                        <ul>
                            <li>Prioritize improving the lowest academic indicators first.</li>
                            <li>Increase study time gradually and maintain a consistent study schedule.</li>
                            <li>Improve attendance and complete assignments on time.</li>
                            <li>Consider early intervention or additional support from a teacher.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif risk == "Medium Risk":
                st.warning(
                    "Student is classified as Medium Risk and should be monitored."
                )
                st.markdown("### Recommendation")
                st.markdown(
                    """
                    <div class="prediction-recommendation">
                        <ul>
                            <li>Strengthen the academic indicators that are still below target.</li>
                            <li>Maintain regular study habits and avoid a decline in performance.</li>
                            <li>Improve quiz and assignment consistency.</li>
                            <li>Continue monitoring progress before the next assessment.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.success(
                    "Student is classified as Low Risk."
                )
                st.markdown("### Recommendation")
                st.markdown(
                    """
                    <div class="prediction-recommendation">
                        <ul>
                            <li>Maintain current attendance and study habits.</li>
                            <li>Keep quiz and assignment performance consistent.</li>
                            <li>Continue preparing regularly to sustain academic performance.</li>
                            <li>Challenge yourself with higher-level learning goals when appropriate.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )