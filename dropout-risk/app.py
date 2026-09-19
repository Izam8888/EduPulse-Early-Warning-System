
import os
import importlib.util

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="EduPulse",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
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

    p, label {
        color: #4A4655 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: white !important;
    }

    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E2DEF2;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .metric-title {
        color: #4A4655 !important;
        font-size: 14px;
        font-weight: 500;
    }

    .metric-value {
        color: #292832 !important;
        font-size: 28px;
        font-weight: 700;
    }

    .stButton > button {
        background-color: #FFFFFF !important;
        color: #6F5BC3 !important;
        border: 1px solid #E2DEF2 !important;
        border-radius: 8px !important;
    }

    .stButton > button:hover {
        background-color: #E5DFFC !important;
        color: #5F4DB0 !important;
        border-color: #6F5BC3 !important;
        box-shadow: 0 0 0 1px #6F5BC3 !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #6F5BC3 !important;
        color: #FFFFFF !important;
        border: 1px solid #6F5BC3 !important;
        border-radius: 8px !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #806FC9 !important;
        color: #FFFFFF !important;
        border-color: #806FC9 !important;
    }

    .stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #6F5BC3 !important;
        border: 1px solid #E2DEF2 !important;
        border-radius: 8px !important;
    }

    .stDownloadButton > button:hover {
        background-color: #E5DFFC !important;
        color: #5F4DB0 !important;
        border-color: #6F5BC3 !important;
    }

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
        border-color: #806FC9 !important;
        box-shadow: 0 0 0 1px #806FC9 !important;
    }

    div[data-testid="stNumberInput"] button {
        background-color: #FFFFFF !important;
        color: #292832 !important;
    }

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

    div[data-testid="stTextInput"] input::placeholder {
        color: #6F6D78 !important;
        opacity: 1 !important;
    }

    div[data-testid="stSelectbox"] [role="combobox"] {
        background-color: #FFFFFF !important;
        color: #292832 !important;
        border: 1px solid #E2DEF2 !important;
        border-radius: 8px !important;
    }

    div[data-testid="stSelectbox"] [role="combobox"] * {
        color: #292832 !important;
    }

    hr {
        border-color: #E2DEF2 !important;
    }

    .prediction-recommendation {
        color: #292832 !important;
    }

    .prediction-recommendation li {
        color: #292832 !important;
        margin-bottom: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

def apply_plotly_theme(fig):
    fig.update_layout(
        font=dict(
            family="Arial",
            color="#292832",
            size=12
        ),

        legend=dict(
            font=dict(
                color="#292832",
                size=12
            )
        ),

        xaxis=dict(
            title_font=dict(
                color="#292832",
                size=13
            ),
            tickfont=dict(
                color="#292832",
                size=11
            )
        ),

        yaxis=dict(
            title_font=dict(
                color="#292832",
                size=13
            ),
            tickfont=dict(
                color="#292832",
                size=11
            )
        ),

        coloraxis_colorbar=dict(
            tickfont=dict(
                color="#292832",
                size=11
            ),
            title_font=dict(
                color="#292832",
                size=12
            )
        )
    )

    fig.update_coloraxes(
        colorbar_tickfont=dict(
            color="#292832",
            size=11
        ),
        colorbar_title_font=dict(
            color="#292832",
            size=12
        )
    )

    return fig


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "student_dropout_dataset_v3.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "edupulse_dropout_model.joblib"
)

PREDICT_PATH = os.path.join(
    BASE_DIR,
    "predict.py"
)


# =========================================================
# MODEL / FUNCTION LOADING
# =========================================================

@st.cache_resource
def load_model_package():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)

    required_keys = {
        "preprocessor",
        "model",
        "features",
        "target",
        "risk_thresholds"
    }

    missing_keys = required_keys.difference(package.keys())

    if missing_keys:
        raise ValueError(
            f"Isi model package tidak lengkap. Missing keys: {sorted(missing_keys)}"
        )

    return package


@st.cache_resource
def load_prediction_function():
    if not os.path.exists(PREDICT_PATH):
        raise FileNotFoundError(
            f"predict.py tidak ditemukan: {PREDICT_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "edupulse_predict",
        PREDICT_PATH
    )

    if spec is None or spec.loader is None:
        raise ImportError("Gagal memuat predict.py.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "predict_dropout"):
        raise AttributeError(
            "predict.py tidak memiliki fungsi predict_dropout()."
        )

    return module.predict_dropout


@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "Student_ID",
        "GPA",
        "Stress_Index",
        "Attendance_Rate",
        "Study_Hours_per_Day",
        "Travel_Time_Minutes",
        "Assignment_Delay_Days",
        "Dropout"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom dataset tidak lengkap: {missing_columns}"
        )

    return data


try:
    model_package = load_model_package()
    predict_dropout = load_prediction_function()
    df = load_dataset()
except Exception as exc:
    st.error("Aplikasi belum dapat dijalankan karena ada file/konfigurasi yang belum sesuai.")
    st.exception(exc)
    st.stop()


# =========================================================
# MODEL VARIABLES
# =========================================================

preprocessor = model_package["preprocessor"]
model = model_package["model"]
model_features = model_package["features"]
risk_thresholds = model_package["risk_thresholds"]


def classify_risk(probability):
    if probability >= risk_thresholds["high"]:
        return "High Risk"
    elif probability >= risk_thresholds["medium"]:
        return "Medium Risk"
    return "Low Risk"


# =========================================================
# DATA PREDICTION FOR DASHBOARD
# =========================================================

@st.cache_data
def build_dashboard_data(data):
    dashboard_df = data.copy()

    model_input = dashboard_df[model_features].copy()
    transformed = preprocessor.transform(model_input)

    dashboard_df["dropout_probability"] = model.predict_proba(
        transformed
    )[:, 1]

    dashboard_df["risk"] = dashboard_df["dropout_probability"].apply(
        classify_risk
    )

    dashboard_df["dropout_probability"] = (
        dashboard_df["dropout_probability"] * 100
    )

    return dashboard_df


df = build_dashboard_data(df)


# =========================================================
# SIDEBAR / NAVIGATION
# =========================================================

with st.sidebar:

    st.image(
        "new-logo-edupulse/new-logo-edupulse.svg",
        width=125
    )

    st.markdown("### Student Early Warning System")

    st.divider()

    st.markdown("**Student Dropout Risk Dashboard**")
    st.markdown(
        "Pantau indikator siswa untuk mengidentifikasi risiko "
        "dropout dan mendukung intervensi awal."
    )

    st.divider()

    st.markdown("**Model**")
    st.markdown("Logistic Regression")
    st.markdown("6 Indikator")


# =========================================================
# HEADER
# =========================================================

st.title("Student Dropout Risk Dashboard")
st.markdown(
    "Pantau indikator siswa dan identifikasi siswa yang memerlukan "
    "intervensi awal berdasarkan prediksi risiko dropout."
)


# =========================================================
# KPI CARDS
# =========================================================

total_students = len(df)
dropout_rate = df["Dropout"].mean() * 100
high_risk = (df["risk"] == "High Risk").sum()
medium_risk = (df["risk"] == "Medium Risk").sum()
low_risk = (df["risk"] == "Low Risk").sum()
students_at_risk = high_risk + medium_risk

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Students</div>
            <div class="metric-value">{total_students:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Actual Dropout Rate</div>
            <div class="metric-value">{dropout_rate:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Students At Risk</div>
            <div class="metric-value">{students_at_risk:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# =========================================================
# PERFORMANCE KPI CARDS
# =========================================================

st.subheader("Performance Overview")

performance = df[
    [
        "GPA",
        "Attendance_Rate",
        "Study_Hours_per_Day",
        "Stress_Index",
        "Travel_Time_Minutes",
        "Assignment_Delay_Days",
    ]
].mean()

perf1, perf2, perf3 = st.columns(3)

with perf1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average GPA</div>
            <div class="metric-value">{performance["GPA"]:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with perf2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average Attendance</div>
            <div class="metric-value">{performance["Attendance_Rate"]:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with perf3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average Study Hours</div>
            <div class="metric-value">{performance["Study_Hours_per_Day"]:.1f} hrs/day</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

perf4, perf5, perf6 = st.columns(3)

with perf4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average Stress Index</div>
            <div class="metric-value">{performance["Stress_Index"]:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with perf5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average Travel Time</div>
            <div class="metric-value">{performance["Travel_Time_Minutes"]:.1f} min</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with perf6:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average Assignment Delay</div>
            <div class="metric-value">{performance["Assignment_Delay_Days"]:.1f} days</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# =========================================================
# RISK SUMMARY CARDS
# =========================================================

risk_col1, risk_col2, risk_col3 = st.columns(3)

with risk_col1:
    st.markdown(
        f"""
        <div style="background:#FDECEF; border:1px solid #D45A6D;
                    border-radius:12px; padding:18px 20px;">
            <div style="color:#D45A6D; font-size:14px; font-weight:600;">
                High Risk Students
            </div>
            <div style="color:#292832; font-size:28px; font-weight:700; margin-top:6px;">
                {high_risk:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with risk_col2:
    st.markdown(
        f"""
        <div style="background:#FFF7E3; border:1px solid #D0A447;
                    border-radius:12px; padding:18px 20px;">
            <div style="color:#B18420; font-size:14px; font-weight:600;">
                Medium Risk Students
            </div>
            <div style="color:#292832; font-size:28px; font-weight:700; margin-top:6px;">
                {medium_risk:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with risk_col3:
    st.markdown(
        f"""
        <div style="background:#EAF7F0; border:1px solid #58A77E;
                    border-radius:12px; padding:18px 20px;">
            <div style="color:#3E8A65; font-size:14px; font-weight:600;">
                Low Risk Students
            </div>
            <div style="color:#292832; font-size:28px; font-weight:700; margin-top:6px;">
                {low_risk:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# =========================================================
# TABS
# =========================================================

tab_dashboard, tab_monitoring, tab_prediction = st.tabs(
    [
        "Dashboard",
        "Student Risk Monitoring",
        "Student Prediction"
    ]
)


# =========================================================
# DASHBOARD TAB
# =========================================================

with tab_dashboard:

    st.markdown("### Risk Level")

    st.markdown("**Filter Tingkat Risiko**")

    if "dashboard_risk_filter" not in st.session_state:
        st.session_state.dashboard_risk_filter = "All"

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        if st.button(
            "All",
            use_container_width=True,
            type="primary" if st.session_state.dashboard_risk_filter == "All" else "secondary",
            key="dashboard_filter_all"
        ):
            st.session_state.dashboard_risk_filter = "All"
            st.rerun()

    with filter_col2:
        if st.button(
            "High Risk",
            use_container_width=True,
            type="primary" if st.session_state.dashboard_risk_filter == "High Risk" else "secondary",
            key="dashboard_filter_high"
        ):
            st.session_state.dashboard_risk_filter = "High Risk"
            st.rerun()

    with filter_col3:
        if st.button(
            "Medium Risk",
            use_container_width=True,
            type="primary" if st.session_state.dashboard_risk_filter == "Medium Risk" else "secondary",
            key="dashboard_filter_medium"
        ):
            st.session_state.dashboard_risk_filter = "Medium Risk"
            st.rerun()

    with filter_col4:
        if st.button(
            "Low Risk",
            use_container_width=True,
            type="primary" if st.session_state.dashboard_risk_filter == "Low Risk" else "secondary",
            key="dashboard_filter_low"
        ):
            st.session_state.dashboard_risk_filter = "Low Risk"
            st.rerun()

    analysis_filter = st.session_state.dashboard_risk_filter

    if analysis_filter == "All":
        analysis_df = df.copy()
    else:
        analysis_df = df[df["risk"] == analysis_filter].copy()

    st.caption(
        f"Menampilkan {len(analysis_df):,} siswa dengan filter: {analysis_filter}."
    )

    # -----------------------------------------------------
    # DISTRIBUTION
    # -----------------------------------------------------

    st.markdown("## Distribution")

    distribution_features = [
        ("Risk Distribution", "risk", "Risk Level", "risk"),
        ("Dropout Probability Distribution", "dropout_probability", "Dropout Probability (%)", "hist"),
        ("GPA Distribution", "GPA", "GPA", "hist"),
        ("Attendance Distribution", "Attendance_Rate", "Attendance (%)", "hist"),
        ("Study Hours Distribution", "Study_Hours_per_Day", "Study Hours / Day", "hist"),
        ("Stress Index Distribution", "Stress_Index", "Stress Index", "hist"),
        ("Assignment Delay Distribution", "Assignment_Delay_Days", "Assignment Delay (Days)", "hist"),
    ]

    for i in range(0, len(distribution_features), 2):
        dist_col1, dist_col2 = st.columns(2)

        for col, item in zip(
            [dist_col1, dist_col2],
            distribution_features[i:i + 2]
        ):
            title, column, x_title, chart_type = item

            with col:
                st.subheader(title)

                if chart_type == "risk":
                    risk_counts = (
                        analysis_df["risk"]
                        .value_counts()
                        .reindex(
                            ["High Risk", "Medium Risk", "Low Risk"]
                        )
                        .fillna(0)
                        .reset_index()
                    )

                    risk_counts.columns = ["Risk", "Students"]

                    fig = px.pie(
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

                else:
                    fig = px.histogram(
                        analysis_df,
                        x=column,
                        nbins=20,
                        template="plotly_white",
                        color_discrete_sequence=["#806FC9"]
                    )

                fig.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#292832"),
                    margin=dict(t=10, b=10, l=10, r=10)
                )

                fig = apply_plotly_theme(fig)


                if chart_type == "risk":
                    fig.update_layout(
                        legend=dict(
                            font=dict(
                                color="#292832",
                                size=12
                            )
                        )
                    )

                    fig.update_traces(
                        textfont=dict(
                            color="#292832",
                            size=11
                        )
                    )
                else:
                    fig.update_traces(
                        textfont=dict(
                            color="#FFFFFF",
                            size=11
                        )
                    )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"dashboard_distribution_{column}"
                )

    # -----------------------------------------------------
    # DISTRIBUTION INSIGHT
    # -----------------------------------------------------

    risk_counts_insight = (
        analysis_df["risk"]
        .value_counts(normalize=True)
        .reindex(["High Risk", "Medium Risk", "Low Risk"])
        .fillna(0)
    )

    dominant_risk = risk_counts_insight.idxmax()
    dominant_risk_pct = risk_counts_insight.max() * 100

    avg_gpa = analysis_df["GPA"].mean()
    avg_attendance = analysis_df["Attendance_Rate"].mean()
    avg_study_hours = analysis_df["Study_Hours_per_Day"].mean()
    avg_stress = analysis_df["Stress_Index"].mean()
    avg_assignment_delay = analysis_df["Assignment_Delay_Days"].mean()

    st.markdown("### 💡 Insight")

    st.info(
        f"**Distribusi data siswa:** kategori risiko yang paling dominan adalah "
        f"**{dominant_risk}** ({dominant_risk_pct:.1f}%). "
        f"Secara umum, rata-rata siswa memiliki GPA **{avg_gpa:.2f}**, "
        f"tingkat kehadiran **{avg_attendance:.1f}%**, "
        f"study hours **{avg_study_hours:.1f} jam/hari**, "
        f"stress index **{avg_stress:.1f}**, dan assignment delay "
        f"**{avg_assignment_delay:.1f} hari**."
    )

    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    st.markdown("## Correlation")

    correlation_columns = [
        "GPA",
        "Stress_Index",
        "Attendance_Rate",
        "Study_Hours_per_Day",
        "Travel_Time_Minutes",
        "Assignment_Delay_Days",
        "Dropout"
    ]

    correlation_labels = [
        "GPA",
        "Stress Index",
        "Attendance",
        "Study Hours",
        "Travel Time",
        "Assignment Delay",
        "Dropout"
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

        fig_heatmap = apply_plotly_theme(fig_heatmap)

        fig_heatmap.update_traces(
            textfont=dict(
                color="#292832",
                size=11
            )
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
            key="dashboard_correlation_matrix"
        )

    with corr_col2:
        st.markdown("**Correlation with Dropout**")

        dropout_corr = (
            corr_matrix["Dropout"]
            .drop("Dropout")
            .reset_index()
        )

        dropout_corr.columns = ["Metric", "Correlation"]

        fig_corr = px.bar(
            dropout_corr.sort_values("Correlation"),
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
                gridcolor="#E2DEF2"
            ),
            yaxis=dict(
                title=""
            ),
            margin=dict(t=10, b=10, l=10, r=10)
        )

        fig_corr = apply_plotly_theme(fig_corr)

        fig_corr.update_traces(
            textfont=dict(
                color="#292832",
                size=11
            )
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True,
            key="dashboard_correlation_dropout"
        )
    
    # -----------------------------------------------------
    # CORRELATION INSIGHT
    # -----------------------------------------------------

    correlation_strength = (
        corr_matrix["Dropout"]
        .drop("Dropout")
        .abs()
        .sort_values(ascending=False)
    )

    strongest_metric = correlation_strength.index[0]
    strongest_value = corr_matrix.loc[strongest_metric, "Dropout"]

    direction = "positif" if strongest_value > 0 else "negatif"

    st.markdown("### 💡 Insight")

    st.info(
        f"Indikator dengan hubungan linear paling kuat terhadap **Dropout** "
        f"pada data yang ditampilkan adalah **{strongest_metric}**, "
        f"dengan koefisien korelasi **{strongest_value:.2f}** "
        f"({direction}). "
        f"Hubungan positif menunjukkan bahwa nilai indikator yang lebih tinggi "
        f"cenderung diikuti oleh nilai Dropout yang lebih tinggi, sedangkan "
        f"hubungan negatif menunjukkan kecenderungan sebaliknya."
    )

    
    # -----------------------------------------------------
    # RELATIONSHIPS WITH DROPOUT
    # -----------------------------------------------------

    st.markdown("## Relationship with Dropout Risk")

    relationship_features = [
        ("GPA vs Dropout Probability", "GPA", "GPA"),
        ("Stress Index vs Dropout Probability", "Stress_Index", "Stress Index"),
        ("Attendance vs Dropout Probability", "Attendance_Rate", "Attendance (%)"),
        ("Study Hours vs Dropout Probability", "Study_Hours_per_Day", "Study Hours / Day"),
        ("Travel Time vs Dropout Probability", "Travel_Time_Minutes", "Travel Time (Minutes)"),
        ("Assignment Delay vs Dropout Probability", "Assignment_Delay_Days", "Assignment Delay (Days)")
    ]

    for i in range(0, len(relationship_features), 2):
        rel_col1, rel_col2 = st.columns(2)

        for col, item in zip(
            [rel_col1, rel_col2],
            relationship_features[i:i + 2]
        ):
            title, x_column, x_title = item

            with col:
                st.markdown(f"**{title}**")

                fig_relationship = px.scatter(
                    analysis_df,
                    x=x_column,
                    y="dropout_probability",
                    template="plotly_white",
                    opacity=0.60,
                    labels={
                        x_column: x_title,
                        "dropout_probability": "Dropout Probability (%)"
                    },
                    trendline=None
                )

                fig_relationship.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#292832"),
                    xaxis=dict(
                        title=x_title,
                        gridcolor="#E2DEF2"
                    ),
                    yaxis=dict(
                        title="Dropout Probability (%)",
                        gridcolor="#E2DEF2"
                    ),
                    margin=dict(t=10, b=10, l=10, r=10)
                )

                fig_relationship = apply_plotly_theme(fig_relationship)

                st.plotly_chart(
                    fig_relationship,
                    use_container_width=True,
                    key=f"dashboard_relationship_{x_column}"
                )

    # -----------------------------------------------------
    # RELATIONSHIP INSIGHT
    # -----------------------------------------------------

    relationship_columns = [
        "GPA",
        "Stress_Index",
        "Attendance_Rate",
        "Study_Hours_per_Day",
        "Travel_Time_Minutes",
        "Assignment_Delay_Days"
    ]

    relationship_labels = {
        "GPA": "GPA",
        "Stress_Index": "Stress Index",
        "Attendance_Rate": "Attendance",
        "Study_Hours_per_Day": "Study Hours",
        "Travel_Time_Minutes": "Travel Time",
        "Assignment_Delay_Days": "Assignment Delay"
    }

    relationship_corr = (
        analysis_df[
            relationship_columns + ["dropout_probability"]
        ]
        .corr(method="spearman")["dropout_probability"]
        .drop("dropout_probability")
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
        "peningkatan probabilitas dropout"
        if strongest_relationship_value > 0
        else
        "nilai indikator yang lebih tinggi cenderung diikuti "
        "penurunan probabilitas dropout"
    )

    st.markdown("### 💡 Insight")

    st.info(
        f"Di antara indikator yang digunakan model, "
        f"**{relationship_labels[strongest_relationship]}** "
        f"menunjukkan pola hubungan monotonic paling kuat dengan "
        f"**Dropout Probability**, dengan korelasi Spearman "
        f"**{strongest_relationship_value:.2f}** "
        f"({relationship_direction}). "
        f"Artinya, pada data yang ditampilkan, "
        f"{relationship_interpretation}. "
        f"Temuan ini menggambarkan pola pada output model "
        f"dan bukan hubungan sebab-akibat."
    )

# =========================================================
# MONITORING TAB
# =========================================================

with tab_monitoring:

    st.markdown("## Student Risk Monitoring")

    st.markdown("**Filter Tingkat Risiko**")

    if "monitoring_risk_filter" not in st.session_state:
        st.session_state.monitoring_risk_filter = "All"

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        if st.button(
            "All",
            use_container_width=True,
            type="primary" if st.session_state.monitoring_risk_filter == "All" else "secondary",
            key="monitoring_filter_all"
        ):
            st.session_state.monitoring_risk_filter = "All"
            st.rerun()

    with filter_col2:
        if st.button(
            "High Risk",
            use_container_width=True,
            type="primary" if st.session_state.monitoring_risk_filter == "High Risk" else "secondary",
            key="monitoring_filter_high"
        ):
            st.session_state.monitoring_risk_filter = "High Risk"
            st.rerun()

    with filter_col3:
        if st.button(
            "Medium Risk",
            use_container_width=True,
            type="primary" if st.session_state.monitoring_risk_filter == "Medium Risk" else "secondary",
            key="monitoring_filter_medium"
        ):
            st.session_state.monitoring_risk_filter = "Medium Risk"
            st.rerun()

    with filter_col4:
        if st.button(
            "Low Risk",
            use_container_width=True,
            type="primary" if st.session_state.monitoring_risk_filter == "Low Risk" else "secondary",
            key="monitoring_filter_low"
        ):
            st.session_state.monitoring_risk_filter = "Low Risk"
            st.rerun()

    risk_filter = st.session_state.monitoring_risk_filter

    if risk_filter == "All":
        risk_students = df.copy()
    else:
        risk_students = df[
            df["risk"] == risk_filter
        ].copy()

    # Optional student search
    search_term = st.text_input(
        "Search Student ID",
        placeholder="Enter Student ID..."
    )

    if search_term.strip():
        risk_students = risk_students[
            risk_students["Student_ID"]
            .astype(str)
            .str.contains(search_term.strip(), case=False, na=False)
        ]

    display_df = risk_students[
        [
            "Student_ID",
            "GPA",
            "Attendance_Rate",
            "Study_Hours_per_Day",
            "Stress_Index",
            "dropout_probability",
            "Dropout",
            "risk"
        ]
    ].copy()

    display_df["dropout_probability"] = (
        display_df["dropout_probability"].round(2)
    )

    display_df["Dropout"] = display_df["Dropout"].map({
        0: "Not Dropout",
        1: "Dropout"
    })

    display_df.columns = [
        "Student ID",
        "GPA",
        "Attendance (%)",
        "Study Hours / Day",
        "Stress Index",
        "Dropout Probability (%)",
        "Actual Status",
        "Risk Level"
    ]

    page_size = 10
    total_rows = len(display_df)

    st.caption(f"Menampilkan {total_rows:,} siswa.")

    if total_rows == 0:
        st.info("No students found.")

    else:
        total_pages = (total_rows - 1) // page_size + 1

        if "monitoring_page_number" not in st.session_state:
            st.session_state.monitoring_page_number = 1

        page_key = f"{risk_filter}|{search_term.strip()}"

        if st.session_state.get("monitoring_filter_key") != page_key:
            st.session_state.monitoring_filter_key = page_key
            st.session_state.monitoring_page_number = 1

        st.session_state.monitoring_page_number = min(
            max(st.session_state.monitoring_page_number, 1),
            total_pages,
        )

        current_page = st.session_state.monitoring_page_number

        start_index = (current_page - 1) * page_size
        end_index = start_index + page_size

        page_df = display_df.iloc[start_index:end_index].copy()

        styled_table = (
            page_df.style
            .format({
                "GPA": "{:.2f}",
                "Attendance (%)": "{:.1f}",
                "Study Hours / Day": "{:.1f}",
                "Stress Index": "{:.1f}",
                "Dropout Probability (%)": "{:.2f}",
            })
            .set_properties(**{
                "background-color": "#FFFFFF",
                "color": "#292832",
                "border-color": "#E2DEF2",
            })
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#6F5BC3"),
                        ("color", "#FFFFFF"),
                        ("font-weight", "600"),
                        ("border-color", "#6F5BC3"),
                    ],
                },
            ])
        )

        st.dataframe(
            styled_table,
            use_container_width=True,
            hide_index=True
        )

        nav1, nav2, nav3 = st.columns([1, 2, 1])

        with nav1:
            if current_page > 1 and st.button(
                "← Previous",
                use_container_width=True,
                key="monitoring_previous",
            ):
                st.session_state.monitoring_page_number -= 1
                st.rerun()

        with nav2:
            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    color:#292832;
                    font-size:14px;
                    padding-top:8px;
                ">
                    Page <b>{current_page}</b> of <b>{total_pages}</b>
                    • Showing <b>{start_index + 1}</b>–<b>{min(end_index, total_rows)}</b>
                    of <b>{total_rows}</b> students
                </div>
                """,
                unsafe_allow_html=True,
            )

        with nav3:
            if current_page < total_pages and st.button(
                "Next →",
                use_container_width=True,
                key="monitoring_next",
            ):
                st.session_state.monitoring_page_number += 1
                st.rerun()

        st.download_button(
            label="Download Current Monitoring Data",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="edupulse_student_risk_monitoring.csv",
            mime="text/csv",
        )

# =========================================================
# PREDICTION TAB
# =========================================================

with tab_prediction:

    st.markdown("---")
    st.subheader("Student Prediction")
    st.markdown(
        "Enter the six indicators used by the EduPulse Lightweight "
        "Logistic Regression model to predict dropout probability."
    )

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            input_gpa = st.number_input(
                "GPA",
                min_value=0.0,
                max_value=4.0,
                value=2.50,
                step=0.01
            )

            input_stress = st.number_input(
                "Stress Index",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1
            )

        with col2:
            input_attendance = st.number_input(
                "Attendance Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0
            )

            input_study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.5
            )

        with col3:
            input_travel_time = st.number_input(
                "Travel Time (Minutes)",
                min_value=0.0,
                max_value=300.0,
                value=30.0,
                step=1.0
            )

            input_assignment_delay = st.number_input(
                "Assignment Delay (Days)",
                min_value=0.0,
                max_value=30.0,
                value=2.0,
                step=1.0
            )

            st.write("")

            predict_button = st.button(
                "Predict Student",
                use_container_width=True,
                type="primary"
            )

    if predict_button:

        input_data = {
            "GPA": input_gpa,
            "Stress_Index": input_stress,
            "Attendance_Rate": input_attendance,
            "Study_Hours_per_Day": input_study_hours,
            "Travel_Time_Minutes": input_travel_time,
            "Assignment_Delay_Days": input_assignment_delay
        }

        try:
            prediction = predict_dropout(input_data)

            probability = prediction["dropout_probability"]
            risk = prediction["risk_level"]
            probability_pct = probability * 100

            # =====================================================
            # PREDICTION RESULT
            # =====================================================

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Dropout Probability",
                    f"{probability_pct:.2f}%"
                )

            with result_col2:
                st.metric(
                    "Risk Level",
                    risk
                )

            # =====================================================
            # RISK THRESHOLD VISUAL
            # =====================================================

            medium_threshold = float(risk_thresholds["medium"])
            high_threshold = float(risk_thresholds["high"])

            probability_clamped = min(max(probability, 0.0), 1.0)

            probability_display = probability_clamped * 100
            medium_display = medium_threshold * 100
            high_display = high_threshold * 100

            st.markdown("### Risk Threshold")

            threshold_html = f"""
            <div style="margin-top:10px;">

            <div style="
            position:relative;
            height:24px;
            background:linear-gradient(
            to right,
            #58A77E 0%,
            #58A77E {medium_display}%,
            #D0A447 {medium_display}%,
            #D0A447 {high_display}%,
            #D45A6D {high_display}%,
            #D45A6D 100%
            );
            border-radius:12px;
            overflow:visible;
            ">

            <div style="
            position:absolute;
            left:{probability_display}%;
            top:-7px;
            transform:translateX(-50%);
            width:4px;
            height:38px;
            background:#292832;
            border-radius:2px;
            "></div>

            </div>

            <div style="
            position:relative;
            height:28px;
            margin-top:8px;
            font-size:12px;
            color:#4A4655;
            ">

            <div style="
            position:absolute;
            left:0%;
            ">
            0%
            </div>

            <div style="
            position:absolute;
            left:{medium_display}%;
            transform:translateX(-50%);
            ">
            {medium_display:.0f}%
            </div>

            <div style="
            position:absolute;
            left:{high_display}%;
            transform:translateX(-50%);
            ">
            {high_display:.0f}%
            </div>

            <div style="
            position:absolute;
            right:0%;
            ">
            100%
            </div>

            </div>

            <div style="
            display:flex;
            justify-content:space-between;
            font-size:13px;
            color:#4A4655;
            margin-top:4px;
            ">
            <span>Low Risk</span>
            <span>Medium Risk</span>
            <span>High Risk</span>
            </div>

            <div style="
            text-align:center;
            margin-top:14px;
            font-size:14px;
            color:#292832;
            ">
            <b>Current Probability: {probability_pct:.2f}%</b>
            </div>

            </div>
            """

            st.markdown(
                threshold_html,
                unsafe_allow_html=True
            )

            # =====================================================
            # DYNAMIC RECOMMENDATION
            # =====================================================

            recommendations = []

            # Academic performance
            if input_gpa < 2.00:
                recommendations.append(
                    "GPA siswa berada di bawah 2.00. Prioritaskan pendampingan akademik dan evaluasi mata kuliah yang menjadi kendala."
                )
            elif input_gpa < 2.50:
                recommendations.append(
                    "GPA siswa masih relatif rendah. Tingkatkan konsistensi belajar dan lakukan pemantauan perkembangan akademik."
                )

            # Attendance
            if input_attendance < 75:
                recommendations.append(
                    f"Tingkat kehadiran rendah ({input_attendance:.0f}%). Prioritaskan peningkatan kehadiran dan tindak lanjut terhadap penyebab ketidakhadiran."
                )
            elif input_attendance < 85:
                recommendations.append(
                    f"Tingkat kehadiran cukup rendah ({input_attendance:.0f}%). Dorong siswa untuk menjaga kehadiran lebih konsisten."
                )

            # Study hours
            if input_study_hours < 2:
                recommendations.append(
                    f"Waktu belajar harian rendah ({input_study_hours:.1f} jam). Bantu siswa membangun jadwal belajar yang lebih konsisten."
                )
            elif input_study_hours < 3:
                recommendations.append(
                    f"Waktu belajar harian masih terbatas ({input_study_hours:.1f} jam). Tingkatkan durasi belajar secara bertahap."
                )

            # Stress
            if input_stress >= 8:
                recommendations.append(
                    f"Tingkat stres tinggi ({input_stress:.1f}/10). Pertimbangkan dukungan dari guru, pembimbing akademik, atau layanan konseling."
                )
            elif input_stress >= 6:
                recommendations.append(
                    f"Tingkat stres cukup tinggi ({input_stress:.1f}/10). Pantau kondisi siswa dan bantu mengelola beban akademik."
                )

            # Assignment delay
            if input_assignment_delay >= 7:
                recommendations.append(
                    f"Keterlambatan tugas cukup tinggi ({input_assignment_delay:.0f} hari). Susun prioritas tugas dan jadwal penyelesaian yang lebih teratur."
                )
            elif input_assignment_delay >= 3:
                recommendations.append(
                    f"Masih terdapat keterlambatan tugas ({input_assignment_delay:.0f} hari). Dorong penyelesaian tugas lebih tepat waktu."
                )

            # Travel time
            if input_travel_time >= 90:
                recommendations.append(
                    f"Waktu perjalanan cukup tinggi ({input_travel_time:.0f} menit). Evaluasi dampaknya terhadap waktu belajar, kehadiran, dan kelelahan siswa."
                )

            # Risk-based recommendation
            if risk == "High Risk":
                st.error(
                    "Siswa teridentifikasi dalam kategori Risiko Tinggi dan memerlukan intervensi awal."
                )
            elif risk == "Medium Risk":
                st.warning(
                    "Siswa teridentifikasi dalam kategori Risiko Sedang dan perlu dipantau."
                )
            else:
                st.success(
                    "Siswa teridentifikasi dalam kategori Risiko Rendah."
                )

            st.markdown("### Rekomendasi")

            if recommendations:
                recommendation_html = "<ul>"

                for recommendation in recommendations:
                    recommendation_html += f"<li>{recommendation}</li>"

                recommendation_html += "</ul>"

                st.markdown(
                    f"""
                    <div class="prediction-recommendation">
                        {recommendation_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="prediction-recommendation">
                        <ul>
                            <li>Pertahankan kebiasaan belajar dan tingkat kehadiran yang baik.</li>
                            <li>Terus selesaikan tugas tepat waktu.</li>
                            <li>Pertahankan tingkat stres pada kondisi yang terkendali.</li>
                            <li>Lanjutkan pemantauan perkembangan siswa secara berkala.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as exc:
            st.error("Prediction gagal diproses.")
            st.exception(exc)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption(
    "EduPulse • Student Early Warning System • "

)
