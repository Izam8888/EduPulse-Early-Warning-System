
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
        color: #6F6D78;
    }

    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E2DEF2;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
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
    st.markdown("# 📊 EduPulse")
    st.markdown("### Student Early Warning System")
    st.divider()
    st.markdown("**Dashboard**")
    st.markdown(
        "Monitor student performance and identify students "
        "who may need early intervention."
    )

    st.divider()
    st.caption("Main Model")
    st.caption("Logistic Regression")
    st.caption("Lightweight • 6 Inputs")


# =========================================================
# HEADER
# =========================================================

st.title("Student Dropout Risk Dashboard")
st.markdown(
    "Monitor student indicators and identify students who may "
    "need early intervention based on predicted dropout risk."
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

performance = df[
    [
        "GPA",
        "Attendance_Rate",
        "Study_Hours_per_Day",
        "Stress_Index"
    ]
].mean()

performance_col1, performance_col2, performance_col3, performance_col4 = st.columns(4)

with performance_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average GPA</div>
            <div class="metric-value">{performance['GPA']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with performance_col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average Attendance</div>
            <div class="metric-value">{performance['Attendance_Rate']:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with performance_col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average Study Hours</div>
            <div class="metric-value">{performance['Study_Hours_per_Day']:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with performance_col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average Stress Index</div>
            <div class="metric-value">{performance['Stress_Index']:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True
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

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"dashboard_distribution_{column}"
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

        st.plotly_chart(
            fig_corr,
            use_container_width=True,
            key="dashboard_correlation_dropout"
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

                st.plotly_chart(
                    fig_relationship,
                    use_container_width=True,
                    key=f"dashboard_relationship_{x_column}"
                )

    # -----------------------------------------------------
    # PERFORMANCE OVERVIEW
    # -----------------------------------------------------

    st.markdown("## Performance Overview")

    overview = analysis_df[
        [
            "GPA",
            "Attendance_Rate",
            "Study_Hours_per_Day",
            "Stress_Index",
            "Travel_Time_Minutes",
            "Assignment_Delay_Days"
        ]
    ].mean()

    overview_df = overview.reset_index()
    overview_df.columns = ["Metric", "Average"]

    overview_df["Metric"] = [
        "GPA",
        "Attendance",
        "Study Hours",
        "Stress Index",
        "Travel Time",
        "Assignment Delay"
    ]

    overview_df["Average"] = overview_df["Average"].round(2)

    fig_performance = px.bar(
        overview_df,
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
            gridcolor="#E2DEF2"
        ),
        yaxis=dict(
            title="Average",
            gridcolor="#E2DEF2"
        ),
        margin=dict(t=20, b=20, l=20, r=20)
    )

    st.plotly_chart(
        fig_performance,
        use_container_width=True,
        key="dashboard_performance_overview"
    )


# =========================================================
# MONITORING TAB
# =========================================================

with tab_monitoring:

    st.markdown("## Student Risk Monitoring")

    risk_filter = st.selectbox(
        "Filter Risk Level",
        ["All", "High Risk", "Medium Risk", "Low Risk"],
        key="monitoring_risk_filter"
    )

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

    st.caption(
        f"Menampilkan {len(display_df):,} siswa."
    )

    if display_df.empty:
        st.info("No students found.")
    else:
        styled_table = (
            display_df
            .style
            .format({
                "GPA": "{:.2f}",
                "Attendance (%)": "{:.1f}",
                "Study Hours / Day": "{:.1f}",
                "Stress Index": "{:.1f}",
                "Dropout Probability (%)": "{:.2f}"
            })
            .set_properties(**{
                "background-color": "#FFFFFF",
                "color": "#292832",
                "border-color": "#E2DEF2"
            })
        )

        st.table(styled_table)

        st.download_button(
            label="Download Current Monitoring Data",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="edupulse_student_risk_monitoring.csv",
            mime="text/csv"
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

            st.progress(
                min(max(probability, 0.0), 1.0),
                text=f"Dropout probability: {probability_pct:.2f}%"
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
                            <li>Prioritize immediate academic monitoring and follow-up.</li>
                            <li>Review attendance, study habits, stress, and delayed assignments.</li>
                            <li>Coordinate early support with the teacher or academic advisor.</li>
                            <li>Monitor the student's progress after intervention.</li>
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
                            <li>Monitor academic and behavioral indicators regularly.</li>
                            <li>Strengthen study consistency and attendance.</li>
                            <li>Address assignment delays before they become persistent.</li>
                            <li>Consider additional academic support when needed.</li>
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
                            <li>Maintain current study and attendance habits.</li>
                            <li>Continue completing assignments on time.</li>
                            <li>Keep stress at a manageable level.</li>
                            <li>Continue monitoring progress to maintain the current condition.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
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
    "Lightweight Logistic Regression"
)
