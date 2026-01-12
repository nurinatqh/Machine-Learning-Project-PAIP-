# ===============================
# IMPORTS
# ===============================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Household Water Consumption Dashboard",
    page_icon="💧",
    layout="wide"
)

# ===============================
# BACKGROUND (WATER DROP THEME)
# ===============================
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1557683316-973673baf926");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .block-container {
            background-color: rgba(255,255,255,0.92);
            padding: 2rem;
            border-radius: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# ===============================
# TITLE
# ===============================
st.title("💧 Household Water Consumption Dashboard")
st.subheader(
    "Machine Learning Analysis using Water Access Indicators\n"
    "with PAIP Treated Water Supply Comparison"
)
st.caption("📊 DOSM | 🚰 PAIP (Pahang) | 🤖 Machine Learning")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    dosm = pd.read_csv(r"C:\Users\USER\OneDrive\ML\group project\cleaned_water_dataset.csv")
    paip = pd.read_csv(r"C:\Users\USER\OneDrive\ML\group project\Treated Overall By Year.csv")
    return dosm, paip

dosm, paip = load_data()

# ===============================
# DATA CLEANING
# ===============================
dosm["Year"] = pd.to_numeric(dosm["Year"], errors="coerce")
dosm["WaterConsumptionMLD"] = pd.to_numeric(
    dosm["WaterConsumptionMLD"], errors="coerce"
)

paip["Year"] = pd.to_numeric(paip["Year"], errors="coerce")
paip["Treated"] = (
    paip["Treated"]
    .astype(str)
    .str.replace(",", "", regex=True)
)
paip["Treated"] = pd.to_numeric(paip["Treated"], errors="coerce")

dosm = dosm.dropna()
paip = paip.dropna()

# Remove national aggregate
dosm = dosm[dosm["State"] != "Malaysia"]

# ===============================
# SIDEBAR FILTERS
# ===============================
st.sidebar.header("🔍 Filters")

selected_state = st.sidebar.selectbox(
    "Select State",
    sorted(dosm["State"].unique())
)

selected_strata = st.sidebar.multiselect(
    "Select Strata",
    options=sorted(dosm["Strata"].unique()),
    default=sorted(dosm["Strata"].unique())
)

year_min, year_max = int(dosm["Year"].min()), int(dosm["Year"].max())
selected_years = st.sidebar.slider(
    "Select Year Range",
    year_min,
    year_max,
    (year_min, year_max)
)

# ===============================
# FILTER DATA
# ===============================
filtered = dosm[
    (dosm["State"] == selected_state) &
    (dosm["Strata"].isin(selected_strata)) &
    (dosm["Year"].between(selected_years[0], selected_years[1]))
]

# ===============================
# KPI METRICS
# ===============================
st.markdown("### 📌 Key Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("📍 State", selected_state)
col2.metric(
    "💧 Avg Consumption (MLD)",
    f"{filtered['WaterConsumptionMLD'].mean():.2f}"
)
col3.metric("📅 Years Covered", filtered["Year"].nunique())

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA",
    "📈 Trends",
    "🤖 ML Results",
    "🚰 Supply vs Demand (PAIP)",
    "ℹ️ About"
])

# ===============================
# TAB 1: EDA
# ===============================
with tab1:
    st.header("📊 Descriptive Analysis")

    fig1 = px.bar(
        filtered,
        x="Year",
        y="WaterConsumptionMLD",
        color="Strata",
        barmode="group",
        labels={"WaterConsumptionMLD": "Water Consumption (MLD)"},
        title="Average Household Water Consumption by Strata"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Urban vs Rural Comparison
    fig_urb_rur = px.box(
        filtered, x="Strata", y="WaterConsumptionMLD", color="Strata",
        title="Urban vs Rural Water Consumption Distribution"
    )
    st.plotly_chart(fig_urb_rur, use_container_width=True)

    # Water Access vs Consumption
    fig_scatter = px.scatter(
        filtered,
        x="WaterAccessPercent", y="WaterConsumptionMLD",
        color="Strata", title="Water Access vs Consumption"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation Matrix (Optional)
    numeric_filtered = filtered.select_dtypes(include="number")
    corr = numeric_filtered.corr()
    fig_heatmap = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="Viridis"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("Correlation matrix showing relationships between variables.")

# ===============================
# TAB 2: TREND
# ===============================
with tab2:
    st.header("📈 Consumption Trend Over Time")

    fig2 = px.line(
        filtered,
        x="Year",
        y="WaterConsumptionMLD",
        color="Strata",
        markers=True,
        labels={"WaterConsumptionMLD": "Water Consumption (MLD)"},
        title="Household Water Consumption Trend"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ===============================
# TAB 3: ML RESULTS (MODEL COMPARISON)
# ===============================
with tab3:
    st.header("🤖 Machine Learning Model Comparison")

    st.markdown("""
    In this study, several machine learning models were developed and evaluated
    to predict household water consumption for **urban and rural areas**.
    The performance of each model was assessed using **RMSE** and **R² score**.
    """)

    # ===============================
    # MODEL COMPARISON TABLE
    # ===============================
    ml_results = pd.DataFrame({
        "Model": [
            "Multiple Linear Regression",
            "Support Vector Regression (SVR)",
            "Random Forest",
            "XGBoost",
            "Artificial Neural Network (ANN)"
        ],
        "Urban RMSE": [4.48, 42.50, 5.47, 8.44, 43.30],
        "Urban R²": [0.9998, 0.9855, 0.9998, 0.9994, 0.9850],
        "Rural RMSE": [15.29, 35.64, 22.26, 17.34, 56.29],
        "Rural R²": [0.9977, 0.9875, 0.9951, 0.9970, 0.9689]
    })

    st.subheader("📊 Model Performance Comparison")
    st.dataframe(
        ml_results,
        use_container_width=True,
        hide_index=True
    )

    # ===============================
    # BEST MODEL HIGHLIGHT
    # ===============================
    st.markdown("### ✅ Final Model Selected")

    st.success("""
    **Multiple Linear Regression (MLR)** was selected as the final model
    for deployment.
    """)

    st.markdown("""
    **Justification:**
    - Achieved the **lowest RMSE** for both urban and rural datasets
    - Recorded the **highest R² values**, indicating excellent explanatory power
    - Provides **high interpretability**, allowing policymakers to understand
      how water access indicators influence consumption patterns

    Due to its strong predictive performance and transparency, MLR is the
    most suitable model for supporting **sustainable water resource planning**.
    """)

# ===============================
# TAB 4: SUPPLY VS DEMAND (AUTO LOCK PAHANG)
# ===============================
with tab4:
    st.header("🚰 Treated Water Supply vs Household Demand")

    if selected_state != "Pahang":
        st.info(
            "ℹ️ **PAIP treated water supply data is only available for Pahang.**  \n"
            "Supply–demand comparison is therefore restricted to Pahang to "
            "avoid misleading interpretation.  \n\n"
            "For other states, the dashboard focuses on household water "
            "consumption analysis only."
        )

    else:
        # Convert PAIP m³/year → MLD
        paip_year = paip.groupby("Year")["Treated"].mean().reset_index()
        paip_year["Treated_MLD"] = paip_year["Treated"] / 365_000

        dosm_year = (
            dosm[
                (dosm["State"] == "Pahang") &
                (dosm["Strata"] == "overall")
            ]
            .groupby("Year")["WaterConsumptionMLD"]
            .mean()
            .reset_index()
        )

        merged = pd.merge(dosm_year, paip_year, on="Year", how="inner")

        fig3 = px.line(
            merged,
            x="Year",
            y=["WaterConsumptionMLD", "Treated_MLD"],
            markers=True,
            labels={
                "value": "Volume (MLD)",
                "variable": "Indicator"
            },
            title="Household Water Demand vs PAIP Treated Water Supply (Pahang)"
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "📌 PAIP data originally reported in m³/year and converted to MLD "
            "to ensure fair comparison with household consumption data."
        )

# ===============================
# TAB 5: ABOUT
# ===============================
with tab5:
    st.header("ℹ️ About This Project")

    st.markdown("""
    ### 📘 Project Description
    This project analyses household water consumption patterns in Malaysia
    using data from the Department of Statistics Malaysia (DOSM).
    Machine learning models are developed to predict domestic water consumption
    based on water access indicators, with a focus on understanding differences
    between urban and rural areas.

    Treated water supply data from Pengurusan Air Pahang Berhad (PAIP) is used
    as a case study to compare predicted household water demand with actual
    treated water supply trends, supporting demand–supply assessment.

    ### 🎯 Project Objectives
    1. To analyse household water consumption patterns across Malaysian states
       and between urban and rural areas.
    2. To examine the relationship between water access indicators and household
       water consumption.
    3. To develop and evaluate machine learning models for predicting household
       water consumption.
    4. To compare predicted consumption trends with treated water supply data
       from PAIP using an interactive dashboard.

    ### 🌍 Project Impact
    This dashboard provides a data-driven decision support tool that helps
    visualise household water demand trends and their alignment with treated
    water supply. The findings can support more informed and sustainable water
    resource planning and policy discussions, particularly in understanding
    urban–rural consumption dynamics.
    """)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("💧 Streamlit Dashboard | Machine Learning Group Project")
