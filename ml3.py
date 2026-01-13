# ==========================================
# HOUSEHOLD WATER CONSUMPTION PREDICTION SYSTEM
# ==========================================

# --- IMPORTS ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Household Water Consumption Dashboard",
    page_icon="💧",
    layout="wide"
)

# --- STYLING & BACKGROUND ---
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
            background-color: rgba(255,255,255,0.95);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        h1, h2, h3 { color: #004e92; }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# --- HEADER ---
st.title("💧 Household Water Consumption Dashboard")
st.markdown("### 🤖 Machine Learning Analysis Using Water Access Indicators\n"
            "with PAIP Treated Water Supply Comparison")
st.caption("📊 DOSM | 🚰 PAIP (Pahang) | 🤖 Machine Learning")

# --- DATA LOADING FUNCTIONS ---
@st.cache_data
def load_csv_data():
    # UPDATE THESE PATHS TO YOUR ACTUAL FILE LOCATIONS
    # Use relative paths if files are in the same folder, e.g., "cleaned_water_dataset.csv"
    dosm = pd.read_csv("cleaned_water_dataset.csv") 
    paip = pd.read_csv("Treated Overall By Year.csv")
    return dosm, paip

@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load("mlr_model.pkl")
        model_columns = joblib.load("model_columns.pkl")
        return model, model_columns
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run Part 1 in Jupyter Notebook to generate .pkl files.")
        return None, None

# Load Data
try:
    dosm, paip = load_csv_data()
    model, model_columns = load_ml_model()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --- PRE-PROCESSING ---
# 1. Clean DOSM Data
dosm["Year"] = pd.to_numeric(dosm["Year"], errors="coerce")
dosm["WaterConsumptionMLD"] = pd.to_numeric(dosm["WaterConsumptionMLD"], errors="coerce")
dosm = dosm.dropna()
dosm = dosm[dosm["State"] != "Malaysia"] # Remove aggregate

# 2. Clean PAIP Data
paip["Year"] = pd.to_numeric(paip["Year"], errors="coerce")
paip["Treated"] = paip["Treated"].astype(str).str.replace(",", "", regex=True)
paip["Treated"] = pd.to_numeric(paip["Treated"], errors="coerce")
paip = paip.dropna()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Data Filters")

# State Selection
state_list = sorted(dosm["State"].unique())
selected_state = st.sidebar.selectbox("Select State", state_list)

# Strata Selection
selected_strata = st.sidebar.multiselect(
    "Select Strata", 
    options=sorted(dosm["Strata"].unique()), 
    default=sorted(dosm["Strata"].unique())
)

# Year Selection
min_year, max_year = int(dosm["Year"].min()), int(dosm["Year"].max())
selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Filter Logic
filtered_data = dosm[
    (dosm["State"] == selected_state) &
    (dosm["Strata"].isin(selected_strata)) &
    (dosm["Year"].between(selected_years[0], selected_years[1]))
]

# --- MAIN METRICS ROW ---
col1, col2, col3 = st.columns(3)
col1.metric("📍 Selected Region", selected_state)
col2.metric("💧 Avg Consumption", f"{filtered_data['WaterConsumptionMLD'].mean():.2f} MLD")
col3.metric("📅 Data Points", filtered_data.shape[0])

# --- TABS FOR NAVIGATION ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA", 
    "📈 Trends", 
    "🤖 Model Evaluation", 
    "🚰 Supply vs Demand", 
    "🔮 Predictor (Deploy)", 
    "ℹ️ Project Info"
])

# ===============================
# TAB 1: EDA
# ===============================
with tab1:
    st.subheader("📊 Descriptive Analysis")
    
    # 1. Bar Chart
    fig_bar = px.bar(filtered_data, x="Year", y="WaterConsumptionMLD", color="Strata", barmode="group",
                     title="Average Household Water Consumption by Strata")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 2. Box Plot & Scatter Plot (Side-by-Side)
    col_a, col_b = st.columns(2)
    with col_a:
        fig_box = px.box(filtered_data, x="Strata", y="WaterConsumptionMLD", color="Strata",
                         title="Urban vs Rural Water Consumption Distribution")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_b:
        fig_scatter = px.scatter(filtered_data, x="WaterAccessPercent", y="WaterConsumptionMLD", color="Strata",
                                 title="Water Access vs Consumption")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    # 3. Correlation Matrix 
    # Select only numeric columns for correlation
    numeric_filtered = filtered_data.select_dtypes(include="number")
    
    if not numeric_filtered.empty:
        corr = numeric_filtered.corr().round(2) 
        
        # Create Annotated Heatmap
        fig_heatmap = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="Viridis",
            showscale=True
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown("Correlation matrix showing relationships between variables.")
        
# ===============================
# TAB 2: TRENDS
# ===============================
with tab2:
    st.subheader("📈 Consumption Trend Over Time")
    fig_line = px.line(filtered_data, x="Year", y="WaterConsumptionMLD", color="Strata", markers=True,
                       title=f"Household Water Consumption Trend")
    st.plotly_chart(fig_line, use_container_width=True)

# ===============================
# TAB 3: ML RESULTS
# ===============================
with tab3:
    st.subheader("🤖 Machine Learning Model Performance")
    st.markdown("""
    In this study, several machine learning models were developed and evaluated
    to predict household water consumption for **urban and rural areas**.
    The performance of each model was assessed using **RMSE** and **R² score**.
    """)
    
    results_data = {
        "Model": ["Multiple Linear Regression (MLR)", "SVR", "Random Forest", "XGBoost", "ANN"],
        "Urban RMSE": [4.48, 42.50, 5.47, 8.44, 43.30],
        "Urban R²": [0.9998, 0.9855, 0.9998, 0.9994, 0.9850],
        "Rural RMSE": [15.29, 35.64, 22.26, 17.34, 56.29],
        "Rural R²": [0.9977, 0.9875, 0.9951, 0.9970, 0.9689]
    }
    st.subheader("📊 Model Performance Comparison")
    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
    
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
# TAB 4: SUPPLY VS DEMAND (PAHANG)
# ===============================
with tab4:
    st.subheader("🚰 Treated Water Supply vs Household Demand")
    
    if selected_state != "Pahang":
        st.warning("ℹ️ **PAIP treated water supply data is only available for Pahang.**  \n"
            "Supply–demand comparison is therefore restricted to Pahang to "
            "avoid misleading interpretation.  \n\n"
            "For other states, the dashboard focuses on household water "
            "consumption analysis only.")
    else:
        # Process PAIP Data (Convert m3/year to MLD)
        paip_agg = paip.groupby("Year")["Treated"].mean().reset_index()
        paip_agg["Supply_MLD"] = paip_agg["Treated"] / 365000  # Conversion Factor
        
        # Process DOSM Data (Overall)
        dosm_pahang = dosm[(dosm["State"] == "Pahang") & (dosm["Strata"] == "overall")]
        dosm_agg = dosm_pahang.groupby("Year")["WaterConsumptionMLD"].mean().reset_index()
        
        # Merge
        merged_df = pd.merge(dosm_agg, paip_agg, on="Year")
        
        # Plot
        fig_val = px.line(merged_df, x="Year", y=["WaterConsumptionMLD", "Supply_MLD"], markers=True,
                          labels={"value": "Volume (MLD)", "variable": "Metric"},
                          title="Household Demand (DOSM) vs Treated Supply (PAIP)")
        st.plotly_chart(fig_val, use_container_width=True)
        st.info("Household Water Demand vs PAIP Treated Water Supply (Pahang)")

        st.caption(
            "📌 PAIP data originally reported in m³/year and converted to MLD "
            "to ensure fair comparison with household consumption data."
        )

# ===============================
# TAB 5: PREDICTOR 
# ===============================
with tab5:
    st.header("🔮 Future Consumption Predictor")
    st.markdown("### Interactive Planning Tool")
    st.write("Use this tool to forecast water demand for future years or hypothetical scenarios.")
    
    if model is None:
        st.error("Model not loaded. Please ensure .pkl files are in the directory.")
    else:
        # --- INPUT FORM ---
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                input_year = st.number_input("Prediction Year", min_value=2024, max_value=2035, value=2026)
                input_state = st.selectbox("State Region", state_list)
            with c2:
                input_strata = st.selectbox("Strata Type", ["Urban", "Rural"])
                input_access = st.slider("Water Access (%)", 80.0, 100.0, 98.5)
            
            submitted = st.form_submit_button("🚀 Generate Prediction")
            
        # --- PREDICTION LOGIC ---
        if submitted:
            # 1. Create Dataframe with 0s (Safe initialization)
            encoded_df = pd.DataFrame(0, index=[0], columns=model_columns)

            # 2. Fill Numerical Features
            encoded_df['WaterAccessPercent'] = input_access
# ✅ FIX #1: Ensure 'Year' is populated if the model expects it
            if 'Year' in model_columns:
                encoded_df['Year'] = input_year
                
            # Fill Interaction Term (if it exists)
            if 'Year_Access_Interaction' in model_columns:
                encoded_df['Year_Access_Interaction'] = input_year * input_access
            
            # 3. Fill Categorical Features
            
            # State One-Hot Encoding
            state_col = f"State_{input_state}"
            if state_col in model_columns:
                encoded_df[state_col] = 1
            
            # ✅ FIX #2: Correct Strata Logic (Handles both Label "0/1" and One-Hot)
            if "Strata_encoded" in model_columns:
                # This matches your Notebook's Label Encoding (Urban=1, Rural=0)
                if input_strata == "Urban":
                    encoded_df["Strata_encoded"] = 1
                else:
                    encoded_df["Strata_encoded"] = 0
            elif "Strata_Urban" in model_columns:
                # Fallback for One-Hot Encoding
                 if input_strata == "Urban":
                     encoded_df["Strata_Urban"] = 1
            elif "Strata_Rural" in model_columns:
                 if input_strata == "Rural":
                     encoded_df["Strata_Rural"] = 1
            
            # 4. Predict
            try:
                prediction = model.predict(encoded_df)[0]
                
                # Display Result
                st.markdown("---")
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.metric(label="Predicted Consumption", value=f"{prediction:,.2f} MLD")
                with col_res2:
                    st.info(f"💡 Planning Insight: In {input_year}, if {input_state} ({input_strata}) has {input_access}% water access, the estimated domestic demand is {prediction:.0f} Million Liters/Day.")
            except Exception as e:
                st.error(f"Prediction Error: {e}. Please check feature column names.")
# ===============================
# TAB 6: ABOUT
# ===============================
with tab6:
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


# --- FOOTER ---
st.markdown("---")
st.markdown("<center>Machine Learning Group Project | Universiti Malaysia Pahang</center>", unsafe_allow_html=True)




