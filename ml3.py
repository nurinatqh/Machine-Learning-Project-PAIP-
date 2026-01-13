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
    page_title="Water Consumption AI Dashboard",
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
st.title("💧 Smart Water Consumption Prediction System")
st.markdown("### 🤖 Machine Learning Analytics & Forecasting for Sustainable Planning")
st.caption("Data Source: DOSM (2020-2023) | Validation: PAIP | Model: Multiple Linear Regression")

# --- DATA LOADING FUNCTIONS ---
@st.cache_data
def load_csv_data():
    # UPDATE THESE PATHS TO YOUR ACTUAL FILE LOCATIONS
    # Use relative paths if files are in the same folder, e.g., "cleaned_water_dataset.csv"
    dosm = pd.read_csv(r"C:\Users\USER\OneDrive\ML\group project\cleaned_water_dataset.csv") 
    paip = pd.read_csv(r"C:\Users\USER\OneDrive\ML\group project\Treated Overall By Year.csv")
    return dosm, paip

@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load(r"C:\Users\USER\OneDrive\ML\group project\mlr_model.pkl")
        model_columns = joblib.load(r"C:\Users\USER\OneDrive\ML\group project\model_columns.pkl")
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
    "🔮 Predictor (Deploy)",  # <--- THIS IS THE NEW CRITICAL TAB
    "ℹ️ Project Info"
])

# ===============================
# TAB 1: EDA
# ===============================
with tab1:
    st.subheader(f"Exploratory Data Analysis: {selected_state}")
    
    # Bar Chart
    fig_bar = px.bar(filtered_data, x="Year", y="WaterConsumptionMLD", color="Strata", barmode="group",
                     title="Annual Consumption by Strata")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Box Plot
    col_a, col_b = st.columns(2)
    with col_a:
        fig_box = px.box(filtered_data, x="Strata", y="WaterConsumptionMLD", color="Strata",
                         title="Consumption Distribution (Urban vs Rural)")
        st.plotly_chart(fig_box, use_container_width=True)
    with col_b:
        fig_scatter = px.scatter(filtered_data, x="WaterAccessPercent", y="WaterConsumptionMLD", color="Strata",
                                 title="Impact of Water Access on Consumption")
        st.plotly_chart(fig_scatter, use_container_width=True)

# ===============================
# TAB 2: TRENDS
# ===============================
with tab2:
    st.subheader("📈 Historical Consumption Trends")
    fig_line = px.line(filtered_data, x="Year", y="WaterConsumptionMLD", color="Strata", markers=True,
                       title=f"Time Series Analysis for {selected_state}")
    st.plotly_chart(fig_line, use_container_width=True)

# ===============================
# TAB 3: ML RESULTS
# ===============================
with tab3:
    st.subheader("🤖 Machine Learning Model Performance")
    st.markdown("Comparative analysis of 5 algorithms to determine the best predictor.")
    
    results_data = {
        "Model": ["Multiple Linear Regression (MLR)", "SVR", "Random Forest", "XGBoost", "ANN"],
        "Urban RMSE": [4.48, 42.50, 5.47, 8.44, 43.30],
        "Urban R²": [0.9998, 0.9855, 0.9998, 0.9994, 0.9850],
        "Rural RMSE": [15.29, 35.64, 22.26, 17.34, 56.29],
        "Rural R²": [0.9977, 0.9875, 0.9951, 0.9970, 0.9689]
    }
    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
    
    st.success("✅ **Selected Model: Multiple Linear Regression (MLR)**\n\nChosen for its superior accuracy (Lowest RMSE) and high interpretability for policy planning.")

# ===============================
# TAB 4: SUPPLY VS DEMAND (PAHANG)
# ===============================
with tab4:
    st.subheader("🚰 Supply vs Demand Validation (Pahang Case Study)")
    
    if selected_state != "Pahang":
        st.warning("⚠️ Supply data is strictly confidential and only available for Pahang. Please select 'Pahang' in the sidebar to view this validation.")
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
        st.info("The gap between the lines represents Non-Revenue Water (NRW) and Industrial Usage.")

# ===============================
# TAB 5: PREDICTOR (DEPLOYMENT)
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
            # 1. Create Raw Input Data
            # Note: We must recreate the exact features used in training
            input_data = pd.DataFrame({
                'Year': [input_year],
                'WaterAccessPercent': [input_access],
                'AccessAdjustedConsumption': [0], # Placeholder if not used directly
                'AccessChange': [0], # Placeholder
                'Year_Access_Interaction': [input_year * input_access] # Interaction Term
            })
            
            # 2. One-Hot Encoding (Manual Matching)
            # Create a dataframe with all 0s for the columns the model expects
            encoded_df = pd.DataFrame(0, index=[0], columns=model_columns)
            
            # Fill Numeric Values
            encoded_df['WaterAccessPercent'] = input_access
            # If your model used 'Year', fill it. If you dropped 'Year' in code, ignore it.
            # Based on your previous code, you dropped 'Year' but kept 'Year_Access_Interaction'
            if 'Year_Access_Interaction' in model_columns:
                encoded_df['Year_Access_Interaction'] = input_year * input_access
            
            # Fill Categorical Values (State & Strata)
            # The column names usually look like 'State_Johor' or 'Strata_Urban'
            state_col = f"State_{input_state}"
            strata_col = f"Strata_encoded_{input_strata}" # OR just 'Strata_Urban' check your pickle columns
            # NOTE: I am guessing the column name format based on standard pandas.get_dummies.
            # Ideally, print(model_columns) in your notebook to check exact names.
            
            # Try to find the matching column for State
            if state_col in model_columns:
                encoded_df[state_col] = 1
            
            # Try to find matching column for Strata
            # If you used 'Strata_encoded' in training, you might need to map Urban->1, Rural->0
            # OR if you used get_dummies on Strata, look for Strata_Urban
            if "Strata_Urban" in model_columns and input_strata == "Urban":
                 encoded_df["Strata_Urban"] = 1
            elif "Strata_Rural" in model_columns and input_strata == "Rural":
                 encoded_df["Strata_Rural"] = 1
            
            # 3. Predict
            try:
                prediction = model.predict(encoded_df)[0]
                
                # 4. Display Result
                st.markdown("---")
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.metric(label="Predicted Consumption", value=f"{prediction:,.2f} MLD")
                with col_res2:
                    st.info(f"💡 Planning Insight: In {input_year}, if {input_state} ({input_strata}) has {input_access}% water access, the estimated domestic demand is {prediction:.0f} Million Liters/Day.")
            except Exception as e:
                st.error(f"Prediction Error: {e}. Please check feature column names.")

# ===============================
# TAB 6: PROJECT INFO
# ===============================
with tab6:
    st.header("ℹ️ Project Background")
    st.markdown("""
    **Objectives:**
    1. Analyse consumption patterns (DOSM).
    2. Correlate water access with usage.
    3. Develop ML models (MLR, RF, XGBoost).
    4. Validate with PAIP data and deploy for planning.
    
    **Team:** [Your Group Name]
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("<center>Machine Learning Group Project | Universiti Malaysia Pahang</center>", unsafe_allow_html=True)