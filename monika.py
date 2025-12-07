import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Healthcare Analysis for Chronic Disease Risk Stratification", layout="wide")
st.title("🏥 Healthcare Analysis for Chronic Disease Risk Stratification")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    file_path = "Chronic.csv"
    df = pd.read_csv(file_path)
    
    # Standardize column names to lowercase and handle duplicates
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Check for duplicate columns and remove them
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Convert numeric columns to proper format
    numeric_columns = ['diabetes_risk_score', 'cvd_risk_score', 'copd_risk_score', 'age', 'bmi', 'glucose']
    for col in numeric_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                st.warning(f"Could not convert column '{col}' to numeric: {e}")
    
    # ===============================
    # UPDATED: Create risk bins & flags with new thresholds
    # ===============================
    
    # DIABETES: Low < 0.40, Medium 0.40-0.70, High ≥ 0.70
    df['diabetes_risk_bin'] = pd.cut(
        df['diabetes_risk_score'],
        bins=[-0.01, 0.40, 0.70, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk'],
        include_lowest=True
    )
    df['has_diabetes'] = (df['diabetes_risk_score'] >= 0.70).astype(int)

    # CVD: Low < 0.10, Medium 0.10-0.20, High ≥ 0.20
    if 'cvd_risk_score' in df.columns:
        df['cvd_risk_bin'] = pd.cut(
            df['cvd_risk_score'],
            bins=[-0.01, 0.10, 0.20, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            include_lowest=True
        )
        df['has_cvd'] = (df['cvd_risk_score'] >= 0.20).astype(int)

    # COPD: Low < 0.35, Medium 0.35-0.65, High ≥ 0.65
    if 'copd_risk_score' in df.columns:
        df['copd_risk_bin'] = pd.cut(
            df['copd_risk_score'],
            bins=[-0.01, 0.50, 0.65, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            include_lowest=True
        )
        df['has_copd'] = (df['copd_risk_score'] >= 0.65).astype(int)
    
    # Additional columns with error handling
    # Clean age column
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 120], 
                                 labels=['0-18', '19-30', '31-45', '46-60', '60+'])
    
    # Clean BMI column
    if 'bmi' in df.columns:
        # Convert to numeric, handling any non-numeric values
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        # Create BMI categories
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                                    labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    return df

df = load_data()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔍 Global Filters")

# Age filter
age_min, age_max = int(df['age'].min()), int(df['age'].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

# Gender filter
gender_options = df['gender'].unique().tolist()
selected_gender = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

# County filter
county_options = sorted(df['county'].dropna().unique().tolist())
selected_counties = st.sidebar.multiselect("County", county_options, default=county_options)

# Apply filters
filtered_df = df[
    (df['age'] >= age_range[0]) & 
    (df['age'] <= age_range[1]) &
    (df['gender'].isin(selected_gender)) &
    (df['county'].isin(selected_counties))
].copy()

st.sidebar.success(f"✅ {len(filtered_df)} patients")
st.sidebar.info(f"📊 Out of {len(df)} total")

# -------------------------------
# Tabs
# -------------------------------
overview_tab, diabetes_tab, cvd_tab, copd_tab, Comorbidity_tab, checking, risk_calculator = st.tabs([
    "📊 Overview", "🩺 Diabetes", "🫀 CVD", "🫁 COPD", "📋 Comorbidity", "🔍 Individual Patient Risk", "🧮 Risk Calculator"
])

# ===============================
# TAB 1: OVERVIEW
# ===============================
with overview_tab:
    st.header("📊 Overview")
    st.info(f"🔍 Showing {len(filtered_df)} of {len(df)} total patients")
    st.markdown("---")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_patients = len(filtered_df)
    high_risk_diabetes = len(filtered_df[filtered_df['diabetes_risk_bin'] == 'High Risk'])
    high_risk_cvd = len(filtered_df[filtered_df['cvd_risk_bin'] == 'High Risk'])
    high_risk_copd = len(filtered_df[filtered_df['copd_risk_bin'] == 'High Risk'])
    
    col1.metric("Total Patients", total_patients)
    col2.metric("🩺 High Risk Diabetes", high_risk_diabetes, delta=f"{(high_risk_diabetes/total_patients*100):.1f}%")
    col3.metric("🫀 High Risk CVD", high_risk_cvd, delta=f"{(high_risk_cvd/total_patients*100):.1f}%")
    col4.metric("🫁 High Risk COPD", high_risk_copd, delta=f"{(high_risk_copd/total_patients*100):.1f}%")
    
    st.markdown("---")

    # Risk Distribution Bar Chart - RIGHT AFTER KPIs
    disease_data = []
    for disease, risk_col in {
        'Diabetes': 'diabetes_risk_bin',
        'CVD': 'cvd_risk_bin',
        'COPD': 'copd_risk_bin'
    }.items():
        if risk_col in filtered_df.columns:
            counts = filtered_df[risk_col].value_counts().reset_index()
            counts.columns = ['Risk Level', 'Count']
            counts['Disease'] = disease
            disease_data.append(counts)

    if disease_data:
        combined_risk = pd.concat(disease_data)
        fig_bar = px.bar(
            combined_risk,
            x='Risk Level',
            y='Count',
            color='Risk Level',
            facet_col='Disease',
            text='Count',
            category_orders={'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk']},
            color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'},
            title='Risk Distribution Across Diseases'
        )
        fig_bar.update_traces(textposition='outside')
        
        # Remove "Disease=" prefix from facet titles
        fig_bar.for_each_annotation(lambda a: a.update(text=a.text.replace("Disease=", "")))
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")

# demographics And risk distribution

with overview_tab:
    st.header("📊 Overview: Demographics & Risk Distribution")
    st.markdown("---")

    # Convert birthdate to birth_year
    if 'birthdate' in df.columns:
        df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
        df['birth_year'] = df['birthdate'].dt.year
    else:
        st.error("❌ 'birthdate' column not found in dataset.")
        st.stop()

    # Disease filter
    selected_disease = st.radio(
        "Select Disease to Display in Overview:",
        options=['Diabetes', 'CVD', 'COPD'],
        horizontal=True
    )

    # Filter dataset
    if selected_disease == "Diabetes" and "has_diabetes" in df.columns:
        disease_df = df[df["has_diabetes"] == 1].copy()
        risk_col = "diabetes_risk_score"
        color_scheme = ['#e74c3c', '#c0392b']
        line_color = "#e74c3c"
    elif selected_disease == "CVD" and "has_cvd" in df.columns:
        disease_df = df[df["has_cvd"] == 1].copy()
        risk_col = "cvd_risk_score"
        color_scheme = ['#3498db', '#2980b9']
        line_color = "#3498db"
    elif selected_disease == "COPD" and "has_copd" in df.columns:
        disease_df = df[df["has_copd"] == 1].copy()
        risk_col = "copd_risk_score"
        color_scheme = ['#f39c12', '#e67e22']
        line_color = "#f39c12"
    else:
        st.warning(f"No records found for {selected_disease}.")
        st.stop()

    # ==============================
    # Gender Distribution (Pie Chart)
    # ==============================
    if "gender" in disease_df.columns and disease_df["gender"].notna().any():
        gender_counts = disease_df["gender"].value_counts()
        fig_gender = px.pie(
            names=gender_counts.index,
            values=gender_counts.values,
            color_discrete_sequence=color_scheme,
            title=f"Gender Distribution for {selected_disease} Patients"
        )
        fig_gender.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_gender, use_container_width=True)
    else:
        st.info(f"No gender data available for {selected_disease} patients.")

    # ==============================
    # Race Distribution (Bar Chart)
    # ==============================
    if "race" in disease_df.columns and disease_df["race"].notna().any():
        race_counts = disease_df["race"].value_counts().head(10).reset_index()
        race_counts.columns = ["Race", "Count"]
        fig_race = px.bar(
            race_counts,
            x="Race",
            y="Count",
            text="Count",
            color="Count",
            color_continuous_scale=color_scheme,
            title=f"Race Distribution for {selected_disease} Patients"
        )
        fig_race.update_traces(textposition='outside')
        st.plotly_chart(fig_race, use_container_width=True)
    else:
        st.info(f"No race data available for {selected_disease} patients.")

    # -----------------------------
    # Average Risk Score by Birth Year
    # -----------------------------
    st.subheader(f"📈 Average {selected_disease} Risk Score by Birth Year")

    valid_df = disease_df[(disease_df['birth_year'].notna()) & (disease_df[risk_col].notna())].copy()
    if not valid_df.empty:
        avg_risk = (
            valid_df.groupby('birth_year')[risk_col]
            .mean()
            .reset_index()
            .rename(columns={risk_col: "Average Risk Score"})
        )

        fig_line = px.line(
            avg_risk,
            x="birth_year",
            y="Average Risk Score",
            title=f"Average {selected_disease} Risk Score by Birth Year",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=[line_color]
        )
        fig_line.update_layout(
            xaxis_title="Birth Year",
            yaxis_title="Average Risk Score",
            hovermode='x unified',
            template="plotly_white",
            height=520
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning(f"No valid birth year or risk score data for {selected_disease}.")

# ===============================
# TAB 2: DIABETES
# ===============================

with diabetes_tab:
    st.header("🩺 Diabetes Insights")

    # -----------------------------
    # KPI Cards (before filters)
    # -----------------------------
    total_patients = len(filtered_df)
    diabetes_patients = filtered_df[filtered_df['has_diabetes'] == 1].copy()

    col1, col2 = st.columns(2)
    col1.metric(
        "Diabetic Patients", 
        len(diabetes_patients),
        delta=f"{len(diabetes_patients)/total_patients*100:.1f}%"
    )
    col2.metric(
        "Non-Diabetic Patients",
        total_patients - len(diabetes_patients),
        delta=f"{(total_patients - len(diabetes_patients))/total_patients*100:.1f}%"
    )

    st.markdown("---")
    # -----------------------------
    # Filters
    # -----------------------------
    # Gender filter
    gender_options = ['All'] + sorted(filtered_df['gender'].dropna().unique().tolist())
    selected_gender = st.selectbox("Select Gender:", options=gender_options, index=0)

    # County filter
    county_options = ['All'] + sorted(filtered_df['county'].dropna().unique().tolist())
    selected_county = st.selectbox("Select County:", options=county_options, index=0)

    # Risk level filter
    risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
    selected_risk = st.multiselect("Select Risk Levels:", options=risk_levels, default=risk_levels)

    # -----------------------------
    # Filter dataset based on selections
    # -----------------------------
    if selected_gender != 'All':
        diabetes_patients = diabetes_patients[diabetes_patients['gender'] == selected_gender]
    if selected_county != 'All':
        diabetes_patients = diabetes_patients[diabetes_patients['county'] == selected_county]
    if selected_risk:
        diabetes_patients = diabetes_patients[diabetes_patients['diabetes_risk_bin'].isin(selected_risk)]
    else:
        diabetes_patients = diabetes_patients.iloc[0:0]
# -----------------------------
    # Data Table
    # -----------------------------
    st.subheader("📋 Diabetic Patient Details")
    display_columns = ['id', 'age', 'gender', 'bmi', 'diabetes_risk_score', 'diabetes_risk_bin', 'county']
    display_columns = [col for col in display_columns if col in diabetes_patients.columns]

    if len(diabetes_patients) > 0:
        rename_dict = {
            'id': 'ID',
            'age': 'Age',
            'gender': 'Gender',
            'bmi': 'BMI',
            'diabetes_risk_score': 'Risk Score',
            'diabetes_risk_bin': 'Risk Bin',
            'county': 'County'
        }
        st.dataframe(diabetes_patients[display_columns].rename(columns=rename_dict), use_container_width=True, height=400)
    else:
        st.info("No diabetic patients found with selected filters.")

    st.markdown("---")


with diabetes_tab:
    ## Diabetes patients by risk level
    st.subheader("Diabetes Patients by Risk Level")

    # Use all patients who have a risk score (instead of filtering has_diabetes)
    patients_with_risk = df[df['diabetes_risk_score'].notna()].copy()

    # Count by risk bin
    risk_counts = patients_with_risk['diabetes_risk_bin'].value_counts().reindex(
        ['Low Risk', 'Medium Risk', 'High Risk'], fill_value=0
    ).reset_index()
    risk_counts.columns = ['Risk Bin', 'Count']

    # Sort for proper order
    risk_counts = risk_counts.sort_values(
        'Risk Bin',
        key=lambda x: x.map({'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2})
    )

    # Plot
    fig = px.bar(
        risk_counts,
        x='Risk Bin',
        y='Count',
        text='Count',
        color='Risk Bin',
        color_discrete_map={
            'Low Risk': '#2ecc71',    # Green
            'Medium Risk': '#f39c12', # Orange
            'High Risk': '#e74c3c'    # Red
        }
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # -----------------------------
    # Top 10 Counties by Diabetic Patients
    # -----------------------------
    st.subheader("Top 10 Counties by Diabetic Patients")
    if len(diabetes_patients) > 0 and 'county' in diabetes_patients.columns:
        top_counties = diabetes_patients['county'].value_counts().head(10).reset_index()
        top_counties.columns = ['County', 'Diabetic Patients']

        fig_county = px.bar(
            top_counties,
            x='County',
            y='Diabetic Patients',
            text='Diabetic Patients',
            color='Diabetic Patients',
            color_continuous_scale='Reds',
        )
        fig_county.update_traces(textposition='outside')
        st.plotly_chart(fig_county, use_container_width=True)
    else:
        st.info("No county data available for the selected filters.")

    st.markdown("---")

# ===============================
# TAB 3: CVD
# ===============================

### CVD Analysis
with cvd_tab:
    st.header("🫀 CVD Insights")

    # -----------------------------
    # KPI Cards (before filters)
    # -----------------------------
    total_patients = len(filtered_df)
    cvd_patients = filtered_df[filtered_df['has_cvd'] == 1].copy()

    col1, col2 = st.columns(2)
    col1.metric(
        "High-Risk CVD Patients", 
        len(cvd_patients[cvd_patients['cvd_risk_bin'] == 'High Risk']),
        delta=f"{len(cvd_patients[cvd_patients['cvd_risk_bin'] == 'High Risk'])/total_patients*100:.1f}%"
    )
    col2.metric(
        "Non-CVD Patients",
        total_patients - len(cvd_patients),
        delta=f"{(total_patients - len(cvd_patients))/total_patients*100:.1f}%"
    )

    st.markdown("---")

    # -----------------------------
    # Filters
    # -----------------------------
    # Gender filter
    gender_options = ['All'] + sorted(filtered_df['gender'].dropna().unique().tolist())
    selected_gender = st.selectbox("Select Gender:", options=gender_options, index=0, key='cvd_gender')

    # County filter
    county_options = ['All'] + sorted(filtered_df['county'].dropna().unique().tolist())
    selected_county = st.selectbox("Select County:", options=county_options, index=0, key='cvd_county')

    # Risk level filter
    risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
    selected_risk = st.multiselect("Select Risk Levels:", options=risk_levels, default=risk_levels, key='cvd_risk')

    # -----------------------------
    # Filter dataset based on selections
    # -----------------------------
    if selected_gender != 'All':
        cvd_patients = cvd_patients[cvd_patients['gender'] == selected_gender]
    if selected_county != 'All':
        cvd_patients = cvd_patients[cvd_patients['county'] == selected_county]
    if selected_risk:
        cvd_patients = cvd_patients[cvd_patients['cvd_risk_bin'].isin(selected_risk)]
    else:
        cvd_patients = cvd_patients.iloc[0:0]

    # -----------------------------
    # Data Table
    # -----------------------------
    st.subheader("📋 CVD Patient Details")
    display_columns = ['id', 'age', 'gender', 'bmi', 'cvd_risk_score', 'cvd_risk_bin', 'county']
    display_columns = [col for col in display_columns if col in cvd_patients.columns]

    if len(cvd_patients) > 0:
        rename_dict = {
            'id': 'ID',
            'age': 'Age',
            'gender': 'Gender',
            'bmi': 'BMI',
            'cvd_risk_score': 'Risk Score',
            'cvd_risk_bin': 'Risk Bin',
            'county': 'County'
        }
        st.dataframe(cvd_patients[display_columns].rename(columns=rename_dict), use_container_width=True, height=400)
    else:
        st.info("No CVD patients found with selected filters.")

    st.markdown("---")

    # -----------------------------
    # CVD Patients by Risk Level
    # -----------------------------
    st.subheader("CVD Patients by Risk Level")
    patients_with_risk = filtered_df[filtered_df['cvd_risk_score'].notna()].copy()
    risk_counts = patients_with_risk['cvd_risk_bin'].value_counts().reindex(risk_levels, fill_value=0).reset_index()
    risk_counts.columns = ['Risk Bin', 'Count']
    risk_counts = risk_counts.sort_values('Risk Bin', key=lambda x: x.map({'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}))

    fig = px.bar(
        risk_counts,
        x='Risk Bin',
        y='Count',
        text='Count',
        color='Risk Bin',
        color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'}
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # -----------------------------
    # Top 10 Counties by CVD Patients
    # -----------------------------
    st.subheader("Top 10 Counties by CVD Patients")
    if len(cvd_patients) > 0 and 'county' in cvd_patients.columns:
        top_counties = cvd_patients['county'].value_counts().head(10).reset_index()
        top_counties.columns = ['County', 'CVD Patients']

        fig_county = px.bar(
            top_counties,
            x='County',
            y='CVD Patients',
            text='CVD Patients',
            color='County',  # optional: color by county
            color_discrete_sequence=px.colors.sequential.Blues[::-1],
        )
        fig_county.update_traces(textposition='outside')
        st.plotly_chart(fig_county, use_container_width=True)
    else:
        st.info("No county data available for the selected filters.")

    st.markdown("---")

# ===============================
# TAB 4: COPD
# ===============================

with copd_tab:
    st.header("🫁 COPD Insights")
# -----------------------------
    # KPI Cards
    # -----------------------------
    total_filtered = len(filtered_df)
    copd_patients_all = filtered_df[filtered_df['has_copd'] == 1]
    non_copd_patients = filtered_df[filtered_df['has_copd'] == 0]
    high_risk_copd = copd_patients_all[copd_patients_all['copd_risk_bin'] == 'High Risk']

    col1, col2 = st.columns(2)
    col1.metric(
        "High-Risk COPD Patients", 
        len(high_risk_copd),
        delta=f"{(len(high_risk_copd)/total_filtered*100):.1f}%"
    )
    col2.metric(
        "Non-COPD Patients",
        len(non_copd_patients),
        delta=f"{(len(non_copd_patients)/total_filtered*100):.1f}%"
    )

    st.markdown("---")

    # -----------------------------
    # Filters
    # -----------------------------
    gender_options = ['All'] + sorted(filtered_df['gender'].dropna().unique().tolist())
    selected_gender = st.selectbox("Select Gender:", options=gender_options, index=0, key="copd_gender")

    county_options = ['All'] + sorted(filtered_df['county'].dropna().unique().tolist())
    selected_county = st.selectbox("Select County:", options=county_options, index=0, key="copd_county")

    risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
    selected_risk = st.multiselect("Select Risk Levels:", options=risk_levels, default=risk_levels, key="copd_risk")

    # -----------------------------
    # Filter dataset
    # -----------------------------
    copd_patients = filtered_df[filtered_df['has_copd'] == 1].copy()

    if selected_gender != 'All':
        copd_patients = copd_patients[copd_patients['gender'] == selected_gender]
    if selected_county != 'All':
        copd_patients = copd_patients[copd_patients['county'] == selected_county]
    if selected_risk:
        copd_patients = copd_patients[copd_patients['copd_risk_bin'].isin(selected_risk)]
    else:
        copd_patients = copd_patients.iloc[0:0]

    # -----------------------------
    # Patient Table
    # -----------------------------
    st.subheader("📋 COPD Patient Details")
    display_columns = ['id', 'age', 'gender', 'bmi', 'copd_risk_score', 'copd_risk_bin', 'county']
    display_columns = [col for col in display_columns if col in copd_patients.columns]

    if len(copd_patients) > 0:
        rename_dict = {
            'id': 'ID',
            'age': 'Age',
            'gender': 'Gender',
            'bmi': 'BMI',
            'copd_risk_score': 'Risk Score',
            'copd_risk_bin': 'Risk Bin',
            'county': 'County'
        }
        st.dataframe(copd_patients[display_columns].rename(columns=rename_dict), use_container_width=True, height=400)
    else:
        st.info("No COPD patients found with selected filters.")

# -----------------------------
    # COPD Patients by Risk Level
    # -----------------------------
    st.subheader("COPD Patients by Risk Level")
    patients_with_risk = filtered_df[filtered_df['copd_risk_score'].notna()].copy()
    risk_counts = patients_with_risk['copd_risk_bin'].value_counts().reindex(risk_levels, fill_value=0).reset_index()
    risk_counts.columns = ['Risk Bin', 'Count']
    risk_counts = risk_counts.sort_values('Risk Bin', key=lambda x: x.map({'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}))

    fig = px.bar(
        risk_counts,
        x='Risk Bin',
        y='Count',
        text='Count',
        color='Risk Bin',
        color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'}
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    with copd_tab:

    # -----------------------------
    # Top 10 Counties by COPD Patients (Bar Chart)
    # -----------------------------
      all_copd_patients = filtered_df[filtered_df['has_copd'] == 1]  # All COPD patients
    st.subheader("Top 10 Counties by COPD Patients")
    
    if len(all_copd_patients) > 0 and 'county' in all_copd_patients.columns:
        top_counties = all_copd_patients['county'].value_counts().head(10).reset_index()
        top_counties.columns = ['County', 'COPD Patients']

        fig_county = px.bar(
            top_counties,
            x='County',
            y='COPD Patients',
            text='COPD Patients',
            color='COPD Patients',
            color_continuous_scale=px.colors.sequential.Oranges,
        )
        fig_county.update_traces(textposition='outside')
        st.plotly_chart(fig_county, use_container_width=True)
    else:
        st.info("No county data available for COPD patients.")

    st.markdown("---")
# ===============================
# TAB 5: COMORBIDITY ANALYSIS
# ===============================
with Comorbidity_tab:
    st.header("📋 Disease Comorbidity Analysis")
    st.markdown("---")

    has_diabetes_count = len(filtered_df[filtered_df['has_diabetes'] == 1])
    has_cvd_count = len(filtered_df[filtered_df['has_cvd'] == 1])
    has_copd_count = len(filtered_df[filtered_df['has_copd'] == 1])

    min_patients_threshold = 10

    if (has_diabetes_count >= min_patients_threshold and 
        has_cvd_count >= min_patients_threshold and 
        has_copd_count >= min_patients_threshold):
        analysis_mode = "three_diseases"
    elif has_diabetes_count >= min_patients_threshold and has_cvd_count >= min_patients_threshold:
        st.warning("⚠ Limited COPD data — analyzing Diabetes & CVD only.")
        analysis_mode = "two_diseases"
    else:
        st.error("❌ Insufficient data for comorbidity analysis")
        analysis_mode = "insufficient"

    # -----------------------------
    # Three-Disease Analysis
    # -----------------------------
    if analysis_mode == "three_diseases":
        st.markdown("---")
        
        # Create disease combination codes
        filtered_df['Disease_Combo'] = (
            filtered_df['has_diabetes'].astype(str) + '-' +
            filtered_df['has_cvd'].astype(str) + '-' +
            filtered_df['has_copd'].astype(str)
        )

        # Keep only desired 4 combinations
        combo_labels = {
            '0-0-0': 'No Disease',
            '1-0-1': 'Diabetes + COPD',
            '0-1-1': 'CVD + COPD',
            '1-1-0': 'Diabetes + CVD'
        }

        filtered_df = filtered_df[filtered_df['Disease_Combo'].isin(combo_labels.keys())]
        filtered_df['Disease_Label'] = filtered_df['Disease_Combo'].map(combo_labels)

        # Count patients per combination
        combo_counts = (
            filtered_df['Disease_Label']
            .value_counts()
            .reindex(combo_labels.values(), fill_value=0)
            .reset_index()
        )
        combo_counts.columns = ['Disease Combination', 'Patient Count']
        combo_counts.insert(0, 'Rank', range(1, len(combo_counts) + 1))

        # Display table
        st.subheader("📋 Selected Disease Combinations Summary")
        st.dataframe(combo_counts, use_container_width=True, hide_index=True, height=250)

        st.markdown("---")

        # Pie chart
        st.subheader("Disease Co-occurrence Distribution")
        fig_pie = px.pie(
            combo_counts,
            values='Patient Count',
            names='Disease Combination',
            color_discrete_sequence=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=500)
        st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------
    # Two-Disease Analysis (Diabetes + CVD)
    # -----------------------------
    elif analysis_mode == "two_diseases":
        st.markdown("---")

        filtered_df['Disease_Combo_2'] = (
            filtered_df['has_diabetes'].astype(str) + '-' +
            filtered_df['has_cvd'].astype(str)
        )

        combo_labels_2 = {
            '0-0': 'No Disease',
            '1-1': 'Diabetes + CVD'
        }

        filtered_df = filtered_df[filtered_df['Disease_Combo_2'].isin(combo_labels_2.keys())]
        filtered_df['Disease_Label_2'] = filtered_df['Disease_Combo_2'].map(combo_labels_2)

        combo_counts_2 = (
            filtered_df['Disease_Label_2']
            .value_counts()
            .reindex(combo_labels_2.values(), fill_value=0)
            .reset_index()
        )
        combo_counts_2.columns = ['Disease Combination', 'Patient Count']
        combo_counts_2.insert(0, 'Rank', range(1, len(combo_counts_2) + 1))

        st.subheader("📋 Diabetes & CVD Comorbidity Summary")
        st.dataframe(combo_counts_2, use_container_width=True, hide_index=True, height=250)

        st.markdown("---")

        # Pie chart
        st.subheader("🥧 Co-occurrence Breakdown")
        fig_pie_2 = px.pie(
            combo_counts_2,
            values='Patient Count',
            names='Disease Combination',
            color_discrete_sequence=['#e74c3c', '#3498db']
        )
        fig_pie_2.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_2.update_layout(height=500)
        st.plotly_chart(fig_pie_2, use_container_width=True)

    # -----------------------------
    # Insufficient Data
    # -----------------------------
    else:
        st.info("📊 Insufficient patient data for meaningful comorbidity analysis.")
        st.write(f"- Diabetes patients: {has_diabetes_count}")
        st.write(f"- CVD patients: {has_cvd_count}")
        st.write(f"- COPD patients: {has_copd_count}")
        st.write(f"\nMinimum {min_patients_threshold} patients per disease required for analysis.")


# ===============================
# TAB 6: INDIVIDUAL PATIENT RISK
# ===============================
with checking:
    st.header("🔍 Individual Patient Risk Assessment")
    st.markdown("### Search for a specific patient to view their detailed health profile")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        patient_ids = sorted(df['id'].unique().tolist())
        selected_patient_id = st.selectbox(
            "🔍 Select Patient ID:",
            options=patient_ids,
            help="Choose a patient ID to view their detailed risk profile"
        )
    
    with col2:
        st.metric("Total Patients", len(patient_ids))
    
    patient = df[df['id'] == selected_patient_id].iloc[0]
    
    st.markdown("---")
    
    # ========================================
    # PATIENT BASIC INFO
    # ========================================
    st.subheader("👤 Patient Demographics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    
    col1.metric("Age", f"{patient.get('age', 0)} years")
    col2.metric("Gender", patient.get('gender', 'N/A'))
    col3.metric("County", patient.get('county', 'N/A'))
    col4.metric("Ethnicity", patient.get('ethnicity', 'N/A'))
    
    st.markdown("---")
    
    st.subheader("🏥 Overall Health Status")
    
    disease_count = (
        patient.get('has_diabetes', 0) + 
        patient.get('has_cvd', 0) + 
        patient.get('has_copd', 0)
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if disease_count == 0:
            st.success("✅ HEALTHY")
        elif disease_count == 1:
            st.warning("⚠️ SINGLE DISEASE")
        elif disease_count == 2:
            st.error("🔴 DUAL COMORBIDITY")
        else:
            st.error("🚨 TRIPLE COMORBIDITY")
    
    with col2:
        st.metric("Active Conditions", disease_count, help="Number of diagnosed chronic diseases")
    
    with col3:
        st.metric("BMI", f"{patient.get('bmi', 0):.1f}", help="Body Mass Index")
    
    st.markdown("---")
    
    st.subheader("📊 Detailed Risk Analysis by Disease")
    
    col1, col2, col3 = st.columns(3)
    
    # ========== DIABETES ==========
    with col1:
        st.markdown("### 🩺 Diabetes")
        
        diabetes_score = patient.get('diabetes_risk_score', 0)
        diabetes_risk_bin = patient.get('diabetes_risk_bin', 'Low Risk')
        
        gauge_color = '#2ecc71'
        if diabetes_risk_bin == 'High Risk':
            gauge_color = '#e74c3c'
        elif diabetes_risk_bin == 'Medium Risk':
            gauge_color = '#f1c40f'
        
        fig_diab = go.Figure(go.Indicator(
            mode="gauge+number",
            value=diabetes_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 16}},
            number={'suffix': "%", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': gauge_color, 'thickness': 0.75},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': diabetes_score * 100
                }
            }
        ))
        fig_diab.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_diab, use_container_width=True)
        
        if diabetes_risk_bin == 'High Risk':
            st.error(f"Risk Level: 🔴 {diabetes_risk_bin}")
        elif diabetes_risk_bin == 'Medium Risk':
            st.warning(f"Risk Level: 🟡 {diabetes_risk_bin}")
        else:
            st.success(f"Risk Level: 🟢 {diabetes_risk_bin}")
        
        st.metric("Glucose", f"{patient.get('glucose', 0):.0f} mg/dL")
        if 'hemoglobin a1c/hemoglobin.total in blood' in patient.index:
            st.metric("HbA1c", f"{patient.get('hemoglobin a1c/hemoglobin.total in blood', 0):.2f}%")
        
        if patient.get('has_diabetes', 0) == 1:
            st.error("Status: Diabetic")
        else:
            st.success("Status: Non-Diabetic")
    
    # ========== CVD ==========
    with col2:
        st.markdown("### 🫀 CVD")
        
        cvd_score = patient.get('cvd_risk_score', 0)
        cvd_risk_bin = patient.get('cvd_risk_bin', 'Low Risk')
        
        gauge_color = '#2ecc71'
        if cvd_risk_bin == 'High Risk':
            gauge_color = '#e74c3c'
        elif cvd_risk_bin == 'Medium Risk':
            gauge_color = '#f1c40f'
        
        fig_cvd = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cvd_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 16}},
            number={'suffix': "%", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': gauge_color, 'thickness': 0.75},
                'steps': [
                    {'range': [0, 10], 'color': "lightgreen"},
                    {'range': [10, 20], 'color': "lightyellow"},
                    {'range': [20, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': cvd_score * 100
                }
            }
        ))
        fig_cvd.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cvd, use_container_width=True)
        
        if cvd_risk_bin == 'High Risk':
            st.error(f"Risk Level: 🔴 {cvd_risk_bin}")
        elif cvd_risk_bin == 'Medium Risk':
            st.warning(f"Risk Level: 🟡 {cvd_risk_bin}")
        else:
            st.success(f"Risk Level: 🟢 {cvd_risk_bin}")
        
        if 'systolic blood pressure' in patient.index:
            st.metric("Systolic BP", f"{patient.get('systolic blood pressure', 0):.0f} mmHg")
        if 'diastolic blood pressure' in patient.index:
            st.metric("Diastolic BP", f"{patient.get('diastolic blood pressure', 0):.0f} mmHg")
        
        if patient.get('has_cvd', 0) == 1:
            st.error("Status: Has CVD")
        else:
            st.success("Status: No CVD")
    
    # ========== COPD ==========
    with col3:
        st.markdown("### 🫁 COPD")
        
        copd_score = patient.get('copd_risk_score', 0)
        copd_risk_bin = patient.get('copd_risk_bin', 'Low Risk')
        
        gauge_color = '#2ecc71'
        if copd_risk_bin == 'High Risk':
            gauge_color = '#e74c3c'
        elif copd_risk_bin == 'Medium Risk':
            gauge_color = '#f1c40f'
        
        fig_copd = go.Figure(go.Indicator(
            mode="gauge+number",
            value=copd_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 16}},
            number={'suffix': "%", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': gauge_color, 'thickness': 0.75},
                'steps': [
                    {'range': [0, 35], 'color': "lightgreen"},
                    {'range': [35, 65], 'color': "lightyellow"},
                    {'range': [65, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': copd_score * 100
                }
            }
        ))
        fig_copd.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_copd, use_container_width=True)
        
        if copd_risk_bin == 'High Risk':
            st.error(f"Risk Level: 🔴 {copd_risk_bin}")
        elif copd_risk_bin == 'Medium Risk':
            st.warning(f"Risk Level: 🟡 {copd_risk_bin}")
        else:
            st.success(f"Risk Level: 🟢 {copd_risk_bin}")
        
        
        if 'risk_smoking_tobacco' in patient.index:
            st.metric("Smoking Risk", f"{patient['risk_smoking_tobacco']:.2f}")
        if 'risk_respiratory_infection' in patient.index:
            st.metric("Respiratory Risk", f"{patient['risk_respiratory_infection']:.2f}")
        
        if patient.get('has_copd', 0) == 1:
            st.error("Status: Has COPD")
        else:
            st.success("Status: No COPD")
    
    st.markdown("---")
    
    st.subheader("💡 Health Recommendations")
    
    if disease_count == 0:
        st.success("""
        ✅ Excellent Health Status
        - Maintain current healthy lifestyle
        - Continue regular health checkups
        - Keep monitoring key health metrics
        - Focus on preventive care
        """)
    elif disease_count == 1:
        if patient.get('has_diabetes', 0) == 1:
            st.warning("""
            🩺 Diabetes Management
            - Monitor blood glucose regularly
            - Follow prescribed medication schedule
            - Maintain healthy diet (low sugar, high fiber)
            - Regular exercise (30 min/day)
            - HbA1c testing every 3 months
            """)
        elif patient.get('has_cvd', 0) == 1:
            st.warning("""
            🫀 Cardiovascular Care
            - Monitor blood pressure daily
            - Reduce sodium intake
            - Regular cardio exercise
            - Stress management
            - Medication compliance
            """)
        else:
            st.warning("""
            🫁 COPD Management
            - Avoid smoking and pollutants
            - Use prescribed inhalers correctly
            - Pulmonary rehabilitation exercises
            - Get vaccinated (flu, pneumonia)
            - Monitor oxygen levels
            """)
    elif disease_count == 2:
        st.error("""
        🔴 Multiple Disease Management Required
        - Integrated treatment plan from multiple specialists
        - More frequent monitoring (weekly checkups)
        - Strict medication adherence
        - Lifestyle modifications across all areas
        - Regular lab work and imaging
        """)
    else:
        st.error("""
        🚨 CRITICAL: Triple Comorbidity
        - High priority for coordinated care team
        - Daily monitoring required
        - Strict treatment protocol adherence
        - Possible hospitalization risk
        - Immediate lifestyle intervention needed
        """)
    
    st.markdown("---")
# ===============================
# TAB 7: PERSONAL RISK CALCULATOR
# ===============================
with risk_calculator:
    st.header("🩺 Check Your Personal Health Risk")
    st.markdown("### Enter your health metrics to assess your risk for Diabetes, CVD, and COPD")
    st.markdown("---")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Basic Information")
        
        # Age input
        user_age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=95,
            value=30,
            step=1,
            help="Enter your age in years"
        )
        
        # Gender selection
        user_gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Select your biological gender"
        )
        
        # BMI input
        user_bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0,
            max_value=60.0,
            value=22.0,
            step=0.1,
            help="Body Mass Index = weight(kg) / height(m)²"
        )
        
        # Systolic Blood Pressure
        user_sbp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=80,
            max_value=200,
            value=120,
            step=1,
            help="The top number in blood pressure reading"
        )
        
        # Diastolic Blood Pressure
        user_dbp = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=50,
            max_value=130,
            value=80,
            step=1,
            help="The bottom number in blood pressure reading"
        )
    
    with col2:
        st.subheader("🧪 Lab Values & Lifestyle")
        
        # Glucose
        user_glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=50.0,
            max_value=400.0,
            value=90.0,
            step=1.0,
            help="Fasting blood glucose level"
        )
        
        # HbA1c
        user_hba1c = st.number_input(
            "Hemoglobin A1c (%)",
            min_value=4.0,
            max_value=15.0,
            value=5.5,
            step=0.1,
            help="Average blood sugar over 3 months"
        )
        
        # Total Cholesterol
        user_cholesterol = st.number_input(
            "Total Cholesterol (mg/dL)",
            min_value=100.0,
            max_value=400.0,
            value=180.0,
            step=1.0,
            help="Total cholesterol level"
        )
        
        # Smoking status
        user_smoking = st.radio(
            "Do you smoke or use tobacco?",
            options=["No", "Yes"],
            horizontal=True,
            help="Current or past tobacco use"
        )
        
        # Family history
        user_family_history = st.radio(
            "Family history of heart/lung disease?",
            options=["No", "Yes"],
            horizontal=True,
            help="Do your parents or siblings have chronic diseases?"
        )
    
    st.markdown("---")
    
    # Calculate Risk Button
    if st.button("🔍 Check My Risk", type="primary", use_container_width=True):
        
        st.markdown("---")
        st.subheader("📊 Your Health Risk Assessment Results")
        st.markdown("---")
        
        # ===========================
        # CALCULATE DIABETES RISK
        # ===========================
        # Normalize inputs for diabetes calculation
        glucose_risk = np.clip((user_glucose - 70) / 250, 0, 1)
        hba1c_risk = np.clip((user_hba1c - 4.0) / 10.0, 0, 1)
        bmi_risk_diabetes = np.clip((user_bmi - 18.5) / 25, 0, 1)
        
        # Calculate diabetes risk score (0-1)
        diabetes_risk_score = np.clip(
            0.5 * glucose_risk + 0.3 * hba1c_risk + 0.2 * bmi_risk_diabetes,
            0, 1
        )
        
        # Determine diabetes risk level (using your thresholds)
        if diabetes_risk_score < 0.40:
            diabetes_risk_level = "Low Risk"
            diabetes_color = "#2ecc71"
        elif diabetes_risk_score < 0.70:
            diabetes_risk_level = "Medium Risk"
            diabetes_color = "#f39c12"
        else:
            diabetes_risk_level = "High Risk"
            diabetes_color = "#e74c3c"
        
        # ===========================
        # CALCULATE CVD RISK
        # ===========================
        # Normalize inputs for CVD calculation
        if user_bmi < 18.5:
            bmi_risk_cvd = 0.1
        elif user_bmi < 25:
            bmi_risk_cvd = 0.2
        elif user_bmi < 30:
            bmi_risk_cvd = 0.5
        elif user_bmi < 35:
            bmi_risk_cvd = 0.7
        elif user_bmi < 40:
            bmi_risk_cvd = 0.85
        else:
            bmi_risk_cvd = 0.95
        
        if user_sbp < 120:
            sbp_risk = 0.2
        elif user_sbp < 130:
            sbp_risk = 0.4
        elif user_sbp < 140:
            sbp_risk = 0.6
        elif user_sbp < 160:
            sbp_risk = 0.8
        else:
            sbp_risk = 0.95
        
        if user_dbp < 80:
            dbp_risk = 0.2
        elif user_dbp < 90:
            dbp_risk = 0.5
        elif user_dbp < 100:
            dbp_risk = 0.75
        else:
            dbp_risk = 0.9
        
        # Calculate CVD risk score (0-1)
        cvd_risk_score = np.clip(
            0.25 * bmi_risk_cvd + 0.40 * sbp_risk + 0.35 * dbp_risk,
            0, 1
        )
        
        # Determine CVD risk level (using your thresholds)
        if cvd_risk_score < 0.10:
            cvd_risk_level = "Low Risk"
            cvd_color = "#2ecc71"
        elif cvd_risk_score < 0.20:
            cvd_risk_level = "Medium Risk"
            cvd_color = "#f39c12"
        else:
            cvd_risk_level = "High Risk"
            cvd_color = "#e74c3c"
        
        # ===========================
        # CALCULATE COPD RISK
        # ===========================
        # Age risk
        if user_age < 40:
            age_risk = 0.1
        elif user_age < 50:
            age_risk = 0.3
        elif user_age < 60:
            age_risk = 0.5
        elif user_age < 70:
            age_risk = 0.7
        else:
            age_risk = 0.9
        
        # Smoking risk
        smoking_risk = 0.8 if user_smoking == "Yes" else 0.2
        
        # Family history risk
        genetic_risk = 0.6 if user_family_history == "Yes" else 0.3
        
        # Calculate COPD risk score (0-1)
        copd_risk_score = np.clip(
            0.40 * age_risk + 0.45 * smoking_risk + 0.15 * genetic_risk,
            0, 1
        )
        
        # Determine COPD risk level (using your thresholds)
        if copd_risk_score < 0.35:
            copd_risk_level = "Low Risk"
            copd_color = "#2ecc71"
        elif copd_risk_score < 0.65:
            copd_risk_level = "Medium Risk"
            copd_color = "#f39c12"
        else:
            copd_risk_level = "High Risk"
            copd_color = "#e74c3c"
        
        # ===========================
        # DISPLAY RESULTS
        # ===========================
        
        # Create 3 columns for gauge charts
        col1, col2, col3 = st.columns(3)
        
        # DIABETES GAUGE
        with col1:
            st.markdown("### 🩺 Diabetes Risk")
            fig_diabetes = go.Figure(go.Indicator(
                mode="gauge+number",
                value=diabetes_risk_score * 100,
                title={'text': diabetes_risk_level, 'font': {'size': 20, 'color': diabetes_color}},
                number={'suffix': "%", 'font': {'size': 30}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': diabetes_color, 'thickness': 0.75},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': diabetes_risk_score * 100
                    }
                }
            ))
            fig_diabetes.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_diabetes, use_container_width=True)
            
            if diabetes_risk_level == "High Risk":
                st.error(f"**Risk Score:** {diabetes_risk_score:.2%}")
                st.warning("⚠️ **Action Needed:** Consult a doctor for diabetes screening")
            elif diabetes_risk_level == "Medium Risk":
                st.warning(f"**Risk Score:** {diabetes_risk_score:.2%}")
                st.info("💡 **Recommendation:** Monitor glucose levels regularly")
            else:
                st.success(f"**Risk Score:** {diabetes_risk_score:.2%}")
                st.info("✅ **Status:** Continue healthy lifestyle")
        
        # CVD GAUGE
        with col2:
            st.markdown("### 🫀 CVD Risk")
            fig_cvd = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cvd_risk_score * 100,
                title={'text': cvd_risk_level, 'font': {'size': 20, 'color': cvd_color}},
                number={'suffix': "%", 'font': {'size': 30}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': cvd_color, 'thickness': 0.75},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgreen"},
                        {'range': [10, 20], 'color': "lightyellow"},
                        {'range': [20, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': cvd_risk_score * 100
                    }
                }
            ))
            fig_cvd.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_cvd, use_container_width=True)
            
            if cvd_risk_level == "High Risk":
                st.error(f"**Risk Score:** {cvd_risk_score:.2%}")
                st.warning("⚠️ **Action Needed:** Urgent cardiovascular check-up required")
            elif cvd_risk_level == "Medium Risk":
                st.warning(f"**Risk Score:** {cvd_risk_score:.2%}")
                st.info("💡 **Recommendation:** Monitor blood pressure regularly")
            else:
                st.success(f"**Risk Score:** {cvd_risk_score:.2%}")
                st.info("✅ **Status:** Heart health looks good")
        
        # COPD GAUGE
        with col3:
            st.markdown("### 🫁 COPD Risk")
            fig_copd = go.Figure(go.Indicator(
                mode="gauge+number",
                value=copd_risk_score * 100,
                title={'text': copd_risk_level, 'font': {'size': 20, 'color': copd_color}},
                number={'suffix': "%", 'font': {'size': 30}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': copd_color, 'thickness': 0.75},
                    'steps': [
                        {'range': [0, 35], 'color': "lightgreen"},
                        {'range': [35, 65], 'color': "lightyellow"},
                        {'range': [65, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': copd_risk_score * 100
                    }
                }
            ))
            fig_copd.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_copd, use_container_width=True)
            
            if copd_risk_level == "High Risk":
                st.error(f"**Risk Score:** {copd_risk_score:.2%}")
                st.warning("⚠️ **Action Needed:** Pulmonary function test recommended")
            elif copd_risk_level == "Medium Risk":
                st.warning(f"**Risk Score:** {copd_risk_score:.2%}")
                st.info("💡 **Recommendation:** Quit smoking, avoid pollutants")
            else:
                st.success(f"**Risk Score:** {copd_risk_score:.2%}")
                st.info("✅ **Status:** Lung health is good")
        
        st.markdown("---")
        
        # OVERALL HEALTH SUMMARY
        st.subheader("🏥 Overall Health Summary")
        
        high_risk_count = sum([
            diabetes_risk_level == "High Risk",
            cvd_risk_level == "High Risk",
            copd_risk_level == "High Risk"
        ])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if high_risk_count == 0:
                st.success("✅ **Great News!** No high-risk conditions detected. Keep maintaining your healthy lifestyle!")
            elif high_risk_count == 1:
                st.warning("⚠️ **Attention Required:** You have 1 high-risk condition. Please consult a healthcare provider.")
            elif high_risk_count == 2:
                st.error("🔴 **Important:** You have 2 high-risk conditions. Immediate medical consultation is strongly recommended.")
            else:
                st.error("🚨 **URGENT:** You have 3 high-risk conditions. Please seek medical attention immediately!")
            
            st.markdown("### 💡 General Recommendations:")
            recommendations = []
            
            if user_bmi > 25:
                recommendations.append("- 🏃 **Weight Management:** Aim for a healthy BMI (18.5-25)")
            if user_sbp > 120 or user_dbp > 80:
                recommendations.append("- 🧂 **Blood Pressure:** Reduce salt intake, exercise regularly")
            if user_glucose > 100:
                recommendations.append("- 🍎 **Diet:** Reduce sugar intake, eat more fiber")
            if user_smoking == "Yes":
                recommendations.append("- 🚭 **Smoking:** Quit smoking immediately - it affects all three conditions")
            if user_age > 50:
                recommendations.append("- 📅 **Regular Checkups:** Annual health screenings recommended")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.markdown("- ✅ Continue your healthy lifestyle habits")
                st.markdown("- 📅 Regular health check-ups every 1-2 years")
        
        with col2:
            st.metric("Total Risk Factors", high_risk_count)
            st.metric("Age Group", f"{user_age} years")
            st.metric("BMI Category", 
                     "Underweight" if user_bmi < 18.5 else
                     "Normal" if user_bmi < 25 else
                     "Overweight" if user_bmi < 30 else
                     "Obese")
        

        st.markdown("---")
