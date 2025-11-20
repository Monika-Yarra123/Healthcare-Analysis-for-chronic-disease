import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
    file_path = r"C:\Users\Monika Yarra\Downloads\Chronic Diseases.csv"  # <-- Update path
    df = pd.read_csv(file_path)
    
    df.columns = [c.strip().lower() for c in df.columns]  # lowercase
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate columns
    
    # Numeric conversions
    for col in ['diabetes_risk_score', 'cvd_risk_score', 'copd_risk_score', 'age', 'bmi', 'glucose']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Risk bins
    df['diabetes_risk_bin'] = pd.cut(df['diabetes_risk_score'], bins=[-0.01, 0.4, 0.7, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk'])
    df['has_diabetes'] = (df['diabetes_risk_score'] >= 0.7).astype(int)
    
    if 'cvd_risk_score' in df.columns:
        df['cvd_risk_bin'] = pd.cut(df['cvd_risk_score'], bins=[-0.01, 0.1, 0.2, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk'])
        df['has_cvd'] = (df['cvd_risk_score'] >= 0.2).astype(int)
        
    if 'copd_risk_score' in df.columns:
        df['copd_risk_bin'] = pd.cut(df['copd_risk_score'], bins=[-0.01, 0.5, 0.65, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk'])
        df['has_copd'] = (df['copd_risk_score'] >= 0.65).astype(int)
    
    # Age & BMI categories
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0,18,30,45,60,120], labels=['0-18','19-30','31-45','46-60','60+'])
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,100], labels=['Underweight','Normal','Overweight','Obese'])
    
    # Birth year
    if 'birthdate' in df.columns:
        df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
        df['birth_year'] = df['birthdate'].dt.year
    
    return df

df = load_data()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔍 Global Filters")
age_range = st.sidebar.slider("Age Range", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))
gender_options = df['gender'].dropna().unique().tolist()
selected_gender = st.sidebar.multiselect("Gender", gender_options, default=gender_options)
county_options = sorted(df['county'].dropna().unique().tolist())
selected_counties = st.sidebar.multiselect("County", county_options, default=county_options)

filtered_df = df[
    (df['age'] >= age_range[0]) & 
    (df['age'] <= age_range[1]) &
    (df['gender'].isin(selected_gender)) &
    (df['county'].isin(selected_counties))
].copy()

st.sidebar.success(f"✅ {len(filtered_df)} patients filtered")
st.sidebar.info(f"📊 Out of {len(df)} total")

# -------------------------------
# Tabs
# -------------------------------
overview_tab, diabetes_tab, cvd_tab, copd_tab, comorbidity_tab, individual_tab = st.tabs([
    "📊 Overview", "🩺 Diabetes", "🫀 CVD", "🫁 COPD", "📋 Comorbidity", "🔍 Individual Patient Risk"
])

# -------------------------------
# TAB 1: OVERVIEW
# -------------------------------
with overview_tab:
    st.header("📊 Overview")
    st.info(f"🔍 Showing {len(filtered_df)} of {len(df)} total patients")
    st.markdown("---")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_patients = len(filtered_df)
    col1.metric("Total Patients", total_patients)
    col2.metric("High Risk Diabetes", len(filtered_df[filtered_df['diabetes_risk_bin']=='High Risk']), delta=f"{len(filtered_df[filtered_df['diabetes_risk_bin']=='High Risk'])/total_patients*100:.1f}%")
    col3.metric("High Risk CVD", len(filtered_df[filtered_df['cvd_risk_bin']=='High Risk']), delta=f"{len(filtered_df[filtered_df['cvd_risk_bin']=='High Risk'])/total_patients*100:.1f}%")
    col4.metric("High Risk COPD", len(filtered_df[filtered_df['copd_risk_bin']=='High Risk']), delta=f"{len(filtered_df[filtered_df['copd_risk_bin']=='High Risk'])/total_patients*100:.1f}%")
    
    st.markdown("---")
    
    # Risk Distribution Bar Chart
    disease_data = []
    for disease, risk_col in {'Diabetes':'diabetes_risk_bin','CVD':'cvd_risk_bin','COPD':'copd_risk_bin'}.items():
        if risk_col in filtered_df.columns:
            counts = filtered_df[risk_col].value_counts().reset_index()
            counts.columns = ['Risk Level','Count']
            counts['Disease'] = disease
            disease_data.append(counts)
    
    if disease_data:
        combined_risk = pd.concat(disease_data)
        fig_bar = px.bar(combined_risk, x='Risk Level', y='Count', color='Risk Level',
                         facet_col='Disease', text='Count',
                         category_orders={'Risk Level':['Low Risk','Medium Risk','High Risk']},
                         color_discrete_map={'Low Risk':'#2ecc71','Medium Risk':'#f39c12','High Risk':'#e74c3c'},
                         title='Risk Distribution Across Diseases')
        fig_bar.update_traces(textposition='outside')
        fig_bar.for_each_annotation(lambda a: a.update(text=a.text.replace("Disease=","")))
        st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------
# TAB 2: DIABETES
# -------------------------------
with diabetes_tab:
    st.header("🩺 Diabetes Insights")
    diabetes_patients = filtered_df[filtered_df['has_diabetes']==1].copy()
    total_patients = len(filtered_df)
    
    col1, col2 = st.columns(2)
    col1.metric("Diabetic Patients", len(diabetes_patients), delta=f"{len(diabetes_patients)/total_patients*100:.1f}%")
    col2.metric("Non-Diabetic Patients", total_patients-len(diabetes_patients), delta=f"{(total_patients-len(diabetes_patients))/total_patients*100:.1f}%")
    
    st.markdown("---")
    
    # Risk Level Distribution
    patients_with_risk = df[df['diabetes_risk_score'].notna()]
    risk_counts = patients_with_risk['diabetes_risk_bin'].value_counts().reindex(['Low Risk','Medium Risk','High Risk'], fill_value=0).reset_index()
    risk_counts.columns = ['Risk Bin','Count']
    fig = px.bar(risk_counts, x='Risk Bin', y='Count', text='Count',
                 color='Risk Bin', color_discrete_map={'Low Risk':'#2ecc71','Medium Risk':'#f39c12','High Risk':'#e74c3c'})
    fig.update_traces(textposition='outside')
    fig.update_layout(title='Diabetes Risk Level Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
# -------------------------------
# TAB 3: CVD
# -------------------------------
with cvd_tab:
    st.header("🫀 CVD Insights")
    cvd_patients = filtered_df[filtered_df['has_cvd']==1].copy()
    total_patients = len(filtered_df)
    
    col1, col2 = st.columns(2)
    col1.metric("High-Risk CVD Patients", len(cvd_patients[cvd_patients['cvd_risk_bin']=='High Risk']), delta=f"{len(cvd_patients[cvd_patients['cvd_risk_bin']=='High Risk'])/total_patients*100:.1f}%")
    col2.metric("Non-CVD Patients", total_patients-len(cvd_patients), delta=f"{(total_patients-len(cvd_patients))/total_patients*100:.1f}%")
    
    st.markdown("---")
    
    patients_with_risk = filtered_df[filtered_df['cvd_risk_score'].notna()]
    risk_counts = patients_with_risk['cvd_risk_bin'].value_counts().reindex(['Low Risk','Medium Risk','High Risk'], fill_value=0).reset_index()
    risk_counts.columns = ['Risk Bin','Count']
    fig = px.bar(risk_counts, x='Risk Bin', y='Count', text='Count',
                 color='Risk Bin', color_discrete_map={'Low Risk':'#2ecc71','Medium Risk':'#f39c12','High Risk':'#e74c3c'})
    fig.update_traces(textposition='outside')
    fig.update_layout(title='CVD Risk Level Distribution')
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 4: COPD
# -------------------------------
with copd_tab:
    st.header("🫁 COPD Insights")
    copd_patients = filtered_df[filtered_df['has_copd']==1].copy()
    total_patients = len(filtered_df)
    
    col1, col2 = st.columns(2)
    col1.metric("High-Risk COPD Patients", len(copd_patients[copd_patients['copd_risk_bin']=='High Risk']), delta=f"{len(copd_patients[copd_patients['copd_risk_bin']=='High Risk'])/total_patients*100:.1f}%")
    col2.metric("Non-COPD Patients", total_patients-len(copd_patients), delta=f"{(total_patients-len(copd_patients))/total_patients*100:.1f}%")
    
    st.markdown("---")
    
    patients_with_risk = filtered_df[filtered_df['copd_risk_score'].notna()]
    risk_counts = patients_with_risk['copd_risk_bin'].value_counts().reindex(['Low Risk','Medium Risk','High Risk'], fill_value=0).reset_index()
    risk_counts.columns = ['Risk Bin','Count']
    fig = px.bar(risk_counts, x='Risk Bin', y='Count', text='Count',
                 color='Risk Bin', color_discrete_map={'Low Risk':'#2ecc71','Medium Risk':'#f39c12','High Risk':'#e74c3c'})
    fig.update_traces(textposition='outside')
    fig.update_layout(title='COPD Risk Level Distribution')
    st.plotly_chart(fig, use_container_width=True)

       
# ===============================
# TAB 5: COMORBIDITY ANALYSIS
# ===============================
with Comorbidity_tab:
    st.header("📋 Disease Comorbidity Analysis")
    st.markdown("### Understanding Disease Interlinking & Co-occurrence")
    st.markdown("---")

    has_diabetes_count = len(filtered_df[filtered_df['has_diabetes'] == 1])
    has_cvd_count = len(filtered_df[filtered_df['has_cvd'] == 1])
    has_copd_count = len(filtered_df[filtered_df['has_copd'] == 1])

    min_patients_threshold = 10

    if (has_diabetes_count >= min_patients_threshold and 
        has_cvd_count >= min_patients_threshold and 
        has_copd_count >= min_patients_threshold):
        st.success("✅ Sufficient data for 3-Disease Comorbidity Analysis")
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
        st.subheader("🥧 Disease Co-occurrence Distribution")
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
        st.markdown("### 🫀 CVD ")
        
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
                    {'range': [0, 5], 'color': "lightgreen"},
                    {'range': [5, 7.5], 'color': "lightyellow"},
                    {'range': [7.5, 100], 'color': "lightcoral"}
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
        
        st.metric("Age", f"{patient.get('age', 0)} years")
        
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






