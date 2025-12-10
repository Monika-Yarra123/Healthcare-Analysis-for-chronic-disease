Healthcare Analysis for Chronic Disease Risk Stratification
Diabetes | Cardiovascular Disease (CVD) | COPD

Project Overview:

This project analyzes synthetic patient data from Synthea (~4,000 patients) to predict and stratify risk for three chronic diseases: Diabetes, CVD, and COPD.
Using Logistic Regression, risk scores were generated and categorized into Low, Medium, and High risk groups.
A Streamlit dashboard with 7 tabs was built in Visual Studio Code to explore patient-level and population-level insights.

Objectives:

1.Predict risk for Diabetes, CVD, COPD

2.Generate individual risk scores using Logistic Regression

3.Categorize patients into Low/Medium/High risk bins

4.Provide an interactive dashboard for analyzing the results

5.Support Healthcare Analysts in risk stratification

Data Source:

Synthetic patient data generated using Synthea

Includes demographics, vitals, encounters, chronic conditions, labs, smoking status, etc.

Modeling Approach:

Select predictors relevant to each disease

Train Logistic Regression models

Generate predicted probability (risk score) for each disease

Classify into:

Low Risk

Medium Risk

High Risk

Streamlit Dashboard (7 Tabs)

Overview

Diabetes Insights

CVD Insights

COPD Insights

Individual Patient Check

Risk Calculator

Tech Stack:

Python

Streamlit

Pandas, NumPy

Scikit-learn

Matplotlib

Visual Studio Code

Synthea (data generation)
