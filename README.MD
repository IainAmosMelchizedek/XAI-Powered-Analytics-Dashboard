# XAI-Powered Analytics Dashboard

## Overview
The **XAI-Powered Analytics Dashboard** is an interactive tool developed using Streamlit to demonstrate Explainable Artificial Intelligence (XAI) insights. It showcases key analytics and predictions using a hypothetical dataset. The dashboard is designed to aid decision-making by providing transparency, interpretability, and actionable insights.

## Features
1. **Model Interpretability (SHAP/LIME Insights)**:
   - Visualizes feature contributions to predictions using SHAP values.
   - Helps in understanding the importance of features like Loan Amount, Income, and Credit Utilization.

2. **Data Sources**:
   - Displays the proportional contribution of data sources such as Transactions, CRM Data, and Financial Indicators.
   - A pie chart with annotations highlights the distribution of processed features.

3. **Key Metrics (Accuracy & Compliance Rates)**:
   - Bar chart comparing model accuracy across multiple algorithms.
   - Line chart tracking compliance rates over time.

4. **Real-World Scenario (Financial Risk Prediction)**:
   - Heatmap showcasing risk score contributions by factors like Loan-to-Income Ratio and Payment History.
   - Interactive sliders for testing scenarios and exploring impacts on risk scores.

## Dataset
The dashboard uses a **hypothetical dataset**, created specifically for demonstration purposes. The dataset includes:
- **Model Interpretability**: SHAP and LIME values for various features.
- **Data Sources**: Contributions of multiple data streams and their processed features.
- **Key Metrics**: Accuracy and compliance rates for machine learning models.
- **Real-World Scenarios**: Risk factor contributions and correlations.

## Technical Details
- **Framework**: Streamlit
- **Libraries**: pandas, matplotlib, seaborn, plotly
- **Dataset**: `expanded_hypothetical_data.csv`

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<YourUsername>/XAI-Powered-Analytics-Dashboard.git
