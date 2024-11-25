# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Financial Risk Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dashboard title and description
st.title("⚓ Financial Risk Prediction Dashboard")
st.write("An interactive dashboard showcasing SHAP insights, data sources, key metrics, and financial risk prediction scenarios.")

# Load your dataset (adjust the file path if needed)
file_path = "combined_hypothetical_data.csv"  # Ensure this file is in the same directory
data = pd.read_csv(file_path)

# --- 1. Model Interpretability Section ---
st.markdown("## **1. Model Interpretability (SHAP/LIME Insights)**")
model_data = data[data["Component"] == "Model Interpretability"]

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=model_data, x="SHAP Value", y="Feature", palette="Blues_d", ax=ax)
ax.set_title("SHAP Values by Feature", fontsize=16, color="#023e8a")
st.pyplot(fig)

# Spacer
st.markdown("---")

# --- 2. Data Sources Section ---
st.markdown("## **2. Data Sources**")
source_data = data[data["Component"] == "Data Sources"].dropna()

fig = go.Figure(
    go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=source_data["Source"].drop_duplicates().tolist(),
            color=["#0077b6", "#00b4d8", "#caf0f8"],
        ),
        link=dict(
            source=[0, 1, 1],
            target=[2, 3, 4],
            value=[40, 30, 30],
        ),
    )
)
fig.update_layout(title_text="Data Flow Sankey Diagram", font_size=12)
st.plotly_chart(fig)

# Spacer
st.markdown("---")

# --- 3. Key Metrics Section ---
st.markdown("## **3. Key Metrics (Accuracy & Compliance Rates)**")
metrics_data = data.dropna(subset=["Accuracy", "Compliance Rate"])
col1, col2 = st.columns(2)
with col1:
    st.bar_chart(metrics_data.set_index("Model")["Accuracy"], use_container_width=True)
    st.caption("Model Accuracy")
with col2:
    st.line_chart(metrics_data.set_index("Model")["Compliance Rate"], use_container_width=True)
    st.caption("Compliance Rates")

# Spacer
st.markdown("---")

# --- 4. Real-World Scenario Section ---
st.markdown("## **4. Real-World Scenario (Financial Risk Prediction)**")
scenario_data = data.dropna(subset=["Risk Factor", "Correlation"])

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(
    scenario_data.pivot_table(index="Risk Factor", columns="Correlation", aggfunc="size"),
    cmap="Blues",
    annot=True,
    fmt=".0f",
)
st.pyplot(fig)

st.markdown("### **Interactive Scenario Testing**")
loan_to_income_ratio = st.slider("Loan-to-Income Ratio", 0, 100, 35)
payment_history = st.slider("Payment History Impact", 0, 100, 30)
credit_utilization = st.slider("Credit Utilization Impact", 0, 100, 20)

st.write(f"Scenario Impact: Loan-to-Income = {loan_to_income_ratio}%, Payment History = {payment_history}%, Credit Utilization = {credit_utilization}%")

st.markdown("---")
st.markdown("### Thank you for exploring the dashboard!")