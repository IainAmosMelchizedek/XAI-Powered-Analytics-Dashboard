# streamlit_dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def main():
    # Set up the page configuration
    st.set_page_config(
        page_title="XAI-Powered Analytics Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load the dataset
    data = load_data("expanded_hypothetical_data.csv")

    # Dashboard title and description
    st.title("⚓ XAI-Powered Analytics Dashboard")
    st.write("An interactive dashboard showcasing SHAP insights, data sources, key metrics, and financial risk prediction scenarios.")

    # Render each section of the dashboard
    model_interpretability_section(data)
    data_sources_section(data)
    key_metrics_section(data)
    real_world_scenario_section(data)

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def model_interpretability_section(data):
    """
    Display the Model Interpretability section with SHAP and LIME insights.

    Args:
        data (pd.DataFrame): The dataset containing the required data.
    """
    st.markdown("## **1. Model Interpretability (SHAP/LIME Insights)**")
    st.write("Analyzing feature contributions to model predictions.")

    # Filter data for this section
    model_data = data[data["Component"] == "Model Interpretability"].dropna(subset=["Feature", "SHAP Value", "LIME Weight"])

    # Create a bar plot for SHAP Values
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=model_data, x="SHAP Value", y="Feature", palette="Blues_d", ax=ax)
    ax.set_title("SHAP Values by Feature", fontsize=14, color="#023e8a")
    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

    st.markdown("---")

def data_sources_section(data):
    """
    Display the Data Sources section with an annotated pie chart.

    Args:
        data (pd.DataFrame): The dataset containing the required data.
    """
    st.markdown("## **2. Data Sources**")
    st.write("Visualizing data source contributions to the model.")

    # Filter data for this section
    source_data = data[data["Component"] == "Data Sources"].dropna(subset=["Source", "Percentage Contribution (%)"])

    # Prepare data for the pie chart
    labels = source_data["Source"]
    sizes = source_data["Percentage Contribution (%)"]
    colors = sns.color_palette("Blues", len(labels))

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        textprops=dict(color="black")
    )
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    ax.set_title("Data Sources Distribution", fontsize=14, color="#023e8a")
    st.pyplot(fig)

    st.markdown("---")

def key_metrics_section(data):
    """
    Display the Key Metrics section with accuracy and compliance rate charts.

    Args:
        data (pd.DataFrame): The dataset containing the required data.
    """
    st.markdown("## **3. Key Metrics (Accuracy & Compliance Rates)**")
    st.write("Comparing model performance metrics.")

    # Filter data for this section
    metrics_data = data[data["Component"] == "Key Metrics"].dropna(subset=["Model", "Accuracy (%)", "Compliance Rate (%)"])

    # Set up columns for side-by-side charts
    col1, col2 = st.columns(2)

    # Bar chart for Accuracy
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=metrics_data, x="Model", y="Accuracy (%)", palette="Blues_d", ax=ax)
        ax.set_title("Model Accuracy Comparison", fontsize=12)
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

    # Line chart for Compliance Rates
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=metrics_data, x="Model", y="Compliance Rate (%)", marker='o', ax=ax, color="#00b4d8")
        ax.set_title("Model Compliance Rates", fontsize=12)
        ax.set_xlabel("Model")
        ax.set_ylabel("Compliance Rate (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

    st.markdown("---")

def real_world_scenario_section(data):
    """
    Display the Real-World Scenario section with a heatmap and interactive sliders.

    Args:
        data (pd.DataFrame): The dataset containing the required data.
    """
    st.markdown("## **4. Real-World Scenario (Financial Risk Prediction)**")
    st.write("Exploring risk factor contributions and testing scenarios.")

    # Filter data for this section
    scenario_data = data[data["Component"] == "Real-World Scenario"].dropna(subset=["Risk Factor", "Risk Score Contribution (%)", "Correlation"])

    # Create a heatmap of Risk Score Contributions
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot_table = scenario_data.pivot_table(
        values="Risk Score Contribution (%)",
        index="Risk Factor",
        aggfunc='first'
    )
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar=False,
        ax=ax
    )
    ax.set_title("Risk Score Contributions Heatmap", fontsize=12)
    ax.set_ylabel("")
    st.pyplot(fig)

    st.markdown("### **Interactive Scenario Testing**")
    # Interactive sliders for scenario testing
    factors = scenario_data["Risk Factor"].tolist()
    default_values = scenario_data["Risk Score Contribution (%)"].tolist()
    inputs = {}
    for factor, default in zip(factors, default_values):
        inputs[factor] = st.slider(factor, min_value=0, max_value=100, value=int(default))

    # Display the adjusted risk contributions
    st.write("#### Adjusted Risk Contributions:")
    for factor, value in inputs.items():
        st.write(f"- **{factor}**: {value}%")

    st.markdown("---")
    st.markdown("### Thank you for exploring the dashboard!")

if __name__ == "__main__":
    main()
