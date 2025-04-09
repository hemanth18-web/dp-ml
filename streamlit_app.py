import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io

# GitHub URL for the dataset
github_url = "https://raw.githubusercontent.com/hemanth18-web/dp-ml/refs/heads/main/Updated_Flight_Fare_Data%20(20).csv"

# Function to load the dataset from GitHub
@st.cache_data
def load_data_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = response.content.decode('utf-8')
        data = pd.read_csv(io.StringIO(csv_data))
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download the dataset from GitHub: {e}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Failed to parse the CSV data: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Load the dataset
data = load_data_from_github(github_url)

# --- STREAMLIT APP ---
st.title("Flight Fare Data Exploration2")

if data is not None:
    # --- Data Cleaning and Conversion ---
    # Replace "non-stop" with 0
    data['Total_Stops'] = data['Total_Stops'].replace("non-stop", 0)

    # Handle missing values (NaN)
    data['Total_Stops'] = data['Total_Stops'].replace('NaN', np.nan)  # Replace string 'NaN' with actual NaN
    data['Total_Stops'] = data['Total_Stops'].fillna(0)  # Fill NaN with 0 (or another appropriate value)

    # --- Identify Non-Convertible Values ---
    def is_number(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    non_numeric_values = data['Total_Stops'][data['Total_Stops'].apply(lambda x: not is_number(x))]
    unique_non_numeric = non_numeric_values.unique()

    if len(unique_non_numeric) > 0:
        st.warning(f"Found non-numeric values in 'Total_Stops': {unique_non_numeric}")
        # Replace these values with a suitable numeric value (e.g., 0) or NaN
        # For example, if you want to replace all non-numeric values with 0:
        for val in unique_non_numeric:
            data['Total_Stops'] = data['Total_Stops'].replace(val, 0)

    else:
        st.success("No non-numeric values found in 'Total_Stops'")

    # Convert 'Total_Stops' to numeric
    data['Total_Stops'] = pd.to_numeric(data['Total_Stops'])

    st.header("Data Preview")
    st.dataframe(data.head())

    st.header("Data Summary")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")

    # --- VISUALIZATIONS ---
    st.header("Visualizations")

    # Airline Distribution
    st.subheader("Airline Distribution")
    fig_airline, ax_airline = plt.subplots(figsize=(10, 6))
    sns.countplot(x="Airline", data=data, ax=ax_airline, palette="muted", order=data['Airline'].value_counts().index)  # Order by frequency
    ax_airline.tick_params(axis='x', rotation=90)
    st.pyplot(fig_airline)

    # Price vs. Number of Stops (Scatter Plot)
    st.subheader("Price vs. Number of Stops")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="Price", y="Total_Stops", data=data, ax=ax_scatter)
    st.pyplot(fig_scatter)

    # Price Distribution (Histogram)
    st.subheader("Price Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Price'], kde=True, ax=ax_hist)
    st.pyplot(fig_hist)

    # --- ADDED: Scatter plot with customized aesthetics ---
    st.header("Price vs Number of Stops (Customized)")
    fig_scatter_custom, ax_scatter_custom = plt.subplots(figsize=(10, 6))
    scatter = ax_scatter_custom.scatter(data['Price'], data['Total_Stops'], s=80, alpha=0.7, c=data['Price'], cmap='viridis', edgecolors='black')
    ax_scatter_custom.set_title('Price vs Number of Stops', fontsize=14, fontweight='bold')
    ax_scatter_custom.set_xlabel('Price', fontsize=12)
    ax_scatter_custom.set_ylabel('Number of Stops', fontsize=12)
    ax_scatter_custom.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig_scatter_custom.colorbar(scatter, label='Price')  # Add colorbar to the figure
    st.pyplot(fig_scatter_custom)

    # --- ADDED: Airline Distribution Countplot with Styling ---
    st.header("Airline Distribution (Styled)")
    fig_countplot, ax_countplot = plt.subplots(figsize=(10, 6))
    sns.countplot(x="Airline", data=data, hue="Airline", palette="muted", legend=False, ax=ax_countplot)
    ax_countplot.set_title("✈️ Airline Distribution ✈️", fontweight="bold", fontsize=14, color="#80aaff")
    ax_countplot.set_xlabel("Airline")
    ax_countplot.set_ylabel("Count")
    ax_countplot.tick_params(axis='x', rotation=90, labelsize=8)
    fig_countplot.tight_layout()
    st.pyplot(fig_countplot)

else:
    st.write("Failed to load data.  Check the GitHub URL and your internet connection.")
