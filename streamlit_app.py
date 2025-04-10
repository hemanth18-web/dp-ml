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
st.title("Flight Fare Data Exploration3")

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

    # Create the line plot
    st.subheader("Ticket Price Trends Over Time")
    fig_lineplot, ax_lineplot = plt.subplots(figsize=(10, 6))
    sns.lineplot(x="Date_of_Journey", y="Price", data=data, hue="Airline", marker="o", palette="viridis", ax=ax_lineplot)  # Changed x to "Date_of_Journey" and y to "Price"
    ax_lineplot.set_title("Ticket Price Trends Over Time", fontsize=14, fontweight="bold")
    ax_lineplot.set_xlabel("Date", fontsize=12)
    ax_lineplot.set_ylabel("Price (₹) ", fontsize=12)
    ax_lineplot.tick_params(axis='x', rotation=90)
    fig_lineplot.tight_layout()
    st.pyplot(fig_lineplot)

    # Assuming 'Days_Until_Departure' is in your data, we first calculate the frequency of each unique value
    st.subheader("Days Until Departure (Line Plot)")
    days_count = data['Days_Until_Departure'].value_counts().sort_index()
    fig_days_line, ax_days_line = plt.subplots(figsize=(10, 6))
    ax_days_line.plot(days_count.index, days_count.values, marker='o', color='skyblue', linewidth=2)
    ax_days_line.set_title('Days Until Departure (Line Plot)', fontsize=12, fontweight='bold')
    ax_days_line.set_xlabel('Days Until Departure', fontsize=12)
    ax_days_line.set_ylabel('Count', fontsize=12)
    fig_days_line.tight_layout()
    st.pyplot(fig_days_line)

    # Set style
    st.subheader("Price Distribution by Cabin Class (Customized)")
    sns.set(style="whitegrid")
    fig_boxplot_cabin, ax_boxplot_cabin = plt.subplots(figsize=(10, 8))
    sns.boxplot(x="Cabin_Class", y="Price", data=data, palette="Set1", hue="Airline", ax=ax_boxplot_cabin)
    ax_boxplot_cabin.set_title("Price Distribution by Cabin Class (Customized)", fontsize=14, fontweight='bold')
    ax_boxplot_cabin.set_xlabel("Cabin Class", fontsize=12)
    ax_boxplot_cabin.set_ylabel("Price ()", fontsize=12)
    ax_boxplot_cabin.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig_boxplot_cabin.tight_layout()
    st.pyplot(fig_boxplot_cabin)

    # Set style for a clean, modern look
    st.subheader("Price Distribution by Cabin Class")
    sns.set(style="whitegrid")
    fig_boxplot_cabin2, ax_boxplot_cabin2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Cabin_Class", y="Price", data=data, palette="Set2", hue="Cabin_Class", ax=ax_boxplot_cabin2)
    ax_boxplot_cabin2.set_title("Price Distribution by Cabin Class ", fontsize=14, fontweight='bold', color="#2c3e50")
    ax_boxplot_cabin2.set_xlabel("Cabin Class", fontsize=12)
    ax_boxplot_cabin2.set_ylabel("Price ($)", fontsize=12)
    ax_boxplot_cabin2.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)
    fig_boxplot_cabin2.tight_layout()
    st.pyplot(fig_boxplot_cabin2)

    # Set style for a clean, modern look
    st.subheader("Price Distribution by Airline")
    sns.set(style="whitegrid")
    fig_boxplot_airline, ax_boxplot_airline = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Airline", y="Price", data=data.sort_values('Price', ascending=False), hue="Airline", palette="Set2", ax=ax_boxplot_airline)
    ax_boxplot_airline.set_title(" Price Distribution by Airline", fontsize=14, fontweight='bold', color="#2c3e50")
    ax_boxplot_airline.set_xlabel("Airline ", fontsize=12)
    ax_boxplot_airline.set_ylabel("Price", fontsize=12)
    ax_boxplot_airline.tick_params(axis='x', rotation=90)
    ax_boxplot_airline.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig_boxplot_airline.tight_layout()
    st.pyplot(fig_boxplot_airline)

else:
    st.write("Failed to load data.  Check the GitHub URL and your internet connection.")
