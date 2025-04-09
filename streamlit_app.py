import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import re  # Import the regular expression module

# GitHub URL for the dataset
github_url = "https://raw.githubusercontent.com/hemanth18-web/dp-ml/refs/heads/main/Updated_Flight_Fare_Data%20(20).csv"

# Function to load the dataset from GitHub
@st.cache_data
def load_data_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        csv_data = response.content.decode('utf-8')
        data = pd.read_csv(io.StringIO(csv_data))
        return data
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTPError: Could not download the dataset from GitHub. Status code: {e.response.status_code}")
        return None
    except requests.exceptions.ConnectionError as e:
        st.error(f"ConnectionError: Could not connect to GitHub. Please check your internet connection.")
        return None
    except requests.exceptions.Timeout as e:
        st.error(f"TimeoutError: Request to GitHub timed out.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"RequestException: An error occurred while making the request to GitHub: {e}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"ParserError: Failed to parse the CSV data: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Load the dataset
data = load_data_from_github(github_url)

# --- STREAMLIT APP ---
st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️")

# Custom CSS for styling
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:18px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✈️ Flight Fare Prediction App2")

y_pred = None  # Initialize y_pred to None
random_forest_model = None # Initialize the model to None

if data is not None:
    st.write("Data loaded successfully!")  # Debugging statement

    # --- Data Cleaning and Conversion ---
    # ... (Data cleaning and conversion code) ...

    # --- Feature Engineering and Encoding ---
    # ... (Feature engineering and encoding code) ...

    # --- Data Preparation for Modeling ---
    # ... (Data preparation code) ...

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if X_train is not None and not X_train.empty:
        st.write("X_train is not None and not empty!")  # Debugging statement

        # --- Model Training ---
        st.header("Random Forest Model Training")
        random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
        random_forest_model.fit(X_train, y_train)

        y_pred = random_forest_model.predict(X_test)  # Calculate y_pred
        st.write("Model prediction successful!") # Debugging statement
    else:
        st.error("X_train is None or empty. Model training cannot be performed.")

    # --- Model Evaluation ---
    if y_pred is not None:  # Check if y_pred was successfully calculated
        st.header("Model Evaluation")
        col1, col2 = st.columns(2)
        with col1:
            mse = mean_squared_error(y_test, y_pred)
            st.metric("Mean Squared Error", f"{mse:.2f}")
        with col2:
            r2 = r2_score(y_test, y_pred)
            st.metric("R^2 Score", f"{r2:.2f}")

        # --- Feature Importance Plot ---
        st.subheader("Feature Importance")
        feature_importance = pd.Series(random_forest_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig_feature_importance, ax_feature_importance = plt.subplots(figsize=(10, 6))
        feature_importance.plot(kind='bar', ax=ax_feature_importance)
        ax_feature_importance.set_title("Feature Importance from Random Forest")
        ax_feature_importance.set_ylabel("Importance Score")
        st.pyplot(fig_feature_importance)
    else:
        st.write("Model evaluation could not be performed because the data failed to load or X_train is empty.")

    # --- Prediction Interface ---
    st.sidebar.header("Flight Details for Prediction")

    # Use original names in selectboxes
    if 'source_mapping' in locals() and 'destination_mapping' in locals() and 'airline_mapping' in locals():
        source = st.sidebar.selectbox("Source Airport", options=list(source_mapping.values()))
        destination = st.sidebar.selectbox("Destination Airport", options=list(destination_mapping.values()))
        airline = st.sidebar.selectbox("Airline", options=list(airline_mapping.values()))

        stops = st.sidebar.slider("Number of Stops", min_value=0, max_value=5, value=1)
        dep_hour = st.sidebar.slider("Departure Hour", min_value=0, max_value=23, value=10)
        dep_minute = st.sidebar.slider("Departure Minute", min_value=0, max_value=59, value=30)
        arrival_hour = st.sidebar.slider("Arrival Hour", min_value=0, max_value=23, value=13)
        arrival_minute = st.sidebar.slider("Arrival Minute", min_value=0, max_value=59, value=45)
        journey_day = st.sidebar.slider("Journey Day", min_value=1, max_value=31, value=15)
        journey_month = st.sidebar.slider("Journey Month", min_value=1, max_value=12, value=3)

        # Prediction button
        if st.sidebar.button("Predict Fare") and random_forest_model is not None:
            # Prepare input data

            # Map names back to numerical codes
            source_code = [k for k, v in source_mapping.items() if v == source][0]
            destination_code = [k for k, v in destination_mapping.items() if v == destination][0]
            airline_code = [k for k, v in airline_mapping.items() if v == airline][0]

            input_data = pd.DataFrame({
                'Airline': [airline_code],
                'Source': [source_code],
                'Destination': [destination_code],
                'Total_Stops': [stops],
                'Dep_Time_hour': [dep_hour],
                'Dep_Time_minute': [dep_minute],
                'Arrival_Time_hour': [arrival_hour],
                'Arrival_Time_minute': [arrival_minute],
                'Journey_Day': [journey_day],
                'Journey_Month': [journey_month]
            })

            # Ensure the input data has the same columns as the training data
            for col in X_train.columns:
                if col not in input_data.columns:
                    input_data[col] = 0  # Or some other appropriate default value
            input_data = input_data[X_train.columns]  # Ensure correct column order

            # Make prediction
            predicted_fare = random_forest_model.predict(input_data)[0]
            st.success(f"The predicted flight fare is: ₹{predicted_fare:.2f}")
        elif random_forest_model is None:
            st.warning("The model has not been trained yet. Please ensure the data loads correctly.")
    else:
        st.warning("Mapping dictionaries are not available. Please ensure the data loads correctly.")

    with st.expander("Data Preview"):
        if data is not None:
            st.header("Data Preview")
            st.dataframe(data.head())
        else:
            st.write("Data preview is not available because the data failed to load.")

    with st.expander("Data Summary"):
        if data is not None:
            st.header("Data Summary")
            st.write(f"Number of rows: {data.shape[0]}")
            st.write(f"Number of columns: {data.shape[1]}")
        else:
            st.write("Data summary is not available because the data failed to load.")

    with st.expander("Visualizations"):
        if data is not None:
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
            st.subheader("Price vs Number of Stops (Customized)")
            fig_scatter_custom, ax_scatter_custom = plt.subplots(figsize=(10, 6))
            scatter = ax_scatter_custom.scatter(data['Price'], data['Total_Stops'], s=80, alpha=0.7, c=data['Price'], cmap='viridis', edgecolors='black')
            ax_scatter_custom.set_title('Price vs Number of Stops', fontsize=14, fontweight='bold')
            ax_scatter_custom.set_xlabel('Price', fontsize=12)
            ax_scatter_custom.set_ylabel('Number of Stops', fontsize=12)
            ax_scatter_custom.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            fig_scatter_custom.colorbar(scatter, label='Price')  # Add colorbar to the figure
            st.pyplot(fig_scatter_custom)

            # --- ADDED: Airline Distribution Countplot with Styling ---
            st.subheader("Airline Distribution (Styled)")
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
            st.write("Visualizations are not available because the data failed to load.")
else:
    st.write("Failed to load data.  Check the GitHub URL and your internet connection.")

