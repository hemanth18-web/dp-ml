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
st.set_page_config(page_title="Flight Fare Predictor22", page_icon="✈️")

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

st.title("✈️ Flight Fare Prediction App22")

if data is not None:
    # --- Data Cleaning and Conversion ---

    # Replace "non-stop" with 0
    data['Total_Stops'] = data['Total_Stops'].replace("non-stop", 0)

    # Handle missing values (NaN)
    data['Total_Stops'] = data['Total_Stops'].replace('NaN', np.nan)  # Replace string 'NaN' with actual NaN
    data['Total_Stops'] = data['Total_Stops'].fillna(0)  # Fill NaN with 0 (or another appropriate value)

    # **Correctly Handle Non-Numeric Values in Total_Stops**
    def convert_stops_to_numeric(stops):
        if isinstance(stops, (int, float)):  # Check if already numeric
            return stops
        elif isinstance(stops, str) and '→' in stops:
            return len(stops.split('→'))  # Count the number of stops based on '→'
        else:
            return 0  # Default to 0 for unknown values

    data['Total_Stops'] = data['Total_Stops'].astype(str).apply(convert_stops_to_numeric) # Convert to string first to handle mixed types

    # Convert 'Total_Stops' to numeric *after* cleaning
    data['Total_Stops'] = pd.to_numeric(data['Total_Stops'], errors='coerce').fillna(0)

    # **Handle Dep_Time Column**
    try:
        # Attempt to convert 'Dep_Time' to datetime objects
        data['Dep_Time'] = pd.to_datetime(data['Dep_Time'], errors='coerce')

        # Extract hour and minute
        data['Dep_Time_hour'] = data['Dep_Time'].dt.hour
        data['Dep_Time_minute'] = data['Dep_Time'].dt.minute

        # Drop the original 'Dep_Time' column
        data.drop('Dep_Time', axis=1, inplace=True, errors='ignore')  # Use errors='ignore'

    except Exception as e:
        st.error(f"Error processing 'Dep_Time' column: {e}")

    # **Handle Arrival_Time Column**
    try:
        # Attempt to convert 'Arrival_Time' to datetime objects
        data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'], errors='coerce')

        # Extract hour and minute
        data['Arrival_Time_hour'] = data['Arrival_Time'].dt.hour
        data['Arrival_Time_minute'] = data['Arrival_Time'].dt.minute

        # Drop the original 'Arrival_Time' column
        data.drop('Arrival_Time', axis=1, inplace=True, errors='ignore')  # Use errors='ignore'

    except Exception as e:
        st.error(f"Error processing 'Arrival_Time' column: {e}")

    # **Handle Duration Column**
    def convert_duration_to_minutes(duration):
        try:
            # Use regular expression to extract hours and minutes
            match = re.match(r'(\d+)h\s*(\d+)m', str(duration))  # Added str() conversion
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                return hours * 60 + minutes
            else:
                # Handle cases where only hours or minutes are present
                match_hours = re.match(r'(\d+)h', str(duration))
                match_minutes = re.match(r'(\d+)m', str(duration))

                if match_hours:
                    hours = int(match_hours.group(1))
                    return hours * 60
                elif match_minutes:
                    minutes = int(match_minutes.group(1))
                    return minutes
                else:
                    return 0  # Default to 0 if no match

        except:
            return 0  # Handle any unexpected errors

    data['Duration_minutes'] = data['Duration'].apply(convert_duration_to_minutes)
    data.drop('Duration', axis=1, inplace=True, errors='ignore')

    # **Handle Additional_Info Column**
    data.drop('Additional_Info', axis=1, inplace=True, errors='ignore')

    # **Handle Cabin_Class Column**
    data['Cabin_Class'] = data['Cabin_Class'].astype('category').cat.codes

    # **Handle Flight_Layover Column**
    data['Flight_Layover'] = data['Flight_Layover'].astype('category').cat.codes

    # **Handle Booking_Date Column**
    try:
        data['Booking_Date'] = pd.to_datetime(data['Booking_Date'], errors='coerce')
        data['Booking_Day'] = data['Booking_Date'].dt.day
        data['Booking_Month'] = data['Booking_Date'].dt.month
        data['Booking_Year'] = data['Booking_Date'].dt.year  # Extract year as well
        data.drop('Booking_Date', axis=1, inplace=True, errors='ignore')

    except Exception as e:
        st.error(f"Error processing 'Booking_Date' column: {e}")

    # --- Feature Engineering and Encoding ---

    # Store original values for mapping in prediction
    airline_mapping = dict(enumerate(data['Airline'].astype('category').cat.categories))
    source_mapping = dict(enumerate(data['Source'].astype('category').cat.categories))
    destination_mapping = dict(enumerate(data['Destination'].astype('category').cat.categories))

    # Convert categorical features to numerical
    for col in ['Airline', 'Source', 'Destination']:
        data[col] = data[col].astype('category').cat.codes

    # Extract date features
    data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], errors='coerce') # Handle potential parsing errors
    data['Journey_Day'] = data['Date_of_Journey'].dt.day
    data['Journey_Month'] = data['Date_of_Journey'].dt.month
    #data.drop('Date_of_Journey', axis=1, inplace=True)

    # --- Data Preparation for Modeling ---
    # Drop any columns with non-finite values (NaN, inf, -inf)
    data = data.dropna(axis=1, how='any')

    # **General Column Check**
    for col in data.columns:
        st.write(f"Column: {col}")
        st.write(f"  Data Type: {data[col].dtype}")
        st.write(f"  Unique Values: {data[col].nunique()}")
        #st.write(f"  First 5 Values: {data[col].head().to_list()}") # Print first 5 values
        try:
            st.write(f"  Min: {data[col].min()}, Max: {data[col].max()}")
        except:
            st.write("  Cannot calculate min/max for this data type.")
        st.write("-" * 30)

    # Ensure all columns are numeric
    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except ValueError:
            st.error(f"Could not convert column '{col}' to numeric.  Please investigate.")
            st.stop()  # Stop execution if a column cannot be converted

    X = data.drop(['Price'], axis=1, errors='ignore') # Ignore if 'Price' is already dropped
    y = data['Price']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Training ---
    st.header("Random Forest Model Training")
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train, y_train)

    # --- Model Evaluation ---
    y_pred = random_forest_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R^2 Score: {r2:.2f}")

    # --- Feature Importance Plot ---
    st.subheader("Feature Importance")
    feature_importance = pd.Series(random_forest_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_feature_importance, ax_feature_importance = plt.subplots(figsize=(10, 6))
    feature_importance.plot(kind='bar', ax=ax_feature_importance)
    ax_feature_importance.set_title("Feature Importance from Random Forest")
    ax_feature_importance.set_ylabel("Importance Score")
    st.pyplot(fig_feature_importance)

    # --- Prediction Interface ---
    st.header("Flight Fare Prediction")

    # Input fields
    # Use original names in selectboxes
    source = st.selectbox("Source", options=list(source_mapping.values()))
    destination = st.selectbox("Destination", options=list(destination_mapping.values()))
    airline = st.selectbox("Airline", options=list(airline_mapping.values()))

    stops = st.slider("Number of Stops", min_value=0, max_value=5, value=1)
    dep_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=10)
    dep_minute = st.slider("Departure Minute", min_value=0, max_value=59, value=30)
    arrival_hour = st.slider("Arrival Hour", min_value=0, max_value=23, value=13)
    arrival_minute = st.slider("Arrival Minute", min_value=0, max_value=59, value=45)
    journey_day = st.slider("Journey Day", min_value=1, max_value=31, value=15)
    journey_month = st.slider("Journey Month", min_value=1, max_value=12, value=3)

    # Prediction button
    if st.button("Predict Fare"):
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
        st.success(f"Predicted Flight Fare: ₹{predicted_fare:.2f}")

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
