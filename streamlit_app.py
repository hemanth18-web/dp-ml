import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import re  # Import the regular expression module
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

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
st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    /* General styling */
    body {
        font-family: 'Helvetica Neue', sans-serif;
        color: #333;
        background-color: #f9f9f9;
    }
    .stApp {
        max-width: 1600px;
        margin: 0 auto;
        padding: 20px;
    }
    /* Header styling */
    h1 {
        color: #2E86C1;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Sidebar styling */
    .stSidebar {
        background-color: #EBF5FB;
        padding: 20px;
        border-radius: 10px;
    }
    .stSidebar h2 {
        color: #2E86C1;
        margin-bottom: 20px;
    }
    /* Input elements styling */
    .stSelectbox, .stSlider, .stDateInput {
        margin-bottom: 20px;
    }
    /* Button styling */
    .stButton > button {
        background-color: #2E86C1;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #1A5276;
    }
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        background-color: white;
    }
    /* Metric styling */
    .stMetric {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.05);
    }
    .stMetric label {
        font-weight: bold;
        color: #555;
    }
    .stMetric > div:nth-child(2) {
        font-size: 24px;
        color: #2E86C1;
    }
    /* Visualization styling */
    .stPlot {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA PREPROCESSING (Move this outside the conditional page logic) ---
if data is not None:
    # --- Data Cleaning and Conversion ---
    data['Total_Stops'] = data['Total_Stops'].replace("non-stop", 0)
    data['Total_Stops'] = data['Total_Stops'].replace('NaN', np.nan)
    data['Total_Stops'] = data['Total_Stops'].fillna(0)

    def convert_stops_to_numeric(stops):
        if isinstance(stops, (int, float)):
            return stops
        elif isinstance(stops, str) and '→' in stops:
            return len(stops.split('→'))
        else:
            return 0
    data['Total_Stops'] = data['Total_Stops'].astype(str).apply(convert_stops_to_numeric)
    data['Total_Stops'] = pd.to_numeric(data['Total_Stops'], errors='coerce').fillna(0)

    try:
        data['Dep_Time'] = pd.to_datetime(data['Dep_Time'], errors='coerce')
        data['Dep_Time_hour'] = data['Dep_Time'].dt.hour
        data['Dep_Time_minute'] = data['Dep_Time'].dt.minute
        data.drop('Dep_Time', axis=1, inplace=True, errors='ignore')
    except Exception as e:
        st.error(f"Error processing 'Dep_Time' column: {e}")

    try:
        data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'], errors='coerce')
        data['Arrival_Time_hour'] = data['Arrival_Time'].dt.hour
        data['Arrival_Time_minute'] = data['Arrival_Time'].dt.minute
        data.drop('Arrival_Time', axis=1, inplace=True, errors='ignore')
    except Exception as e:
        st.error(f"Error processing 'Arrival_Time' column: {e}")

    def convert_duration_to_minutes(duration):
        try:
            match = re.match(r'(\d+)h\s*(\d+)m', str(duration))
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                return hours * 60 + minutes
            else:
                match_hours = re.match(r'(\d+)h', str(duration))
                match_minutes = re.match(r'(\d+)m', str(duration))
                if match_hours:
                    hours = int(match_hours.group(1))
                    return hours * 60
                elif match_minutes:
                    minutes = int(match_minutes.group(1))
                    return minutes
                else:
                    return 0
        except:
            return 0
    data['Duration_minutes'] = data['Duration'].apply(convert_duration_to_minutes)
    data.drop('Duration', axis=1, inplace=True, errors='ignore')

    data.drop('Additional_Info', axis=1, inplace=True, errors='ignore')
    data['Cabin_Class'] = data['Cabin_Class'].astype('category').cat.codes
    data['Flight_Layover'] = data['Flight_Layover'].astype('category').cat.codes

    try:
        data['Booking_Date'] = pd.to_datetime(data['Booking_Date'], errors='coerce')
        data['Booking_Day'] = data['Booking_Date'].dt.day
        data['Booking_Month'] = data['Booking_Date'].dt.month
        data['Booking_Year'] = data['Booking_Date'].dt.year
        data.drop('Booking_Date', axis=1, inplace=True, errors='ignore')
    except Exception as e:
        st.error(f"Error processing 'Booking_Date' column: {e}")

    airline_mapping = dict(enumerate(data['Airline'].astype('category').cat.categories))
    source_mapping = dict(enumerate(data['Source'].astype('category').cat.categories))
    destination_mapping = dict(enumerate(data['Destination'].astype('category').cat.categories))

    for col in ['Airline', 'Source', 'Destination']:
        data[col] = data[col].astype('category').cat.codes

    data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], errors='coerce')
    data['Journey_Day'] = data['Date_of_Journey'].dt.day
    data['Journey_Month'] = data['Date_of_Journey'].dt.month

    data = data.dropna(axis=1, how='any')

    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except ValueError:
            st.error(f"Could not convert column '{col}' to numeric.  Please investigate.")
            st.stop()

    X = data.drop(['Price'], axis=1, errors='ignore')
    y = data['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Training (Train the model *once* outside the prediction) ---
    @st.cache_resource  # Use cache_resource for models
    def train_model(X_train, y_train):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    random_forest_model = train_model(X_train, y_train)  # Train the model

    # --- Sidebar for navigation ---
    with st.sidebar:
        st.title("Flight Fare Prediction")
        st.markdown("Explore flight data and predict fares.")
        page = st.radio("Choose a section:", ["Data Overview", "Prediction", "Visualizations", "Model Evaluation"])

    # --- Main App Content ---
    st.title("✈️ Flight Fare Prediction App")

    if page == "Data Overview":
        st.header("Data Overview")
        st.dataframe(data.head(10))  # Show first 10 rows
        st.write(f"Number of rows: {data.shape[0]}")
        st.write(f"Number of columns: {data.shape[1]}")

    elif page == "Prediction":
        st.header("Flight Fare Prediction")

        # Input fields using columns for layout
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Source", options=list(source_mapping.values()), help="Select the origin city")
            destination = st.selectbox("Destination", options=list(destination_mapping.values()), help="Select the destination city")
            airline = st.selectbox("Airline", options=list(airline_mapping.values()), help="Select the airline")
            stops = st.slider("Number of Stops", min_value=0, max_value=5, value=0, help="Number of layovers")

        with col2:
            # Use date_input for journey date
            journey_date = st.date_input("Journey Date", datetime.now(), help="Select the date of travel")
            dep_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=12, help="Hour of departure")
            dep_minute = st.slider("Departure Minute", min_value=0, max_value=59, value=0, help="Minute of departure")
            arrival_hour = st.slider("Arrival Hour", min_value=0, max_value=23, value=15, help="Hour of arrival")
            arrival_minute = st.slider("Arrival Minute", min_value=0, max_value=59, value=0, help="Minute of arrival")

        if st.button("Predict Fare"):
            # Prepare input data
            source_code = [k for k, v in source_mapping.items() if v == source][0]
            destination_code = [k for k, v in destination_mapping.items() if v == destination][0]
            airline_code = [k for k, v in airline_mapping.items() if v == airline][0]

            # Extract day and month from journey_date
            journey_day = journey_date.day
            journey_month = journey_date.month

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

            for col in X_train.columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[X_train.columns]

            # Make prediction
            predicted_fare = random_forest_model.predict(input_data)[0]
            st.success(f"Predicted Flight Fare: ₹{predicted_fare:.2f}")

    elif page == "Visualizations":
        st.header("Data Visualizations")

        # Airline Distribution
        st.subheader("Airline Distribution")
        fig_airline, ax_airline = plt.subplots(figsize=(10, 6))
        sns.countplot(x="Airline", data=data, ax=ax_airline, palette="viridis", order=data['Airline'].value_counts().index)
        ax_airline.tick_params(axis='x', rotation=90)
        st.pyplot(fig_airline)

        # Price vs. Number of Stops
        st.subheader("Price vs. Number of Stops")
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="Price", y="Total_Stops", data=data, ax=ax_scatter, alpha=0.7)
        st.pyplot(fig_scatter)

        # Price Distribution
        st.subheader("Price Distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        sns.histplot(data['Price'], kde=True, ax=ax_hist, color="#2E86C1")
        st.pyplot(fig_hist)

    elif page == "Model Evaluation":
        st.header("Model Evaluation")
        y_pred = random_forest_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.2f}")
        with col2:
            st.metric("R^2 Score", f"{r2:.2f}")

        st.subheader("Feature Importance")
        feature_importance = pd.Series(random_forest_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig_feature_importance, ax_feature_importance = plt.subplots(figsize=(10, 6))
        feature_importance.plot(kind='bar', ax=ax_feature_importance, color="#2E86C1")
        ax_feature_importance.set_title("Feature Importance from Random Forest")
        ax_feature_importance.set_ylabel("Importance Score")
        st.pyplot(fig_feature_importance)

else:
    st.error("Failed to load data. Check the GitHub URL and your internet connection.")
