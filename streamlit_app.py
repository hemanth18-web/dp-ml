import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn import metrics
import warnings
import requests

warnings.filterwarnings("ignore")

# Streamlit App Title
st.set_page_config(page_title="Flight Price Prediction App", layout="wide")
st.title("✈️ Flight Price Prediction App")
st.write("This app allows you to preprocess flight data, train models, and predict flight prices.")

# Sidebar for user inputs
st.sidebar.header("User Input for Prediction")
st.sidebar.write("Provide the details below to predict the flight price.")

# GitHub URL for the dataset
github_url = "https://raw.githubusercontent.com/hemanth18-web/dp-ml/refs/heads/main/Data_Train%20(1).csv"

# Function to load the dataset from GitHub
@st.cache_resource
def load_data_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("temp_data.csv", "wb") as f:
            f.write(response.content)  # Save the file locally
        data = pd.read_csv("temp_data.csv")  # Read the file with Pandas
        return data
    else:
        st.error("Failed to download the dataset from GitHub.")
        return None

# Load the dataset
data = load_data_from_github(github_url)

if data is not None:
    st.success("✅ Dataset loaded successfully from GitHub!")

    # Dataset Preview
    with st.expander("📊 Dataset Preview"):
        st.write("### Raw Dataset:")
        st.dataframe(data.head())

    # Preprocessing
    st.write("### 🔄 Preprocessing the Data")
    st.write("Dropping missing values...")
    data.dropna(inplace=True)

    # Copy the data for processing
    new_data = data.copy()

    # Convert columns to datetime
    def change_into_datetime(col):
        new_data[col] = pd.to_datetime(new_data[col])

    for feature in ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']:
        change_into_datetime(feature)

    # Extract day, month, and year
    new_data["Journey_day"] = new_data['Date_of_Journey'].dt.day
    new_data["Journey_month"] = new_data['Date_of_Journey'].dt.month

    # Extract hour and minute
    def extract_hour_min(df, col):
        df[col + "_hour"] = df[col].dt.hour
        df[col + "_minute"] = df[col].dt.minute

    extract_hour_min(new_data, "Dep_Time")
    extract_hour_min(new_data, "Arrival_Time")

    # Drop unnecessary columns
    new_data.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time'], axis=1, inplace=True)

    # Process duration
    def process_duration(x):
        if 'h' not in x:
            x = '0h ' + x
        elif 'm' not in x:
            x = x + ' 0m'
        return x

    new_data['Duration'] = new_data['Duration'].apply(process_duration)
    new_data['Duration_hours'] = new_data['Duration'].apply(lambda x: int(x.split(' ')[0][:-1]))
    new_data['Duration_mins'] = new_data['Duration'].apply(lambda x: int(x.split(' ')[1][:-1]))
    new_data.drop(['Duration'], axis=1, inplace=True)

    # Encode categorical columns
    new_data['Airline'] = new_data['Airline'].astype('category').cat.codes
    new_data['Source'] = new_data['Source'].astype('category').cat.codes
    new_data['Destination'] = new_data['Destination'].astype('category').cat.codes
    new_data['Total_Stops'] = new_data['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})

    # Drop unnecessary columns
    new_data.drop(['Route', 'Additional_Info'], axis=1, inplace=True)

    with st.expander("🔍 Processed Data Preview"):
        st.write("### Processed Data:")
        st.dataframe(new_data.head())

    # Feature Selection
    st.write("### 📈 Feature Importance")
    X = new_data.drop(['Price'], axis=1)
    y = new_data['Price']
    imp = mutual_info_regression(X, y)
    imp_df = pd.DataFrame(imp, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
    st.bar_chart(imp_df)

    # Train-Test Split
    st.write("### ✂️ Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    st.write(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

    # Model Training
    st.write("### 🤖 Model Training")
    rf_model = RandomForestRegressor()
    dt_model = DecisionTreeRegressor()

    rf_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)

    # Calculate training scores
    rf_training_score = rf_model.score(X_train, y_train)
    dt_training_score = dt_model.score(X_train, y_train)

    # Select the best model based on the highest training score
    if rf_training_score > dt_training_score:
        best_model = {
            "model_name": "Random Forest Regressor",
            "model": rf_model,
            "training_score": rf_training_score,
        }
    else:
        best_model = {
            "model_name": "Decision Tree Regressor",
            "model": dt_model,
            "training_score": dt_training_score,
        }

    # Display the selected model
    st.write("### 🏆 Best Model (Based on Training Score)")
    st.write(f"**Selected Model:** {best_model['model_name']}")
    st.write(f"**Training Score:** {best_model['training_score']:.2f}")

    # User Input for Prediction
    source = st.sidebar.selectbox("Source", ["Banglore", "Delhi", "Kolkata", "Mumbai", "Chennai"])
    destination = st.sidebar.selectbox("Destination", ["Banglore", "Delhi", "Kolkata", "Mumbai", "Chennai"])
    stops = st.sidebar.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])
    airline = st.sidebar.selectbox("Airline", list(data['Airline'].unique()))
    dep_hour = st.sidebar.slider("Departure Hour", 0, 23, 10)
    dep_minute = st.sidebar.slider("Departure Minute", 0, 59, 30)
    arrival_hour = st.sidebar.slider("Arrival Hour", 0, 23, 13)
    arrival_minute = st.sidebar.slider("Arrival Minute", 0, 59, 45)
    duration_hours = st.sidebar.number_input("Duration (Hours)", min_value=0, max_value=24, value=3)
    duration_minutes = st.sidebar.number_input("Duration (Minutes)", min_value=0, max_value=59, value=15)
    journey_day = st.sidebar.number_input("Journey Day", min_value=1, max_value=31, value=15)
    journey_month = st.sidebar.number_input("Journey Month", min_value=1, max_value=12, value=3)

    # Map stops to numerical values
    stop_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    stops_mapped = stop_mapping[stops]

    # Prediction
    if st.sidebar.button("Predict Price"):
        predicted_price = predict_price(
            source, destination, stops_mapped, airline, dep_hour, dep_minute,
            arrival_hour, arrival_minute, duration_hours, duration_minutes,
            journey_day, journey_month
        )
        st.sidebar.success(f"The predicted price for the flight is: ₹{predicted_price:.2f}")

else:
    st.error("Failed to load the dataset from GitHub.")
