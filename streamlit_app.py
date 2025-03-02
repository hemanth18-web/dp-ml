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
st.title("Flight Price Prediction App")
st.write("This app allows you to preprocess flight data, train models, and predict flight prices.")

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
    st.success("Dataset loaded successfully from GitHub!")
    st.write("### Dataset Preview:")
    st.dataframe(data.head())

    # Preprocessing
    st.write("### Preprocessing the Data")
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
    st.write("Encoding categorical columns...")
    new_data['Airline'] = new_data['Airline'].astype('category').cat.codes
    new_data['Source'] = new_data['Source'].astype('category').cat.codes
    new_data['Destination'] = new_data['Destination'].astype('category').cat.codes
    new_data['Total_Stops'] = new_data['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})

    # Drop unnecessary columns
    new_data.drop(['Route', 'Additional_Info'], axis=1, inplace=True)

    st.write("### Processed Data Preview:")
    st.dataframe(new_data.head())

    # Feature Selection
    st.write("### Feature Importance")
    X = new_data.drop(['Price'], axis=1)
    y = new_data['Price']
    imp = mutual_info_regression(X, y)
    imp_df = pd.DataFrame(imp, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
    st.bar_chart(imp_df)

    # Train-Test Split
    st.write("### Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    st.write(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

    # Model Training
    st.write("### Model Training")
    rf_model = RandomForestRegressor()
    dt_model = DecisionTreeRegressor()

    rf_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)

    # Evaluate models
    rf_r2 = metrics.r2_score(y_test, rf_model.predict(X_test))
    dt_r2 = metrics.r2_score(y_test, dt_model.predict(X_test))

    # Determine the best model
    if rf_r2 > dt_r2:
        best_model = rf_model
        best_model_name = "Random Forest Regressor"
    else:
        best_model = dt_model
        best_model_name = "Decision Tree Regressor"

    # Display the best model
    st.write("The best model is:", best_model_name)
    st.write(f"Random Forest R2 Score: {rf_r2:.2f}")
    st.write(f"Decision Tree R2 Score: {dt_r2:.2f}")

    # Prediction Function
    def predict_price(source, destination, stops, airline, dep_hour, dep_minute, arrival_hour, arrival_minute, duration_hours, duration_minutes, journey_day, journey_month):
        """
        Predict the flight price based on user input.
        """
        # Map categorical inputs to their encoded values
        source_mapping = {"Banglore": 0, "Delhi": 1, "Kolkata": 2, "Mumbai": 3, "Chennai": 4}
        destination_mapping = {"Banglore": 0, "Delhi": 1, "Kolkata": 2, "Mumbai": 3, "Chennai": 4}

        # Encode the inputs
        source_encoded = source_mapping[source]
        destination_encoded = destination_mapping[destination]

        # Create a dictionary for the input data
        input_data = {
            "Source": source_encoded,
            "Destination": destination_encoded,
            "Total_Stops": stops,
            "Airline": airline,
            "Dep_Time_hour": dep_hour,
            "Dep_Time_minute": dep_minute,
            "Arrival_Time_hour": arrival_hour,
            "Arrival_Time_minute": arrival_minute,
            "Duration_hours": duration_hours,
            "Duration_mins": duration_minutes,
            "Journey_day": journey_day,
            "Journey_month": journey_month
        }

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Align the input data with the training data (X_train)
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns with default value 0
        input_df = input_df[X.columns]  # Reorder columns to match X_train

        # Predict the price using the best model
        predicted_price = best_model.predict(input_df)[0]
        return predicted_price

    # User Input for Prediction
    st.write("### Predict Flight Price")
    source = st.selectbox("Source", ["Banglore", "Delhi", "Kolkata", "Mumbai", "Chennai"])
    destination = st.selectbox("Destination", ["Banglore", "Delhi", "Kolkata", "Mumbai", "Chennai"])
    stops = st.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])
    airline = st.selectbox("Airline", new_data['Airline'].unique())
    dep_hour = st.slider("Departure Hour", 0, 23, 10)
    dep_minute = st.slider("Departure Minute", 0, 59, 30)
    arrival_hour = st.slider("Arrival Hour", 0, 23, 13)
    arrival_minute = st.slider("Arrival Minute", 0, 59, 45)
    duration_hours = st.number_input("Duration (Hours)", min_value=0, max_value=24, value=3)
    duration_minutes = st.number_input("Duration (Minutes)", min_value=0, max_value=59, value=15)
    journey_day = st.number_input("Journey Day", min_value=1, max_value=31, value=15)
    journey_month = st.number_input("Journey Month", min_value=1, max_value=12, value=3)

    stop_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    stops_mapped = stop_mapping[stops]

    if st.button("Predict Price"):
        predicted_price = predict_price(
            source, destination, stops_mapped, airline, dep_hour, dep_minute,
            arrival_hour, arrival_minute, duration_hours, duration_minutes,
            journey_day, journey_month
        )
        st.success(f"The predicted price for the flight is: ₹{predicted_price:.2f}")

else:
    st.error("Failed to load the dataset from GitHub.")
