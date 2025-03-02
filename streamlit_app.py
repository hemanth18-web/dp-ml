import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor



# Load the data and mappings
dict_airlines = {
    'Jet Airways': 0,
    'IndiGo': 1,
    'Air India': 2,
    'Multiple carriers': 3,
    'SpiceJet': 4,
    'Vistara': 5,
    'GoAir': 6,
    'Multiple carriers Premium economy': 7,
    'Jet Airways Business': 8,
    'Vistara Premium economy': 9,
    'Trujet': 10
}

stop_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}

# Provide the raw URL of the dataset from GitHub
github_url = "https://raw.githubusercontent.com/hemanth18-web/dp-ml/main/Data_Train(1).xlsx"

# Function to download the dataset from GitHub
@st.cache_resource
def load_data_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("temp_data.xlsx", "wb") as f:
            f.write(response.content)
        data = pd.read_excel("temp_data.xlsx")
        return data
    else:
        st.error("Failed to download the dataset from GitHub.")
        return None

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file is not None:
    # If a file is uploaded, use it
    data = pd.read_excel(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    # If no file is uploaded, use the default dataset from GitHub
    st.warning("No file uploaded. Using default dataset from GitHub.")
    data = load_data_from_github(github_url)

# Display the dataset
if data is not None:
    st.write("Dataset Preview:")
    st.dataframe(data.head())

# Define the prediction function
def predict_price_with_best_model(source, destination, stops, airline, dep_hour, dep_minute, arrival_hour, arrival_minute, duration_hours, duration_minutes, journey_day, journey_month):
    """
    Predict the flight price based on user input for source, destination, stops, airline, and time details.
    """
    # Map the input values to the encoded values used in the model
    source_col = f"Source_{source}"
    destination_col = f"Destination_{destination}"
    
    # Create a dictionary for the input data
    input_data = {
        "Total_Stops": stops,
        "Airline": dict_airlines.get(airline, -1),  # Map airline to its encoded value
        "Dep_Time_hour": dep_hour,
        "Dep_Time_minute": dep_minute,
        "Arrival_Time_hour": arrival_hour,
        "Arrival_Time_minute": arrival_minute,
        "Duration_hours": duration_hours,
        "Duration_mins": duration_minutes,
        "Journey_day": journey_day,
        "Journey_month": journey_month,
    }
    
    # Add one-hot encoded columns for Source and Destination
    for col in data.columns:  # `data` is the dataset used for the model
        if col.startswith("Source_"):
            input_data[col] = 1 if col == source_col else 0
        if col.startswith("Destination_"):
            input_data[col] = 1 if col == destination_col else 0
    
    # Ensure all features used during training are present in the input data
    for col in data.columns:  # `data` is the dataset used for the model
        if col not in input_data:
            input_data[col] = 0  # Add missing features with a default value of 0
    
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Predict the price using the best model
    predicted_price = best_model.predict(input_df)[0]
    
    return predicted_price

# Streamlit App
st.title("Flight Price Prediction App")
st.write("Enter the flight details below to predict the price.")

# Input fields for the user
source = st.selectbox("Source", ["Banglore", "Delhi", "Kolkata", "Mumbai", "Chennai"])
destination = st.selectbox("Destination", ["Banglore", "Delhi", "Kolkata", "Mumbai", "Chennai"])
stops = st.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])
airline = st.selectbox("Airline", list(dict_airlines.keys()))
dep_hour = st.slider("Departure Hour", 0, 23, 10)
dep_minute = st.slider("Departure Minute", 0, 59, 30)
arrival_hour = st.slider("Arrival Hour", 0, 23, 13)
arrival_minute = st.slider("Arrival Minute", 0, 59, 45)
duration_hours = st.number_input("Duration (Hours)", min_value=0, max_value=24, value=3)
duration_minutes = st.number_input("Duration (Minutes)", min_value=0, max_value=59, value=15)
journey_day = st.number_input("Journey Day", min_value=1, max_value=31, value=15)
journey_month = st.number_input("Journey Month", min_value=1, max_value=12, value=3)

# Predict button
if st.button("Predict Price"):
    # Map stops to numeric values
    stops_mapped = stop_mapping[stops]
    
    # Predict the price
    predicted_price = predict_price_with_best_model(
        source, destination, stops_mapped, airline, dep_hour, dep_minute,
        arrival_hour, arrival_minute, duration_hours, duration_minutes,
        journey_day, journey_month
    )
    
    # Display the result
    st.success(f"The predicted price for the flight is: ₹{predicted_price:.2f}")
