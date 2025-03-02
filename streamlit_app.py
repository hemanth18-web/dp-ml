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

warnings.filterwarnings("ignore")

# Streamlit App Title
st.title("Flight Price Prediction App")
st.write("This app allows you to preprocess flight data, train models, and predict flight prices.")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_excel(uploaded_file)
    st.success("Dataset uploaded successfully!")
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

    rf_train_score = rf_model.score(X_train, y_train)
    dt_train_score = dt_model.score(X_train, y_train)

    st.write(f"Random Forest Training Score: {rf_train_score:.2f}")
    st.write(f"Decision Tree Training Score: {dt_train_score:.2f}")

    # Model Evaluation
    st.write("### Model Evaluation")
    rf_y_pred = rf_model.predict(X_test)
    dt_y_pred = dt_model.predict(X_test)

    rf_r2 = metrics.r2_score(y_test, rf_y_pred)
    dt_r2 = metrics.r2_score(y_test, dt_y_pred)

    st.write(f"Random Forest R2 Score: {rf_r2:.2f}")
    st.write(f"Decision Tree R2 Score: {dt_r2:.2f}")

    # Select the best model
    best_model = rf_model if rf_r2 > dt_r2 else dt_model
    st.write(f"Best Model: {'Random Forest' if rf_r2 > dt_r2 else 'Decision Tree'}")

    # Prediction Function
    def predict_price(source, destination, stops, airline, dep_hour, dep_minute, arrival_hour, arrival_minute, duration_hours, duration_minutes, journey_day, journey_month):
        input_data = {
            "Source": source,
            "Destination": destination,
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
        input_df = pd.DataFrame([input_data])
        return best_model.predict(input_df)[0]

    # User Input for Prediction
    st.write("### Predict Flight Price")
    source = st.selectbox("Source", new_data['Source'].unique())
    destination = st.selectbox("Destination", new_data['Destination'].unique())
    stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])
    airline = st.selectbox("Airline", new_data['Airline'].unique())
    dep_hour = st.slider("Departure Hour", 0, 23, 10)
    dep_minute = st.slider("Departure Minute", 0, 59, 30)
    arrival_hour = st.slider("Arrival Hour", 0, 23, 13)
    arrival_minute = st.slider("Arrival Minute", 0, 59, 45)
    duration_hours = st.number_input("Duration (Hours)", min_value=0, max_value=24, value=3)
    duration_minutes = st.number_input("Duration (Minutes)", min_value=0, max_value=59, value=15)
    journey_day = st.number_input("Journey Day", min_value=1, max_value=31, value=15)
    journey_month = st.number_input("Journey Month", min_value=1, max_value=12, value=3)

    if st.button("Predict Price"):
        predicted_price = predict_price(source, destination, stops, airline, dep_hour, dep_minute, arrival_hour, arrival_minute, duration_hours, duration_minutes, journey_day, journey_month)
        st.success(f"The predicted price for the flight is: ₹{predicted_price:.2f}")

else:
    st.warning("Please upload a dataset to proceed.")
