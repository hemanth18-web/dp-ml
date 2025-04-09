import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# --- LOAD DATA ---
github_url = "https://raw.githubusercontent.com/hemanth18-web/dp-ml/refs/heads/main/Data_Train%20(1).csv"

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

data = load_data_from_github(github_url)

# --- DATA PREPROCESSING (Simplified for demonstration) ---
def preprocess_data(data):
    # Handle missing values (replace with mean for numerical columns)
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].mean())

    # Convert 'Date_of_Journey' to datetime and extract features
    if 'Date_of_Journey' in data.columns:
        try:
            data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], errors='coerce')
            data['Day'] = data['Date_of_Journey'].dt.day
            data['Month'] = data['Date_of_Journey'].dt.month
            data['Year'] = data['Date_of_Journey'].dt.year
            data.drop("Date_of_Journey", inplace=True, axis=1, errors='ignore')
        except Exception as e:
            st.error(f"Error processing 'Date_of_Journey': {e}")

    # One-hot encode categorical features (handle missing columns)
    categorical_cols = data.select_dtypes(include='object').columns
    data = pd.get_dummies(data, columns=categorical_cols, dummy_na=False)  # dummy_na=False to avoid creating extra columns for NaN

    return data

if data is not None:
    data = preprocess_data(data.copy())  # Process a copy to avoid modifying the original cached data

    # --- MODEL TRAINING ---
    X = data.drop(['Price'], axis=1, errors='ignore')  # Drop 'Price' if it exists
    y = data['Price'] if 'Price' in data.columns else None  # Check if 'Price' exists

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        # Train Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_y_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_y_pred)

        # Determine the best model based on R2 score
        if rf_r2 > r2:
            best_model = rf_model
            best_model_name = "Random Forest Regressor"
            best_r2 = rf_r2
        else:
            best_model = model
            best_model_name = "Linear Regression"
            best_r2 = r2

        st.success(f"Model training complete. Best model: {best_model_name} with R2 score: {best_r2:.2f}")

        # --- USER INPUT AND PREDICTION ---
        st.sidebar.header("Prediction Input")

        # Create input fields for all features in the dataset
        input_data = {}
        for col in X.columns:
            if X[col].dtype == 'float64' or X[col].dtype == 'int64':
                input_data[col] = st.sidebar.number_input(f"{col}", value=X[col].mean())
            else:
                input_data[col] = st.sidebar.text_input(f"{col}", value="N/A")  # Handle non-numeric features

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data], columns=X.columns)

        if st.sidebar.button("Predict Price"):
            try:
                prediction = best_model.predict(input_df)[0]
                st.success(f"Predicted Price: {prediction:.2f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}.  Make sure your input values are valid.")

    else:
        st.warning("The 'Price' column is missing.  Cannot train the model or make predictions.")

    # --- DATA EXPLORATION AND VISUALIZATION (Simplified) ---
    st.header("Data Preview")
    st.dataframe(data.head())

    st.header("Data Summary")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")

    st.header("Price Distribution")
    if 'Price' in data.columns:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        sns.histplot(data['Price'], kde=True, ax=ax_hist)
        st.pyplot(fig_hist)
    else:
        st.warning("Price column not found, cannot display price distribution.")

else:
    st.write("Failed to load data. Check the GitHub URL and your internet connection.")
