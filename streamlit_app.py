import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import pickle

# Function to calculate MAPE
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to evaluate models
def evaluate_model(ml_model, model_name, X_train, y_train, X_test, y_test):
    model = ml_model.fit(X_train, y_train)
    training_score = model.score(X_train, y_train)
    
    y_prediction = model.predict(X_test)
    r2_score = metrics.r2_score(y_test, y_prediction)
    mae = metrics.mean_absolute_error(y_test, y_prediction)
    mse = metrics.mean_squared_error(y_test, y_prediction)
    rmse = np.sqrt(mse)
    mape_value = mape(y_test, y_prediction)
    
    # Store all results in a dictionary
    results = {
        "model_name": model_name,
        "training_score": training_score,
        "r2_score": r2_score,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape_value,
        "residuals": y_test - y_prediction
    }
    
    return results

# Streamlit App
st.title("Flight Price Prediction App")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (Excel format)", type=["xlsx"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Preprocessing
    st.subheader("Data Preprocessing")
    data.dropna(inplace=True)
    data['Journey_day'] = pd.to_datetime(data['Date_of_Journey']).dt.day
    data['Journey_month'] = pd.to_datetime(data['Date_of_Journey']).dt.month
    data['Dep_Time_hour'] = pd.to_datetime(data['Dep_Time']).dt.hour
    data['Dep_Time_minute'] = pd.to_datetime(data['Dep_Time']).dt.minute
    data['Arrival_Time_hour'] = pd.to_datetime(data['Arrival_Time']).dt.hour
    data['Arrival_Time_minute'] = pd.to_datetime(data['Arrival_Time']).dt.minute
    data['Duration_total_mins'] = data['Duration'].str.replace('h', '*60').str.replace('m', '*1').str.replace(' ', '+').apply(eval)
    data['Total_Stops'] = data['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
    data['Destination'].replace('New Delhi', 'Delhi', inplace=True)
    data.drop(columns=['Date_of_Journey', 'Route', 'Additional_Info', 'Duration', 'Source'], inplace=True)

    st.write("Processed Dataset:")
    st.dataframe(data.head())

    # Splitting Data
    X = data.drop(['Price'], axis=1)
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Model Training
    st.subheader("Model Training and Evaluation")
    rf_model = RandomForestRegressor()
    dt_model = DecisionTreeRegressor()

    rf_results = evaluate_model(rf_model, "Random Forest Regressor", X_train, y_train, X_test, y_test)
    dt_results = evaluate_model(dt_model, "Decision Tree Regressor", X_train, y_train, X_test, y_test)

    # Display Results
    st.write("Random Forest Results:")
    st.json(rf_results)

    st.write("Decision Tree Results:")
    st.json(dt_results)

    # Determine Best Model
    if rf_results["training_score"] > dt_results["training_score"]:
        best_model = rf_results
    else:
        best_model = dt_results

    st.write(f"Best Model: {best_model['model_name']}")
    st.write(f"Training Score: {best_model['training_score']}")
    st.write(f"R2 Score: {best_model['r2_score']}")
    st.write(f"MAE: {best_model['mae']}")
    st.write(f"MSE: {best_model['mse']}")
    st.write(f"RMSE: {best_model['rmse']}")
    st.write(f"MAPE: {best_model['mape']}")

    # Plot Residuals
    st.subheader("Residuals Plot")
    fig, ax = plt.subplots()
    sns.histplot(best_model["residuals"], kde=True, ax=ax)
    st.pyplot(fig)

    # Save Model
    st.subheader("Save the Model")
    if st.button("Save Random Forest Model"):
        with open("rf_model.pkl", "wb") as f:
            pickle.dump(rf_model, f)
        st.success("Model saved as rf_model.pkl")
