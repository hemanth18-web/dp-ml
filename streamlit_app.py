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

# Option to upload a dataset or use the default dataset from GitHub
st.subheader("Upload Dataset or Use Default Dataset")

# Provide the raw URL of the dataset from GitHub
github_url = "https://raw.githubusercontent.com/hemanth18-web/dp-ml/main/Data_Train(1).xlsx"


# File uploader
uploaded_file = st.file_uploader("Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file is not None:
    # If a file is uploaded, use it
    data = pd.read_excel(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    # If no file is uploaded, use the default dataset from GitHub
    st.warning("No file uploaded. Using default dataset from GitHub.")
    @st.cache
    def load_data():
        data = pd.read_excel(github_url)
        return data
    data = load_data()

# Display the dataset
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

# Feature Selection
st.subheader("Feature Selection")
all_features = list(data.columns)
all_features.remove('Price')  # Remove the target column
selected_features = st.multiselect("Select features for training:", all_features, default=all_features)

# Splitting Data
X = data[selected_features]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model Training
st.subheader("Model Training and Evaluation")

# Random Forest Hyperparameters
n_estimators = st.slider("Number of Trees (n_estimators):", min_value=10, max_value=500, value=100, step=10)
max_depth = st.slider("Maximum Depth (max_depth):", min_value=1, max_value=50, value=10, step=1)

# Models
rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
dt_model = DecisionTreeRegressor()

# Choose Model
model_choice = st.selectbox("Select a model:", ["Random Forest", "Decision Tree"])
if model_choice == "Random Forest":
    model = rf_model
else:
    model = dt_model

# Evaluate Model
results = evaluate_model(model, model_choice, X_train, y_train, X_test, y_test)

# Display Results
st.write(f"{model_choice} Results:")
st.json(results)

# Plot Residuals
st.subheader("Residuals Plot")
fig, ax = plt.subplots()
sns.histplot(results["residuals"], kde=True, ax=ax)
st.pyplot(fig)

# Save Model
st.subheader("Save the Model")
if st.button("Save Model"):
    with open(f"{model_choice.lower()}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success(f"Model saved as {model_choice.lower()}_model.pkl")
