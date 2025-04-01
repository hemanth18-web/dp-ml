import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# --- Load Data ---
@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_data(file_path):
    try:
        data = pd.read_excel(file_path, engine='openpyxl')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Data Preprocessing ---
def preprocess_data(data):
    # Cabin Class (Random Assignment - Replace with actual logic if available)
    cabin_classes = ['Economy', 'Business', 'First Class']
    data['Cabin_Class'] = np.random.choice(cabin_classes, size=len(data))

    # Number of Stops and Flight Layover
    def get_layover(route):
        stops = route.count('→')
        return stops, "Direct" if stops == 0 else f"{stops} Stop(s)"

    data[['Number_of_Stops', 'Flight_Layover']] = data['Route'].astype(str).apply(lambda x: pd.Series(get_layover(x)))

    # Date Conversions
    data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], format='%d/%m/%Y')
    data['Booking_Date'] = data['Date_of_Journey'] - pd.to_timedelta(np.random.randint(1, 60, size=len(data)), unit='D')
    data['Days_Until_Departure'] = (data['Date_of_Journey'] - data['Booking_Date']).dt.days

    # Fuel Price (Random Generation - Replace with actual data if available)
    fuel_price_data = pd.DataFrame({
        'Date': pd.date_range(start='2019-01-01', periods=365, freq='D'),
        'Fuel_Price_INR': np.random.uniform(60000, 90000, size=365)
    })
    fuel_price_data['Date'] = pd.to_datetime(fuel_price_data['Date'])
    data = data.merge(fuel_price_data, left_on='Booking_Date', right_on='Date', how='left', suffixes=('', '_fuel'))
    data.rename(columns={'Fuel_Price_INR': 'ATF_Price_INR'}, inplace=True)
    data.drop(columns=['Date'], inplace=True)

    # Extract Date Features
    data['Day'] = data['Date_of_Journey'].dt.day
    data['Month'] = data['Date_of_Journey'].dt.month
    data['Year'] = data['Date_of_Journey'].dt.year

    # Extract Time Features
    data['Dep_Time'] = pd.to_datetime(data['Dep_Time'])
    data['Dep_Hour'] = data['Dep_Time'].dt.hour
    data['Dep_Minute'] = data['Dep_Time'].dt.minute

    data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'])
    data['Arrival_Hour'] = data['Arrival_Time'].dt.hour
    data['Arrival_Minute'] = data['Arrival_Time'].dt.minute

    # Duration
    def preprocess_duration(x):
        if 'h' not in x:
            x = '0h ' + x
        elif 'm' not in x:
            x = x + ' 0m'
        return x

    data['Duration'] = data['Duration'].astype(str).apply(preprocess_duration)
    data['Duration_hours'] = data['Duration'].str.split(' ').str[0].str.replace('h', '').astype(int)
    data['Duration_minutes'] = data['Duration'].str.split(' ').str[1].str.replace('m', '').astype(int)
    data['Duration_total_mins'] = data['Duration'].str.replace('h', "*60").str.replace(' ', '+').str.replace('m', "*1").apply(eval)

    # Categorical Encoding
    airlines = data.groupby(['Airline'])['Price'].mean().sort_values().index
    dict_airlines = {key: index for index, key in enumerate(airlines, 0)}
    data['Airline'] = data['Airline'].map(dict_airlines)

    data['Destination'] = data['Destination'].replace('New Delhi', 'Delhi')
    destination = data.groupby(['Destination'])['Price'].mean().sort_values().index
    dict_destination = {key: index for index, key in enumerate(destination, 0)}
    data['Destination'] = data['Destination'].map(dict_destination)

    Total_Stops = data['Total_Stops'].unique()
    dict_Total_Stops = {key: index for index, key in enumerate(Total_Stops, 0)}
    data['Total_Stops'] = data['Total_Stops'].map(dict_Total_Stops)

    Cabin_Class = data['Cabin_Class'].unique()
    dict_Cabin_Class = {key: index for index, key in enumerate(Cabin_Class, 0)}
    data['Cabin_Class'] = data['Cabin_Class'].map(dict_Cabin_Class)

    # Drop Unnecessary Columns
    cols_to_drop = ['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Additional_Info', 'Source', 'Route', 'Duration', 'Flight_Layover']
    data.drop(columns=cols_to_drop, axis=1, inplace=True)

    return data

# --- Model Training and Prediction ---
def train_and_predict(data, model_name):
    X = data.drop(['Price'], axis=1)
    y = data['Price']

    # Convert 'Booking_Date' to ordinal
    X['Booking_Date'] = pd.to_datetime(X['Booking_Date']).apply(lambda date: date.toordinal())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return r2, mae, mse, rmse, mape, model

# --- Visualization Functions ---
def plot_price_distribution(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data['Price'], kde=True, ax=ax)
    ax.set_title('Price Distribution')
    st.pyplot(fig)

def plot_feature_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        fig, ax = plt.subplots(figsize=(8, 6))
        importances = pd.Series(model.feature_importances_, index=features.columns)
        importances.sort_values().plot(kind='barh', ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)
    else:
        st.write("Feature importance not available for this model.")

# --- Streamlit App ---
def main():
    st.title("Flight Fare Prediction App2")

    # File Upload
    file_path = st.file_uploader("Upload Flight Data (Excel file)", type=["xlsx"])

    if file_path is not None:
        data = load_data(file_path)

        if data is not None:
            st.subheader("Data Preview")
            st.dataframe(data.head())

            # Preprocess Data
            data = preprocess_data(data.copy())  # Use a copy to avoid modifying the original

            # Model Selection
            model_name = st.selectbox("Select a Model", [
                "Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "Gradient Boosting"
            ])

            if st.button("Train and Predict"):
                with st.spinner(f"Training {model_name}..."):
                    r2, mae, mse, rmse, mape, model = train_and_predict(data, model_name)

                st.subheader("Model Performance")
                st.write(f"R² Score: {r2:.4f}")
                st.write(f"Mean Absolute Error: {mae:.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Root Mean Squared Error: {rmse:.2f}")
                st.write(f"Mean Absolute Percentage Error: {mape:.2f}%")

                # Visualizations
                st.subheader("Visualizations")
                plot_price_distribution(data)
                plot_feature_importance(model, data.drop(['Price'], axis=1))

if __name__ == "__main__":
    main()
