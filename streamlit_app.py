# User Input for Prediction
st.write("### Predict Flight Price")

# Predefined options for Source, Destination, and Total Stops
source = st.selectbox("Source", ["Banglore", "Delhi", "Kolkata", "Mumbai", "Chennai"])
destination = st.selectbox("Destination", ["Banglore", "Delhi", "Kolkata", "Mumbai", "Chennai"])
stops = st.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])
airline = st.selectbox("Airline", new_data['Airline'].unique())  # Use unique airline values from the dataset
dep_hour = st.slider("Departure Hour", 0, 23, 10)
dep_minute = st.slider("Departure Minute", 0, 59, 30)
arrival_hour = st.slider("Arrival Hour", 0, 23, 13)
arrival_minute = st.slider("Arrival Minute", 0, 59, 45)
duration_hours = st.number_input("Duration (Hours)", min_value=0, max_value=24, value=3)
duration_minutes = st.number_input("Duration (Minutes)", min_value=0, max_value=59, value=15)
journey_day = st.number_input("Journey Day", min_value=1, max_value=31, value=15)
journey_month = st.number_input("Journey Month", min_value=1, max_value=12, value=3)

# Map stops to numeric values
stop_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
stops_mapped = stop_mapping[stops]

# Predict button
if st.button("Predict Price"):
    # Predict the price
    predicted_price = predict_price(
        source, destination, stops_mapped, airline, dep_hour, dep_minute,
        arrival_hour, arrival_minute, duration_hours, duration_minutes,
        journey_day, journey_month
    )
    st.success(f"The predicted price for the flight is: ₹{predicted_price:.2f}")
