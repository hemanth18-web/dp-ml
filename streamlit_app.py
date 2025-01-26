import streamlit as st
st.title('Predict Fare of Ailines Tickets')
st.write("please enter the sourcce and destination for your travel")
user_input = st.text_input("please enter your source:")
user_destiantion = st.text_input("please enter your destination:")
st.write("You entered:", user_input)
st.link_button("Check the ticket fare", "https://streamlit.io/gallery")
# Take input from the user
user_input = st.text_input("Please enter your source:")
user_destination = st.text_input("Please enter your destination:")

# Add a submit button to trigger the display of the results
if st.button("Submit"):
    # Display the entered source and destination
    st.write("Source:", user_input)
    st.write("Destination:", user_destination)
    
    # Add a link button to check ticket fare
    st.markdown("[Check the ticket fare](https://streamlit.io/gallery)")
