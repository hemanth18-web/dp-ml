import streamlit as st
st.title('Predict Fare of Ailines Tickets')
st.write("please enter the sourcce and destination for your travel")
user_input = st.text_input("please enter your source:")
user_destiantion = st.text_input("please enter your destination:")
st.write("You entered:", user_input)
st.link_button("Check the ticket fare", "https://streamlit.io/gallery")
