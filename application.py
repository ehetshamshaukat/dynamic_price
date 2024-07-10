from src.pipeline.prediction_pipeline import Features, Prediction
import streamlit as st
import numpy as np

st.header("Dynamic price prediction")
location_category_list = ['Urban', 'Suburban', 'Rural']
loyalty_status_list = ['Regular', 'Silver', 'Gold']
time_of_booking_list = ['Morning', 'Afternoon', 'Evening', 'Night']
vehicle_list = ['Economy', 'Premium']

number_of_riders = st.number_input("Please enter the number of raiders", value=0)
number_of_drivers = st.number_input("please enter the number of drivers", value=0)
location_category = st.selectbox("Location", location_category_list)
customer_loyalty_status = st.selectbox("Loyalty Status", loyalty_status_list)
number_of_past_rides = st.number_input("number of past ride", value=0)
average_ratings = st.number_input("enter the average rating", max_value=5)
time_of_booking = st.selectbox("please enter time of booking", time_of_booking_list)
vehicle_type = st.selectbox("Please enter the vehicle type", vehicle_list)
expected_ride_duration = st.number_input("expected ride duration",value=0)

ok = st.button("Predict")

if ok:
    f = Features(number_of_riders, number_of_drivers, location_category, customer_loyalty_status, number_of_past_rides,
                 average_ratings, time_of_booking, vehicle_type, expected_ride_duration)
    feature = f.to_dataframe()
    p = Prediction()
    output = p.initiate_prediction(feature)
    st.subheader(round(float(output), 2))
