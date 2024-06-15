# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 19:42:41 2023
@author: 20050
"""

import pickle
import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

def load_model():
    with open('./rfcintel1.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the XGBoost model
xgb_model = load_model()

st.title('Customer Churn Prediction')
st.write("""
This application predicts whether a customer will churn (leave the service) based on their usage data and account information.
""")

def churn_prediction(account_length, voice_mail_plan, voice_mail_messages, day_mins,
                     evening_mins, night_mins, international_mins, customer_service_calls,
                     international_plan, day_calls, day_charge, evening_calls, evening_charge,
                     night_calls, night_charge, international_calls, international_charge, total_charge):
    # Encode categorical variables
    voice_mail_plan = 1 if voice_mail_plan == "Yes" else 0
    international_plan = 1 if international_plan == "Yes" else 0
    
    # Prepare input data for prediction
    input_data = np.array([[account_length, voice_mail_plan, voice_mail_messages, day_mins,
                            evening_mins, night_mins, international_mins, customer_service_calls,
                            international_plan, day_calls, day_charge, evening_calls, evening_charge,
                            night_calls, night_charge, international_calls, international_charge, total_charge]])
    
    # Make prediction
    prediction = xgb_model.predict(input_data)
    
    return "Customer will churn" if prediction[0] == 1 else "Customer will not churn"

def main():
    st.sidebar.header("Input Parameters")
    
    # Organizing inputs in columns for better layout
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        account_length = st.number_input("Account Length", min_value=0, max_value=200, value=100, help="Number of months the customer has been with the company")
        voice_mail_plan = st.selectbox("Voice mail plan", ["No", "Yes"])
        voice_mail_messages = st.number_input("Voice mail messages", min_value=0, max_value=50, value=0, help="Number of voice mail messages")
        day_mins = st.number_input("Day mins", min_value=0.0, max_value=400.0, value=200.0, help="Total minutes of day calls")
        evening_mins = st.number_input("Evening mins", min_value=0.0, max_value=400.0, value=200.0, help="Total minutes of evening calls")
        night_mins = st.number_input("Night mins", min_value=0.0, max_value=400.0, value=200.0, help="Total minutes of night calls")
        international_mins = st.number_input("International mins", min_value=0.0, max_value=50.0, value=10.0, help="Total minutes of international calls")
    
    with col2:
        customer_service_calls = st.number_input("CS calls", min_value=0, max_value=20, value=1, help="Number of calls to customer service")
        international_plan = st.selectbox("International plan", ["No", "Yes"])
        day_calls = st.number_input("Day calls", min_value=0, max_value=200, value=100, help="Number of day calls")
        day_charge = st.number_input("Day charge", min_value=0.0, max_value=100.0, value=20.0, help="Total day call charge")
        evening_calls = st.number_input("Evening calls", min_value=0, max_value=200, value=100, help="Number of evening calls")
        evening_charge = st.number_input("Evening charge", min_value=0.0, max_value=100.0, value=20.0, help="Total evening call charge")
        night_calls = st.number_input("Night calls", min_value=0, max_value=200, value=100, help="Number of night calls")
        night_charge = st.number_input("Night charge", min_value=0.0, max_value=100.0, value=20.0, help="Total night call charge")
        international_calls = st.number_input("International calls", min_value=0, max_value=50, value=5, help="Number of international calls")
        international_charge = st.number_input("International charge", min_value=0.0, max_value=20.0, value=2.5, help="Total international call charge")
        total_charge = st.number_input("Total charge", min_value=0.0, max_value=500.0, value=100.0, help="Total charge for all calls")

    if st.button('Predict Churn'):
        result = churn_prediction(account_length, voice_mail_plan, voice_mail_messages, day_mins,
                                  evening_mins, night_mins, international_mins, customer_service_calls,
                                  international_plan, day_calls, day_charge, evening_calls, evening_charge,
                                  night_calls, night_charge, international_calls, international_charge, total_charge)
        st.success(f'Prediction: {result}')

if __name__ == '__main__': 
    main()
