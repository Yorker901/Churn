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

st.title('Churn Prediction Model')

# Load the XGBoost model
pickle_in = open('./rfcintel1.pkl', 'rb')
xgb = pickle.load(pickle_in)

def churn_prediction(account_length, voice_mail_plan, voice_mail_messages, day_mins,
                     evening_mins, night_mins, international_mins, customer_service_calls,
                     international_plan, day_calls, day_charge, evening_calls, evening_charge,
                     night_calls, night_charge, international_calls, international_charge, total_charge):
    # Encode categorical variables
    if voice_mail_plan == "No":
        voice_mail_plan = 0
    else:
        voice_mail_plan = 1
    
    if international_plan == "No":
        international_plan = 0
    else:
        international_plan = 1  
    
    # Prepare input data for prediction
    input_data = np.array([[account_length, voice_mail_plan, voice_mail_messages, day_mins,
                            evening_mins, night_mins, international_mins, customer_service_calls,
                            international_plan, day_calls, day_charge, evening_calls, evening_charge,
                            night_calls, night_charge, international_calls, international_charge, total_charge]])
    
    # Make prediction
    prediction = xgb.predict(input_data)
    
    if prediction[0] == 0:
        return "Customer will not churn"
    else:
        return "Customer will churn"

def main():
    st.title("Churn Prediction")
    
    # Input fields in the sidebar
    account_length = st.sidebar.number_input("Enter the Account Length")
    voice_mail_plan = st.sidebar.selectbox("Do you have a Voice mail plan", ["No", "Yes"])
    voice_mail_messages = st.sidebar.number_input("Enter the number of voice mail messages")
    day_mins = st.sidebar.number_input("Enter the total day mins")
    evening_mins = st.sidebar.number_input("Enter the total evening mins")
    night_mins = st.sidebar.number_input("Enter the total night mins")
    international_mins = st.sidebar.number_input("Enter the total international mins")
    customer_service_calls = st.sidebar.number_input("Enter the number of CS calls")
    international_plan = st.sidebar.selectbox("Do you have an international plan", ["No", "Yes"])
    day_calls = st.sidebar.number_input("Enter the number of day calls")
    day_charge = st.sidebar.number_input("Enter the total day charge")
    evening_calls = st.sidebar.number_input("Enter the number of Evening calls")
    evening_charge = st.sidebar.number_input("Enter the total evening charge")
    night_calls = st.sidebar.number_input("Enter the number of Night calls")
    night_charge = st.sidebar.number_input("Enter the total night charge")
    international_calls = st.sidebar.number_input("Enter the number of INTL calls")
    international_charge = st.sidebar.number_input("Enter the total INTL charge")
    total_charge = st.sidebar.number_input("Enter the total charge")
    
    # Prediction button and display area
    if st.button('Predict Churn'):
        result = churn_prediction(account_length, voice_mail_plan, voice_mail_messages, day_mins,
                                  evening_mins, night_mins, international_mins, customer_service_calls,
                                  international_plan, day_calls, day_charge, evening_calls, evening_charge,
                                  night_calls, night_charge, international_calls, international_charge, total_charge)
        st.success('Prediction: {}'.format(result))

if __name__=='__main__': 
    main()
