# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:30:21 2023

@author: admin
"""

# pip install xgboost

import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from PIL import Image

st.set_page_config(layout="centered", page_title="Customer Churn Prediction")

# Load the model
pickle_in = open("rfcintel1.pkl", "rb")
rfcintel1 = pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def Churn_data(account_length, voice_mail_plan, voice_mail_messages, day_mins,
               evening_mins, night_mins, international_mins, customer_service_calls,
               international_plan, day_calls, day_charge, evening_calls, evening_charge, night_calls,
               night_charge, international_calls, international_charge, total_charge):
    
    voice_mail_plan = 1 if voice_mail_plan == "Yes" else 0
    international_plan = 1 if international_plan == "Yes" else 0
    
    prediction = rfcintel1.predict([[account_length, voice_mail_plan, voice_mail_messages, day_mins, evening_mins, night_mins,
                                     international_mins, customer_service_calls, international_plan, day_calls,
                                     day_charge, evening_calls, evening_charge, night_calls, night_charge,
                                     international_calls, international_charge, total_charge]])
    return prediction

def main():
    st.markdown("""
    <div style="background-color:#17A589;padding:10px;border-radius:10px">
    <h2 style="color:white;text-align:center;">Customer Churn Prediction</h2>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Customer Churn Prediction")
    st.sidebar.write("Please fill in the parameters below to predict customer churn.")
    
    account_length = st.sidebar.number_input("Account Length", min_value=0, max_value=100, value=0)
    voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", ["No", "Yes"])
    voice_mail_messages = st.sidebar.number_input("Voice Mail Messages", min_value=0, max_value=100, value=0)
    day_mins = st.sidebar.number_input("Day Minutes", min_value=0.0, max_value=500.0, value=0.0)
    evening_mins = st.sidebar.number_input("Evening Minutes", min_value=0.0, max_value=500.0, value=0.0)
    night_mins = st.sidebar.number_input("Night Minutes", min_value=0.0, max_value=500.0, value=0.0)
    international_mins = st.sidebar.number_input("International Minutes", min_value=0.0, max_value=100.0, value=0.0)
    customer_service_calls = st.sidebar.number_input("Customer Service Calls", min_value=0, max_value=10, value=0)
    international_plan = st.sidebar.selectbox("International Plan", ["No", "Yes"])
    day_calls = st.sidebar.number_input("Day Calls", min_value=0, max_value=200, value=0)
    day_charge = st.sidebar.number_input("Day Charge", min_value=0.0, max_value=100.0, value=0.0)
    evening_calls = st.sidebar.number_input("Evening Calls", min_value=0, max_value=200, value=0)
    evening_charge = st.sidebar.number_input("Evening Charge", min_value=0.0, max_value=100.0, value=0.0)
    night_calls = st.sidebar.number_input("Night Calls", min_value=0, max_value=200, value=0)
    night_charge = st.sidebar.number_input("Night Charge", min_value=0.0, max_value=100.0, value=0.0)
    international_calls = st.sidebar.number_input("International Calls", min_value=0, max_value=50, value=0)
    international_charge = st.sidebar.number_input("International Charge", min_value=0.0, max_value=50.0, value=0.0)
    total_charge = st.sidebar.number_input("Total Charge", min_value=0.0, max_value=500.0, value=0.0)
    
    if st.sidebar.button("Predict Churn"):
        result = Churn_data(account_length, voice_mail_plan, voice_mail_messages, day_mins,
                            evening_mins, night_mins, international_mins, customer_service_calls,
                            international_plan, day_calls, day_charge, evening_calls, evening_charge, night_calls,
                            night_charge, international_calls, international_charge, total_charge)
        if result[0] == 0:
            st.success("The customer is not likely to churn.")
        else:
            st.warning("The customer is likely to churn.")

if __name__ == '__main__':
    main()
