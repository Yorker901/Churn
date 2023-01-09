# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:30:21 2023

@author: admin
"""

#pip install xgboost

import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble  import RandomForestClassifier
import streamlit as st 

from PIL import Image

st.set_page_config(layout="centered")

pickle_in = open("rfcintel1.pkl","rb")
rfcintel1 = pickle.load(pickle_in)

def welcome():
    return "Welcome All"


def Churn_data(account_length,voice_mail_plan,voice_mail_messages,day_mins,
                   evening_mins,night_mins,international_mins,customer_service_calls,
                   international_plan,day_calls,day_charge,evening_calls,evening_charge,night_calls,
                   night_charge,international_calls,international_charge,total_charge):
    
    if voice_mail_plan == "No":
        voice_mail_plan = 0
    else:
        voice_mail_plan = 1
    if international_plan == "No":
        international_plan = 0
    else:
        international_plan = 1 
   
    prediction = rfcintel1.predict([[account_length,voice_mail_plan,voice_mail_messages,day_mins,evening_mins,night_mins,
                    international_mins,customer_service_calls,international_plan,day_calls,
                    day_charge,evening_calls,evening_charge,night_calls,night_charge,
                    international_calls,international_charge,total_charge]])
    print(prediction)
    return prediction

def main():
    
    html_temp = """
    <div style="background-color:#17A589;padding:10px">
    <h2 style="color:white;text-align:center;">Customer Churn Prediction_ </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    st.sidebar.header("Predicting Customer Churn")
    st.sidebar.text("Choose the Below Parameters to Predict")
    account_length=st.sidebar.number_input("Enter the Account Length")
    voice_mail_plan=st.sidebar.selectbox("Do you have a Voice mail plan",("No","Yes"))
    voice_mail_messages=st.sidebar.number_input("Enter the number of voice mail messages")
    day_mins=st.sidebar.number_input("Enter the total day mins")
    evening_mins=st.sidebar.number_input("Enter the total evening mins")
    night_mins=st.sidebar.number_input("Enter the total night mins")
    international_mins=st.sidebar.number_input("Enter the total international mins")
    customer_service_calls=st.sidebar.number_input("Enter the number of CS calls")
    international_plan=st.sidebar.selectbox("Do you have a international plan", ("No","Yes"))
    day_calls=st.sidebar.number_input("Enter the number of day calls")
    day_charge=st.sidebar.number_input("Enter the total day charge")
    evening_calls=st.sidebar.number_input("Enter the number of Evening calls")
    evening_charge=st.sidebar.number_input("Enter the total evening charge")
    night_calls=st.sidebar.number_input("Enter the number of Night calls")
    night_charge=st.sidebar.number_input("Enter the total night charge")
    international_calls=st.sidebar.number_input("Enter the number of INTL calls")
    international_charge=st.sidebar.number_input("Enter the total INTL charge")
    total_charge=st.sidebar.number_input("Enter the total charge")
    
    result=""
    if st.button("Click Here To Predict Churn"):
        result=Churn_data(account_length,voice_mail_plan,voice_mail_messages,day_mins,
                           evening_mins,night_mins,international_mins,customer_service_calls,
                           international_plan,day_calls,day_charge,evening_calls,evening_charge,night_calls,
                           night_charge,international_calls,international_charge,total_charge)
        st.header('We can predict that our {}% '.format(result))
        if result[0]==0:
            st.error("Customer will not churn")
        else:
            st.success("Customer will churn")
        #st.success('Yes,Customer will churn' if format(result) == [1] else 'No, Customer will not churn')


if __name__=='__main__':
    main()