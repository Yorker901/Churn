# # -*- coding: utf-8 -*-
# """
# Created on Wed Jan  4 19:42:41 2023
# @author: 20050
# """

# import pickle
# import streamlit as st
# import numpy as np
# import pandas as pd
# from xgboost import XGBClassifier
# import matplotlib.pyplot as plt

# def load_model():
#     with open('./rfcintel1.pkl', 'rb') as file:
#         model = pickle.load(file)
#     return model

# # Load the XGBoost model
# xgb_model = load_model()

# def churn_prediction(account_length, voice_mail_plan, voice_mail_messages, day_mins,
#                      evening_mins, night_mins, international_mins, customer_service_calls,
#                      international_plan, day_calls, day_charge, evening_calls, evening_charge,
#                      night_calls, night_charge, international_calls, international_charge, total_charge):
#     # Encode categorical variables
#     voice_mail_plan = 1 if voice_mail_plan == "Yes" else 0
#     international_plan = 1 if international_plan == "Yes" else 0
    
#     # Prepare input data for prediction
#     input_data = np.array([[account_length, voice_mail_plan, voice_mail_messages, day_mins,
#                             evening_mins, night_mins, international_mins, customer_service_calls,
#                             international_plan, day_calls, day_charge, evening_calls, evening_charge,
#                             night_calls, night_charge, international_calls, international_charge, total_charge]])
    
#     # Make prediction
#     prediction = xgb_model.predict(input_data)
    
#     return "Customer will churn" if prediction[0] == 1 else "Customer will not churn"

# def main():
#     st.set_page_config(page_title="Churn Prediction App", layout="wide")
    
#     # Sidebar navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.selectbox("Go to", ["Home", "About"])

#     if page == "Home":
#         st.title('Customer Churn Prediction')
#         st.write("""
#         This application predicts whether a customer will churn (leave the service) based on their usage data and account information.
#         """)
        
#         st.sidebar.header("Input Parameters")
        
#         # Organizing inputs in columns for better layout
#         col1, col2 = st.sidebar.columns(2)
        
#         with col1:
#             account_length = st.number_input("Account Length", min_value=0, max_value=200, value=100, help="Number of months the customer has been with the company")
#             voice_mail_plan = st.selectbox("Voice mail plan", ["No", "Yes"])
#             voice_mail_messages = st.number_input("Voice mail messages", min_value=0, max_value=50, value=0, help="Number of voice mail messages")
#             day_mins = st.number_input("Day mins", min_value=0.0, max_value=400.0, value=200.0, help="Total minutes of day calls")
#             evening_mins = st.number_input("Evening mins", min_value=0.0, max_value=400.0, value=200.0, help="Total minutes of evening calls")
#             night_mins = st.number_input("Night mins", min_value=0.0, max_value=400.0, value=200.0, help="Total minutes of night calls")
#             international_mins = st.number_input("International mins", min_value=0.0, max_value=50.0, value=10.0, help="Total minutes of international calls")
        
#         with col2:
#             customer_service_calls = st.number_input("CS calls", min_value=0, max_value=20, value=1, help="Number of calls to customer service")
#             international_plan = st.selectbox("International plan", ["No", "Yes"])
#             day_calls = st.number_input("Day calls", min_value=0, max_value=200, value=100, help="Number of day calls")
#             day_charge = st.number_input("Day charge", min_value=0.0, max_value=100.0, value=20.0, help="Total day call charge")
#             evening_calls = st.number_input("Evening calls", min_value=0, max_value=200, value=100, help="Number of evening calls")
#             evening_charge = st.number_input("Evening charge", min_value=0.0, max_value=100.0, value=20.0, help="Total evening call charge")
#             night_calls = st.number_input("Night calls", min_value=0, max_value=200, value=100, help="Number of night calls")
#             night_charge = st.number_input("Night charge", min_value=0.0, max_value=100.0, value=20.0, help="Total night call charge")
#             international_calls = st.number_input("International calls", min_value=0, max_value=50, value=5, help="Number of international calls")
#             international_charge = st.number_input("International charge", min_value=0.0, max_value=20.0, value=2.5, help="Total international call charge")
#             total_charge = st.number_input("Total charge", min_value=0.0, max_value=500.0, value=100.0, help="Total charge for all calls")
        
#         if st.button('Predict Churn'):
#             result = churn_prediction(account_length, voice_mail_plan, voice_mail_messages, day_mins,
#                                       evening_mins, night_mins, international_mins, customer_service_calls,
#                                       international_plan, day_calls, day_charge, evening_calls, evening_charge,
#                                       night_calls, night_charge, international_calls, international_charge, total_charge)
#             st.success(f'Prediction: {result}')

#         if st.checkbox('Show Feature Importance'):
#             st.subheader('Feature Importance')
#             feature_importance = xgb_model.get_booster().get_score(importance_type='weight')
#             keys = list(feature_importance.keys())
#             values = list(feature_importance.values())
#             importance_df = pd.DataFrame(data={'Feature': keys, 'Importance': values})
#             importance_df = importance_df.sort_values(by='Importance', ascending=False)
#             st.bar_chart(importance_df.set_index('Feature'))

#     elif page == "About":
#         st.title("About")
#         st.write("""
#         ### Churn Prediction Model
#         This model is designed to predict customer churn based on various features of their account and usage patterns. 

#         **Features used in the model:**
#         - **Account Length:** The number of months the customer has been with the company.
#         - **Voice Mail Plan:** Whether the customer has a voice mail plan.
#         - **Voice Mail Messages:** The number of voice mail messages.
#         - **Day Minutes/Calls/Charge:** Total minutes, calls, and charge for calls made during the day.
#         - **Evening Minutes/Calls/Charge:** Total minutes, calls, and charge for calls made in the evening.
#         - **Night Minutes/Calls/Charge:** Total minutes, calls, and charge for calls made at night.
#         - **International Minutes/Calls/Charge:** Total minutes, calls, and charge for international calls.
#         - **Customer Service Calls:** The number of calls made to customer service.
#         - **Total Charge:** The total charge for all calls.

#         The model used is an XGBoost classifier, a powerful machine learning algorithm that is well-suited for classification tasks.

#         **How to use the application:**
#         1. Enter the required input parameters in the sidebar.
#         2. Click on 'Predict Churn' to see the prediction result.
#         3. Optionally, check 'Show Feature Importance' to see the importance of each feature in the model.
#         """)

# if __name__ == '__main__': 
#     main()


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
import hashlib

# Function to load the model
def load_model():
    with open('./rfcintel1.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the XGBoost model
xgb_model = load_model()

# Function to authenticate users
def authenticate(username, password):
    # Replace this with your actual authentication logic
    # For demonstration purposes, using hardcoded credentials
    # In real-world scenarios, store hashed passwords securely
    authorized_users = {
        'admin': '5f4dcc3b5aa765d61d8327deb882cf99'  # 'password' hashed with MD5
    }
    
    if username in authorized_users:
        stored_password = authorized_users[username]
        hashed_password = hashlib.md5(password.encode()).hexdigest()
        if hashed_password == stored_password:
            return True
    return False

# Main function
def main():
    st.set_page_config(page_title="Churn Prediction App", layout="wide")
    
    # Authentication
    st.sidebar.title("Authentication")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if authenticate(username, password):
            st.sidebar.success("Logged in as {}".format(username))
            app_content()
        else:
            st.sidebar.error("Authentication failed")
    
# Function to display main app content
def app_content():
    st.title('Customer Churn Prediction')
    st.write("""
    This application predicts whether a customer will churn (leave the service) based on their usage data and account information.
    """)
    
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

    if st.checkbox('Show Feature Importance'):
        st.subheader('Feature Importance')
        feature_importance = xgb_model.get_booster().get_score(importance_type='weight')
        keys = list(feature_importance.keys())
        values = list(feature_importance.values())
        importance_df = pd.DataFrame(data={'Feature': keys, 'Importance': values})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))

# Function to load the model
def load_model():
    with open('./rfcintel1.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function for churn prediction
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

if __name__ == '__main__':
    main()

