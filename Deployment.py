import pickle
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# Load the trained model
pickle_in = open('./rfcintel1.pkl', 'rb')
xgb = pickle.load(pickle_in)

def churn_prediction(model, account_length, voice_mail_plan, voice_mail_messages, day_mins, evening_mins, night_mins,
                     international_mins, customer_service_calls, international_plan, day_calls, day_charge,
                     evening_calls, evening_charge, night_calls, night_charge, international_calls,
                     international_charge, total_charge):
    # Prepare input data as a DataFrame
    input_data = {
        'Account Length': [account_length],
        'Voice Mail Plan': [1 if voice_mail_plan == "Yes" else 0],
        'Voice Mail Messages': [voice_mail_messages],
        'Total Day Mins': [day_mins],
        'Total Evening Mins': [evening_mins],
        'Total Night Mins': [night_mins],
        'Total International Mins': [international_mins],
        'Customer Service Calls': [customer_service_calls],
        'International Plan': [1 if international_plan == "Yes" else 0],
        'Total Day Calls': [day_calls],
        'Total Day Charge': [day_charge],
        'Total Evening Calls': [evening_calls],
        'Total Evening Charge': [evening_charge],
        'Total Night Calls': [night_calls],
        'Total Night Charge': [night_charge],
        'Total International Calls': [international_calls],
        'Total International Charge': [international_charge],
        'Total Charge': [total_charge]
    }

    input_df = pd.DataFrame(input_data)

    # Perform prediction
    prediction = model.predict(input_df)

    if prediction[0] == 0:
        return "Customer will not churn"
    else:
        return "Customer will churn"

def main():
    st.title("Churn Prediction")
    st.sidebar.header('User Input Parameters')

    account_length = st.sidebar.number_input("Enter the Account Length")
    voice_mail_plan = st.sidebar.selectbox("Do you have a Voice mail plan", ("No", "Yes"))
    voice_mail_messages = st.sidebar.number_input("Enter the number of voice mail messages")
    day_mins = st.sidebar.number_input("Enter the total day mins")
    evening_mins = st.sidebar.number_input("Enter the total evening mins")
    night_mins = st.sidebar.number_input("Enter the total night mins")
    international_mins = st.sidebar.number_input("Enter the total international mins")
    customer_service_calls = st.sidebar.number_input("Enter the number of CS calls")
    international_plan = st.sidebar.selectbox("Do you have an international plan", ("No", "Yes"))
    day_calls = st.sidebar.number_input("Enter the number of day calls")
    day_charge = st.sidebar.number_input("Enter the total day charge")
    evening_calls = st.sidebar.number_input("Enter the number of Evening calls")
    evening_charge = st.sidebar.number_input("Enter the total evening charge")
    night_calls = st.sidebar.number_input("Enter the number of Night calls")
    night_charge = st.sidebar.number_input("Enter the total night charge")
    international_calls = st.sidebar.number_input("Enter the number of INTL calls")
    international_charge = st.sidebar.number_input("Enter the total INTL charge")
    total_charge = st.sidebar.number_input("Enter the total charge")

    churn = ''
    if st.button('Predict Churn'):
        churn = churn_prediction(xgb, account_length, voice_mail_plan, voice_mail_messages, day_mins,
                                 evening_mins, night_mins, international_mins, customer_service_calls,
                                 international_plan, day_calls, day_charge, evening_calls, evening_charge,
                                 night_calls, night_charge, international_calls, international_charge,
                                 total_charge)
        st.success('Prediction: {}'.format(churn))

if __name__ == '__main__':
    main()
