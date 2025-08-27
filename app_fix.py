import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
credit_model = joblib.load("C:/Users/HP/OneDrive/Desktop/credit default projext fix/credit_model.pkl")

# Title
st.title("Credit Default Risk Prediction System")
st.write("Fill in the customer's details to predict the chance of loan default.")

# Dropdown options
industry_options = ['Construction', 'Consumer Goods', 'Wholesale and retail trade', 
    'Human health and social work activities', 'Agriculture',
    'Transportation and storage', 'Information and communication',
    'Services', 'Finance',
    'Administrative and support service activities', 'Education',
    'Oil and Gas', 'General', 'Manufacturing',
    'Real estate activities',
    'Professional, scientific and technical activities',
    'Industrial Goods', 'Arts, entertainment and recreation',
    'Power and Energy', 'Non-Governmental Organization', 'Government',
    'Utilities', 'Online', 'Natural Resources', 
    'Waste Management Activities', 'Mining and Quarrying', 'Other',
    'Extraterritorial organizations', 'Hardware', 'Household Goods',
    'Dairy Produce', 'Food and Groceries', 'Institution Staff',
    'Organic Farming']

state_options = ['Lagos', 'FCT', 'Edo', 'Ogun', 'Rivers', 'Oyo', 'Akwa Ibom',
    'Kano', 'Kogi', 'Kwara', 'Cross River', 'Ondo', 'Delta', 'Ebonyi',
    'Osun', 'Nasarawa', 'Ekiti', 'Kaduna', 'Yobe', 'Plateau', 'Abia',
    'Niger', 'Zamfara', 'Imo', 'Sokoto', 'Enugu', 'Anambra',
    'Adamawa', 'Jigawa', 'Gombe', 'Bayelsa', 'Benue', 'Borno', 'Kebbi',
    'Katsina', 'Taraba', 'Bauchi']

# Numeric inputs
age = st.text_input("Customer Age")
income = st.text_input("Yearly Income (₦)")
loan_amount = st.text_input("Requested Loan Amount (₦)")
time_employed = st.text_input("Number of Months Working Current Job")
time_at_address = st.text_input("Months Living at Current Address")
num_children = st.text_input("Number of Children")

# Dropdowns
industry = st.selectbox("Industry of Employment", options=industry_options)
employer_state = st.selectbox("State Where Employer is Located", options=state_options)

# Predict button
if st.button("Predict Default Probability"):
    try:
        # Convert numeric fields
        age = int(age)
        income = float(income)
        loan_amount = float(loan_amount)
        time_employed = float(time_employed)
        time_at_address = float(time_at_address)
        num_children = int(num_children)

        # Eligibility check
        if age < 18:
            st.error("Customer is ineligible: Age must be 18 or above.")
        else:
            # Step 1: Create original input DataFrame
            input_data = pd.DataFrame([{
                "Age": age,
                "Income": income,
                "LoanAmount": loan_amount,
                "TimeEmploymentMM": time_employed,
                "TotalTimeAtAddress": time_at_address,
                "NoChildren": num_children,
                "Industry": industry,
                "EmployerState": employer_state
            }])

            # Step 2: Feature Engineering
            input_data["LoanPerIncome"] = loan_amount / (income + 1e-5)
            input_data["LoanToEmployment"] = loan_amount / (time_employed + 1e-5)
            input_data["Log_Income"] = np.log1p(income)
            input_data["Log_TimeEmploymentMM"] = np.log1p(time_employed)
            input_data["Log_TotalTimeAtAddress"] = np.log1p(time_at_address)

            # Step 3: Retain only top 10 features used during training
            top_10_features = [
                'LoanPerIncome', 'LoanToEmployment', 'Log_TimeEmploymentMM', 'Log_Income',
                'Log_TotalTimeAtAddress', 'LoanAmount', 'Age', 'Industry',
                'NoChildren', 'EmployerState'
            ]

            model_input = input_data[top_10_features]

            # Step 4: Make prediction
            default_prob = credit_model.predict_proba(model_input)[0][1]
            not_default_prob = 1 - default_prob

            # Step 5: Show result
            st.success(f"Customer has a {default_prob * 100:.2f}% chance of defaulting.")
            st.info(f"Probability of not defaulting: {not_default_prob * 100:.2f}%")

    except ValueError:
        st.error("Please enter valid numeric values for all fields.")

