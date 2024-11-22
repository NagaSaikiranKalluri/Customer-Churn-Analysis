import streamlit as st
import numpy as np
import joblib

st.title("Churn Prediction")

st.divider()

st.write("Please enter the values for prediction and hit the predict button")

st.divider()

# Inputs from the user
age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Enter gender", ["Male", "Female"])
tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)
monthly_charge = st.number_input("Enter Monthly charges:", min_value=30, max_value=150)

st.divider()

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# Predict button
predict_button = st.button("Predict!")

if predict_button:
    # Encoding gender: 1 -> Female, 0 -> Male
    gender_selected = 1 if gender == "Female" else 0

    # Ensure data is in the correct order: ['Age', 'Gender', 'Tenure', 'MonthlyCharges']
    X = [age, gender_selected, tenure, monthly_charge]
    
    # Convert to numpy array and reshape for the scaler
    X_array = np.array(X).reshape(1, -1)  # Reshape to (1, 4) for the model
    
    # Scale the data
    X_scaled = scaler.transform(X_array)
    
    # Predict the churn
    prediction = model.predict(X_scaled)[0]
    
    # Output prediction result
    predicted = 'Churn' if prediction == 1 else 'Not Churn'
    st.write(f"Predicted: {predicted}")
else:
    st.write("Please enter the values and use the predict button")