import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('vehicle_price_model.pkl')

st.title('Vehicle Price Prediction')

year = st.number_input('Year', min_value=1980, max_value=2025, value=2018)
mileage = st.number_input('Mileage', min_value=0, max_value=300000, value=45000)
cylinders = st.number_input('Cylinders', min_value=1, max_value=16, value=4)

if st.button('Predict Price'):
    X = np.array([[year, mileage, cylinders]])
    predicted_price = model.predict(X)[0]
    st.success(f'Predicted Vehicle Price: ${predicted_price:,.2f}')
