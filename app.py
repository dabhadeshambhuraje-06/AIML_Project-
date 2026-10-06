import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('model/random_forest_car_price_model.pkl')
feature_columns = joblib.load("model/feature_columns.pkl")

st.title("Car Price Prediction")
st.write("Enter the details of the car to predict its selling price.")

car_age = st.number_input("Car Age (in years)", min_value=0, max_value=30, value=5)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])

if st.button("Predict Selling Price"):
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'Car_Age': [car_age],
        'km_driven': [km_driven],
        'fuel': [fuel_type],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner]
    })
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=feature_columns, fill_value=False)  # Ensure the input data has the same columns as the training data
    prediction = model.predict(input_data)

    predicted_price = prediction[0]

    st.success(f"Predicted Selling Price: ₹{predicted_price:.2f} lakh")

    
    