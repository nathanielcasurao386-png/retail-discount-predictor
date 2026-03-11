import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("random_forest.pkl")

st.title("Retail Discount Efficiency Predictor")
st.write("Predict how discounts affect sales.")

# User Inputs
category = st.selectbox(
    "Category",
    ["Patisserie", "Milk Products", "Butchers", "Beverages", "Food"]
)

location = st.selectbox(
    "Location",
    ["Online", "In-store"]
)

price = st.slider("Price Per Unit", 1.0, 500.0)

month = st.selectbox(
    "Transaction Month",
    ["January","February","March","April","May","June",
     "July","August","September","October","November","December"]
)

discount = st.selectbox("Discount Applied", [True, False])

# Convert user input into dataframe
input_data = pd.DataFrame({
    "Category": [category],
    "Location": [location],
    "Price Per Unit": [price],
    "Transaction Month": [month],
    "Discount Applied": [discount]
})

# Convert categorical text to numbers (same logic used during training)
input_data["Category"] = input_data["Category"].astype("category").cat.codes
input_data["Location"] = input_data["Location"].astype("category").cat.codes
input_data["Transaction Month"] = input_data["Transaction Month"].astype("category").cat.codes
input_data["Discount Applied"] = input_data["Discount Applied"].astype(int)

# Prediction
if st.button("Predict Quantity Sold"):

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Quantity Sold: {round(prediction,2)} units")

    revenue = prediction * price

    st.write(f"Estimated Revenue: ₱{round(revenue,2)}")