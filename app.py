import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("retail_store_sales.csv")

# Prepare features
X = df[['Category','Location','Price Per Unit','Transaction Month','Discount Applied']]
y = df['Quantity']

# Convert categorical values to numbers
X["Category"] = X["Category"].astype("category").cat.codes
X["Location"] = X["Location"].astype("category").cat.codes
X["Transaction Month"] = X["Transaction Month"].astype("category").cat.codes
X["Discount Applied"] = X["Discount Applied"].astype(int)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X,y)

st.title("Retail Discount Efficiency Predictor")

st.write("Predict how discounts affect sales.")

category = st.selectbox("Category",["Patisserie","Milk Products","Butchers","Beverages","Food"])
location = st.selectbox("Location",["Online","In-store"])
price = st.slider("Price Per Unit",1.0,500.0)
month = st.selectbox("Transaction Month",
["January","February","March","April","May","June","July","August","September","October","November","December"])
discount = st.selectbox("Discount Applied",[True,False])

input_data = pd.DataFrame({
    "Category":[category],
    "Location":[location],
    "Price Per Unit":[price],
    "Transaction Month":[month],
    "Discount Applied":[discount]
})

input_data["Category"] = input_data["Category"].astype("category").cat.codes
input_data["Location"] = input_data["Location"].astype("category").cat.codes
input_data["Transaction Month"] = input_data["Transaction Month"].astype("category").cat.codes
input_data["Discount Applied"] = input_data["Discount Applied"].astype(int)

if st.button("Predict Quantity Sold"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Quantity Sold: {round(prediction,2)} units")

    revenue = prediction * price
    st.write(f"Estimated Revenue: ₱{round(revenue,2)}")
