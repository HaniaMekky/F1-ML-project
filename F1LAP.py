import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load(open("best_random_forest_model.pkl", 'rb'))

# Prediction function
def predict(features):
    prediction = model.predict(features)
    return prediction

# Page configuration
st.set_page_config(
    page_title='F1 Final Lap Prediction',
    page_icon='🏎️',
    initial_sidebar_state='collapsed'
)

# Home page
st.write('# F1 Final Lap Prediction')
st.subheader('Enter race statistics for final lap prediction')

# User inputs
SpeedI1 = st.number_input("Enter Speed of section 1:", min_value=0.0)
SpeedI2 = st.number_input("Enter Speed of section 2:", min_value=0.0)
Compound = st.selectbox("Enter the tyre type here (1: Medium, 2: Hard, 3: Soft):", options=[1, 2, 3])
TyreLife = st.number_input("Enter the tyre life here:", min_value=0)
FreshTyre = st.selectbox("Are the tyres fresh?", ("TRUE", "FALSE"))

# Convert FreshTyre input to numeric
FreshTyre = 1 if FreshTyre == "TRUE" else 0

# Concatenate features
features = np.array([SpeedI1, SpeedI2, Compound, TyreLife, FreshTyre]).reshape(1, -1)

# Prediction
if st.button("Predict"):
    result = predict(features)
    st.write(f'The predicted SpeedFL is: *{result[0]}*')

# About page
st.write("# About")
st.write("This Application predicts the final lap speed in F1 based on race statistics.")

# Contact page
st.write("# Contact")
st.write("For inquiries, please contact us at contact@example.com")

# Read the data from CSV
data = pd.read_csv('df.csv')
