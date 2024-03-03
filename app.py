import streamlit as st
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import OneHotEncoder
import json


# Load the trained model
model_file = 'banglore_home_prices_model.pickle'

model = pickle.load(open(model_file, 'rb'))


# Load data columns from columns.json
with open('columns.json', 'r') as f:
    columns = json.load(f)
    locations = [col for col in columns['data_columns'] if col != 'total_sqft' and col != 'bath' and col != 'bhk']
  

def encode_location(location, locations):
    encoder = OneHotEncoder()
    encoded_location = encoder.fit_transform([[loc] for loc in locations]).toarray()
    return encoded_location[locations.index(location)]

# Function to predict price
def predict_price(area, bhk, bath, location):
    encoded_location = encode_location(location, locations)
    features = [area, bhk, bath] + list(encoded_location)
    prediction = model.predict([features])
    return prediction[0]

# Title of the app
st.title('Bangalore Real Estate Price Prediction')

# Sidebar
#st.sidebar.header('Input Features')

# Input fields for area, bhk, bath, and location
area = st.number_input('Area (in sqft)')
bhk = st.number_input('BHK')
bath = st.number_input('Bathrooms')
location = st.selectbox('Location', locations)

# Button to predict price
if st.button('Predict Price'):
    price = predict_price(area, bhk, bath, location)
    st.success(f'The predicted price is Rs. {price:.2f} lakhs')