import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# Load the trained model
model = load_model('ann_model.h5')

# Load the scaler and encoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)


# Streamlit app
st.title('Customer Churn Prediction')

# user input
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Balance': [balance],
    'EstimatedSalary': [estimated_salary],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

geography_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

# Combine numeric and one-hot geography data
input_df = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Reorder columns to match training data
input_df = input_df[scaler.feature_names_in_]

# Scale
df = scaler.transform(input_df)

# Prediction
prediction = model.predict(df)
probability = prediction[0][0]

st.write(f'Churn Probability: {probability:.2f}')

# Display the result
if probability > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is likely to stay.')