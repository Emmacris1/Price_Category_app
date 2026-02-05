import streamlit  as st
import pandas as pd
import numpy as np
import joblib

st.title('Price Category Prediction App')
st.write('This Web Application is Integrated With The Logistic Regression Built By Ima')
size=st.number_input('Insert size')
bedrooms=st.slider('Input Numbers of Bedrooms',0,10,1)

model=joblib.load('Log_model.pkl')

input_data = np.array([size,bedrooms]).reshape(1,-1)

if st.button('Predict'):
    result=model.predict(input_data)
    st.write(f' The Price Category is Actually {result[0]}')
