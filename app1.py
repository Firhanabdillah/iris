import streamlit as st
import pandas as pd
import pickle
from PIL import Image
model = pickle.load(open('IRIS-model.pkl', 'rb'))

st.header("Pengelompokan Jenis Bunga Iris:")
image = Image.open('bunga1.jpg')
st.image(image, use_column_width=True, caption='jpg')
st.write("Silakan masukkan nilai, untuk mendapatkan prediksi Jenis Bunga Iris")

SepalLengthCm = st.slider('Panjang Sepal Cm:', 2.0, 6.0)
SepalWidthCm = st.slider('Lebar Sepal Cm', 0.0, 5.0)
PetalLengthCm = st.slider('Panjang kelopak Cm',0.0, 3.0)
PetalWidthCm = st.slider('Lebar Kelopak Cm:', 0.0, 2.0)
data = {'Panjang Sepal Cm:': SepalLengthCm,
        'Lebar Sepal Cm': SepalWidthCm,
        'Panjang kelopak Cm': PetalLengthCm,
        'Lebar Kelopak Cm': PetalWidthCm}

features = pd.DataFrame(data, index=[0])

pred_proba = model.predict_proba(features)
#or
prediction = model.predict(features)

st.subheader('Prediction Percentages:') 
st.write('**Kemungkinan Kelompok Bunga Iris menjadi Iris-setosa adalah ( in % )**:',pred_proba[0][0]*100)
st.write('**Kemungkinan Kelompok Bunga Iris menjadi Iris-versicolor adalah ( in % )**:',pred_proba[0][1]*100)
st.write('**Kemungkinan Kelompok Bunga Iris menjadi Iris Iris-virginica ( in % )**:',pred_proba[0][2]*100)
