import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib

# #Offline uchun sozlama
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

#deploy sozlamalari uchun
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

#model title
st.title('Transportlarni ajratuvchi  model')

#rasmlarni yuklash uchun
file = st.file_uploader("Rasmni yuklash", type=['png', 'jpg', 'jpeg', 'gif'])

if file:
    #convert image
    img = PILImage.create(file)

    #modelni yuklash va predict
    model = load_learner('transport_model.pkl')
    pred, pred_idx, prob = model.predict(img)

    #ehtimollik ko'rsatkichlari
    st.image(img)
    st.success(f"Tahmin: {pred}")
    st.info(f"Ehtimollik: {prob[pred_idx] * 100:.1f}%")

    #plotting
    fig = px.bar(x = prob * 100, y = model.dls.vocab)
    st.plotly_chart(fig)