import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image
st.write("""
          # Malaria Cell Classification
          """
          )
st.sidebar.header('User Input Parameters')
option=st.sidebar.radio("Select the model", ('Convolutional Neural Network', 'VGG-19'))
upload_file = st.sidebar.file_uploader("Upload Cell Images", type="png")
if upload_file is not None:
    st.sidebar.write("File Uploaded Successfully")
Generate_pred=st.sidebar.button("Predict")
model1=tf.keras.models.load_model('cnn1.h5')
model2=tf.keras.models.load_model('vgg19.h5')

def import_n_pred(image_data, model):
    size = (128,128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    print(pred)
    if pred[0][0]<0.5:
        return "Parasitized"
    else:
        return "Uninfected"
    
def import_n_pred1(image_data, model2):
    size = (128,128)
    img = image_data.resize(size)
    img = np.asarray(img)
    img = img/255.0
    img = np.expand_dims(img, axis=0)
    pred = model2.predict(img)
    print(pred[0][0])
    if pred[0][0]>0.5:
        return "Parasitized"
    else:
        return "Uninfected"

if Generate_pred:
    if option=='Convolutional Neural Network':
        if upload_file is not None:
            image=Image.open(upload_file)
            with st.expander('Cell Image', expanded = True):
                st.image(image, use_column_width=True)
            pred=import_n_pred(image, model1)
            labels = ['Parasitized', 'Uninfected']
            st.title("Prediction of image is {}".format(pred))
    
    if option=='VGG-19':
        if upload_file is not None:
            image=Image.open(upload_file)
            with st.expander('Cell Image', expanded = True):
                st.image(image, use_column_width=True)
            pred=import_n_pred1(image, model2)
            labels = ['Parasitized', 'Uninfected']
            st.title("Prediction of image is {}".format(pred))

    else:
        st.write("Please upload cell image")