import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
st.write("""
          # Malaria Cell Classification
          """
          )
upload_file = st.sidebar.file_uploader("Upload Cell Images", type="png")
Generate_pred=st.sidebar.button("Predict")
model=tf.keras.models.load_model('malaria_cell.h5')
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
if Generate_pred:
    image=Image.open(upload_file)
    with st.expander('Cell Image', expanded = True):
        st.image(image, use_column_width=True)
    pred=import_n_pred(image, model)
    labels = ['Parasitized', 'Uninfected']
    st.title("Prediction of image is {}".format(pred))