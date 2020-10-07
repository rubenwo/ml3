import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

import cv2
from PIL import Image, ImageOps
import numpy as np

st.write("""
             # Favorite Object detection CNN
             """
             )
st.write("This is a simple image classification web app to predict ")
st.set_option('deprecation.showfileUploaderEncoding', False)
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

model = load_model('./models/cnn.h5')

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)) / 255.

    img_reshape = img_resize[np.newaxis, ...]

    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    st.write(prediction)

