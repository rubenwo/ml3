import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.write("""
             # Favorite Object detection CNN
             """
         )
st.write("This is a simple web app to classify images in 8 categories")
st.set_option('deprecation.showfileUploaderEncoding', False)
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

model = load_model('./models/best_model.h5')

if file is None:
    st.text("Please upload an image file")
else:
    img = Image.open(file)
    st.image(img, use_column_width=True)

    img.save("./data/temp.png")

    img = image.load_img('./data/temp.png', grayscale=False, target_size=(256, 256),
                         color_mode='rgb', interpolation='bilinear')

    img_array = image.img_to_array(img)
    img_array = np.array([img_array])

    prediction = model.predict(img_array)

    # st.text("Probability (0: Airplane, 1: Car, 2: Cat, 3: Dog, 4: Flower, 5: Fruit, 6: Motorbike, 7: Person")
    df = pd.DataFrame(prediction,
                      columns=['Airplane', 'Car', 'Cat', 'Dog', 'Flower', 'Fruit', 'Motorbike', 'Person'])
    st.dataframe(df)
