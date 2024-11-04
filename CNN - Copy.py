import gdown
import os
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import joblib
file_id="1wVpr8u8uknSBysrTzQ7yNL009vojC3hi"
destination = "cnn_EuroSAT.h5"

# Check if the model file is already downloaded
if not os.path.exists(destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

# Load the model
cnn = tf.keras.models.load_model(destination)
#cnn = joblib.load("cnn_EuroSAT.h5")
classes = {0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 3: 'Highway', 4: 'Industrial',
           5: 'Pasture', 6: 'PermanentCrop', 7: 'Residential', 8: 'River', 9: 'SeaLake'}
def detect(frame):
    img = cv2.resize(frame, (256, 256))
    if np.max(img) > 1:
        img = img / 255.0
    img = np.array([img])
    prediction = cnn.predict(img)[0]
    return classes[np.argmax(prediction)]
st.title("Satellite images classification")
st.write("This CNN model is developed by using EuroSAT Dataset")
st.image("EUROSAT.jpg")
col1, col2, col3=st.columns(3)
with col1:
    st.image("Highway_24.jpg", caption='Example Image', use_column_width=True)
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    # Convert the image to array and make a prediction
    image_np = np.array(image)
    prediction = detect(image_np)
    st.write(f"Prediction: {prediction}")
    st.write("Thank You")