import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Loading the ML Model
model = load_model("D:/tuansihan/06 Projects/01 Machine Learning/05 Plant Disease Detection/plant_disease_prediction.h5")

# Naming Classes
class_names = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Creating the Web App
st.title("Plant Disease Detection System")
st.markdown("Upload Image of Plant Leaf")

plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Upload')

if submit:
    
    if plant_image is not None:
        
        # Converting file into an array, then an OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Display Image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        
        # Resizing the Image
        opencv_image = cv2.resize(opencv_image, (256, 256))
        
        # Convert Image to 4D
        opencv_image.shape = (1, 256, 256, 3)
        
        # Generate Prediction
        Y_pred = model.predict(opencv_image)
        result = class_names[np.argmax(Y_pred)]
        st.title(str("This is a " + result.split('-')[0] + " leaf with " + result.split('-')[1]))
        