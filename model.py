# model.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load  trained model
model = load_model('../image_model.h5')  

# Function to prepare an image for prediction
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  
    img_array = image.img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict_soil_type(img_path):
    img_array = prepare_image(img_path)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    class_label = ['Alluvial soil', 'Black soil', 'Clay soil', 'Red soil']  
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(prediction)
    return predicted_class, confidence
